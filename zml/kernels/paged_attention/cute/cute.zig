const stdx = @import("stdx");

const zml = @import("../../../zml.zig");
pub const AttentionOptions = @import("../options.zig").AttentionOptions;
const kernel = @import("attention.zig");
const sparse = @import("sparse_attention.zig");
const sm10x = @import("sm10x/sm10x.zig");
const sm120 = @import("sm120/sm120.zig");

pub const Options = struct {
    batch_size: usize,
    max_num_pages: usize,
    max_seqlen_q: usize,
    is_prefill: bool,

    pub fn isPrefill(self: Options) bool {
        return self.is_prefill;
    }

    pub fn maxNumPages(self: Options) usize {
        return self.max_num_pages;
    }
};

pub const Parameters = struct {
    block_table: zml.Tensor,
    seq_lens: zml.Tensor,
    query_start_len: zml.Tensor,
    options_: Options,

    pub fn init(options_: Options) Parameters {
        return .{
            .block_table = .init(.{ .b = options_.batch_size, .p = options_.max_num_pages }, .i32),
            .seq_lens = .init(.{ .b = options_.batch_size }, .i32),
            .query_start_len = .init(.{ .b = options_.batch_size + 1 }, .i32),
            .options_ = options_,
        };
    }

    pub fn allocationSize(self: Parameters) usize {
        return self.block_table.byteSize() + self.seq_lens.byteSize() + self.query_start_len.byteSize();
    }

    pub fn options(self: Parameters) Options {
        return self.options_;
    }
};

// TODO(Corentin): This should be moved in the platform probably
pub fn multiprocessorCount(platform: *const zml.Platform) usize {
    const attribute = platform.devices[0].pjrt_desc.attribute(platform.pjrt_api, "core_count") orelse return 1;
    if (attribute.int64 <= 0) return 1;
    return @intCast(attribute.int64);
}

test {
    _ = kernel;
    _ = sparse;
    _ = sm10x;
    _ = sm120;
}

pub fn isAvailable(platform: *const zml.Platform) bool {
    if (platform.target != .cuda) return false;
    const cc = zml.platform.cuda.computeCapability(platform) orelse return false;
    return cc.major > 8 or (cc.major == 8 and cc.minor >= 9);
}

fn validateInputs(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor) void {
    stdx.debug.assert(isAvailable(zml.Compiler.current().platform), "CuTe paged attention requires SM89 or newer", .{});
    stdx.debug.assert(q.dtype() == k.dtype() and q.dtype() == v.dtype() and (q.dtype() == .bf16 or q.dtype() == .f16), "CuTe paged attention requires matching BF16 or FP16 Q/K/V", .{});
    stdx.debug.assert(q.dim(.hd) == 32 or q.dim(.hd) == 64 or q.dim(.hd) == 128 or q.dim(.hd) == 256 or q.dim(.hd) == 512, "CuTe paged attention supports head dimensions 32, 64, 128, 256 and 512", .{});
}

fn sparseIndices(q: zml.Tensor, indices: zml.Tensor) zml.Tensor {
    stdx.debug.assert(indices.rank() == 2 and indices.dtype() == .i32, "Sparse indices must be an i32 [query, topk] tensor, got {f}", .{indices.shape()});
    stdx.debug.assert(indices.shape().hasTags(.{ .b, .topk }), "Sparse indices require .b and .topk tags", .{});
    stdx.debug.assert(indices.dim(.b) == q.dim(.b) and indices.dim(.topk) > 0, "Sparse indices {f} must match queries {f} with a nonempty selection capacity", .{ indices.shape(), q.shape() });
    return indices.transpose(.{ .b, .topk });
}

pub fn pagedAttention(parameters: Parameters, q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, opts: AttentionOptions) zml.Tensor {
    validateInputs(q, k, v);
    if (opts.mask == .indices)
        return pagedSparseAttention(parameters, q, k, v, opts);
    if (sm10x.isAvailable(zml.Compiler.current().platform))
        return sm10x.pagedAttention(parameters, q, k, v, opts);
    if (sm120.isAvailable(zml.Compiler.current().platform))
        return sm120.pagedAttention(parameters, q, k, v, opts);
    return pagedAttentionGeneric(parameters, q, k, v, opts);
}

pub fn pagedAttentionGeneric(parameters: Parameters, q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, opts: AttentionOptions) zml.Tensor {
    validateInputs(q, k, v);
    const q_ = q.transpose(.{ .b, .hkv, .hg, .hd });
    const k_ = k.transpose(.{ .page, .k_chunk, .hkv, .hd });
    const v_ = v.transpose(.{ .page, .k_chunk, .hkv, .hd });
    stdx.debug.assert(q_.dim(.hkv) == k_.dim(.hkv) and k_.shape().eql(v_.shape()) and k_.dim(.hd) == q_.dim(.hd) and q_.dim(.hg) > 0, "CuTe paged attention requires aligned local Q/KV heads and matching caches", .{});
    const indices = switch (opts.mask) {
        .indices => |indices| sparseIndices(q_, indices),
        else => zml.Tensor.zeroes(.init(.{ q_.dim(.b), 1 }, .i32)),
    };
    const cfg: kernel.Config = .{
        .head_dim = q_.dim(.hd),
        .tokens = q_.dim(.b),
        .batch = parameters.block_table.dim(.b),
        .kv_heads = k_.dim(.hkv),
        .group = q_.dim(.hg),
        .pages = k_.dim(.page),
        .pages_per_seq = parameters.block_table.dim(.p),
        .page_size = k_.dim(.k_chunk),
        .dtype = zml.kernel.cute.from(q_.dtype()),
        .causal = opts.mask.isCausal(),
        .window = opts.mask.slidingWindow() orelse 0,
        .scale = opts.scale orelse (1 / @sqrt(@as(f32, @floatFromInt(q_.dim(.hd))))),
        .sink = opts.sink != null,
        .indexed = opts.mask == .indices,
        .topk = indices.dim(-1),
    };
    const sink = if (opts.sink) |s| s.convert(.f32).reshape(.{cfg.kv_heads * cfg.group}) else zml.Tensor.zeroes(.init(.{cfg.kv_heads * cfg.group}, .f32));
    return kernel.Kernel.call(.{
        .q = q_,
        .k = k_,
        .v = v_,
        .table = parameters.block_table,
        .lengths = parameters.seq_lens,
        .starts = parameters.query_start_len,
        .sink = sink,
        .indices = indices,
    }, .{ .out = q_.shape() }, .{
        .cfg = cfg,
        .grid = .{ @intCast(cfg.tokens), @intCast(cfg.kv_heads * cfg.group), 1 },
        .block = .{ 32, 1, 1 },
    }).out.transpose(q.shape());
}

fn pagedSparseAttention(parameters: Parameters, q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, opts: AttentionOptions) zml.Tensor {
    const q_ = q.transpose(.{ .b, .hkv, .hg, .hd });
    const k_ = k.transpose(.{ .page, .k_chunk, .hkv, .hd });
    const v_ = v.transpose(.{ .page, .k_chunk, .hkv, .hd });
    stdx.debug.assert(q_.dim(.hkv) == k_.dim(.hkv) and k_.shape().eql(v_.shape()) and k_.dim(.hd) == q_.dim(.hd) and q_.dim(.hg) > 0, "CuTe paged attention requires aligned local Q/KV heads and matching caches", .{});
    const indices = sparseIndices(q_, opts.mask.indices);
    // A short, single-head selection has no KV reuse to amortize MMA tiling.
    if (q_.dim(.hg) == 1 and indices.dim(.topk) <= 8)
        return pagedAttentionGeneric(parameters, q, k, v, opts);
    var cfg: sparse.Config = .{
        .head_dim = q_.dim(.hd),
        .tokens = q_.dim(.b),
        .batch = parameters.block_table.dim(.b),
        .kv_heads = k_.dim(.hkv),
        .group = q_.dim(.hg),
        .pages_per_seq = parameters.block_table.dim(.p),
        .page_size = k_.dim(.k_chunk),
        .topk = indices.dim(.topk),
        .dtype = zml.kernel.cute.from(q_.dtype()),
        .kv_tile = if (!parameters.options_.is_prefill and q_.dim(.hd) <= 256) 16 else if (q_.dim(.hd) == 512) 32 else 64,
        .warps = if (!parameters.options_.is_prefill and q_.dim(.hd) <= 256) @max(1, @divTrunc(q_.dim(.hd), 128)) else 4,
        .scale = opts.scale orelse 1 / @sqrt(@as(f32, @floatFromInt(q_.dim(.hd)))),
        .sink = opts.sink != null,
    };
    const work = @max(1, cfg.tokens * cfg.kv_heads * cfg.headTiles());
    // Narrow decode CTAs keep about 64 FP32 output registers per thread.
    // One selected tile per partition exposes parallelism at small batches.
    const target = @as(i64, @intCast(multiprocessorCount(zml.Compiler.current().platform))) * 16;
    const tiles = @divTrunc(cfg.topk + cfg.kv_tile - 1, cfg.kv_tile);
    var desired: i64 = 1;
    while (desired * work < target) desired *= 2;
    cfg.splits = if (!parameters.options_.is_prefill) @max(1, @min(128, @min(tiles, desired))) else if (cfg.head_dim <= 128) 1 else @max(1, @min(32, @min(@divTrunc(@divTrunc(target, 8) + work - 1, work), @divTrunc(cfg.topk, 128))));
    const sink = if (opts.sink) |s| s.convert(.f32).reshape(.{cfg.kv_heads * cfg.group}) else zml.Tensor.scalar(0, .f32);
    const out = sparse.Kernel.call(.{
        .q = q_,
        .k = k_,
        .v = v_,
        .table = parameters.block_table,
        .lengths = parameters.seq_lens,
        .starts = parameters.query_start_len,
        .sink = sink,
        .indices = indices,
    }, .{ .out = if (cfg.splits > 1) zml.Shape.init(.{ cfg.splits, cfg.tokens, cfg.kv_heads * cfg.group, cfg.head_dim + 2 }, .f32) else q_.shape() }, .{
        .cfg = cfg,
        .grid = .{ @intCast(cfg.tokens), @intCast(cfg.kv_heads * cfg.headTiles()), @intCast(cfg.splits) },
        .block = .{ @intCast(cfg.warps * 32), 1, 1 },
    }).out;
    const result = if (cfg.splits > 1) sparse.Merge.call(.{ .partials = out, .starts = parameters.query_start_len, .sink = sink }, .{ .out = q_.shape() }, .{
        .cfg = cfg,
        .grid = .{ @intCast(cfg.tokens), @intCast(cfg.kv_heads * cfg.group), @intCast(@divTrunc(cfg.head_dim + 63, 64)) },
        .block = .{ 64, 1, 1 },
    }).out else out;
    return result.transpose(q.shape());
}
