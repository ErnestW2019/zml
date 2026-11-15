// Decode CTA width bounds the output fragment at 64 FP32 values per thread.
// Rows are query heads of one token sharing a KV head.
const std = @import("std");
const mlir = @import("mlir");
const llvm = @import("mlir/dialects/llvm");
const vector = @import("mlir/dialects/vector");

const zml = @import("../../../zml.zig");
const cute = zml.kernel.cute;
const B = cute.Builder;
const V = cute.Value;

pub const Config = struct {
    head_dim: i64,
    tokens: i64,
    batch: i64,
    kv_heads: i64,
    group: i64,
    pages_per_seq: i64,
    page_size: i64,
    topk: i64,
    dtype: cute.DType,
    kv_tile: i64,
    warps: i64 = 4,
    scale: f32,
    sink: bool,
    splits: i64 = 1,

    pub fn rows(c: Config) i64 {
        _ = c;
        return 16;
    }
    pub fn headTiles(c: Config) i64 {
        return @divTrunc(c.group + c.rows() - 1, c.rows());
    }
    fn kvBase(c: Config) i64 {
        return c.rows() * c.head_dim * 2;
    }
    fn pBase(c: Config) i64 {
        return c.kvBase() + c.kv_tile * c.head_dim * 2;
    }
    fn redBase(c: Config) i64 {
        return c.pBase() + c.rows() * c.kv_tile * 2;
    }
    fn indexBase(c: Config) i64 {
        return c.redBase() + 2 * 64 * 4;
    }
    fn sharedBytes(c: Config) i64 {
        return c.indexBase() + c.kv_tile * 8;
    }
};

pub const Kernel = cute.Kernel(Config, .{
    .name = "paged_attention_cute_indexed_grouped",
    .inputs = &.{ "q", "k", "v", "table", "lengths", "starts", "sink", "indices" },
    .outputs = &.{"out"},
    .run = run,
});

fn flatten(comptime n: usize, values: [n][4]V) [n * 4]V {
    var result: [n * 4]V = undefined;
    for (0..n) |i| @memcpy(result[i * 4 ..][0..4], &values[i]);
    return result;
}

fn packAcc(b: *B, values: [4]V) V {
    return b.emit(vector.from_elements(b.ctx, &.{ values[0].inner, values[1].inner, values[2].inner, values[3].inner }, .vector(&.{4}, .float(b.ctx, .f32)), b.loc()));
}

fn loadI32(b: *B, ptr: V, index: V) V {
    return b.loadWords(1, ptr.add(index.to(.i64).mul(4)), .global, .{})[0];
}

// XOR whole 16-byte vectors so cp.async and ldmatrix agree on bank layout.
fn swizzle(row: V, col: V, width: i64) V {
    return row.mul(width).add(col.bitXor(row.rem(@min(8, @divExact(width, 8))).mul(8))).mul(2);
}

fn unpackStruct(b: *B, comptime n: usize, value: V, dtype: cute.DType) [n]V {
    var result: [n]V = undefined;
    for (&result, 0..) |*r, i| r.* = b.emit(llvm.extractvalue(b.ctx, value.inner, dtype.toMlir(b.ctx), &.{@intCast(i)}, b.loc()));
    return result;
}

fn matrixLoad(b: *B, comptime n: usize, comptime transpose: bool, address: V) [n]V {
    const assembly = std.fmt.comptimePrint("ldmatrix.sync.aligned.m8n8.x{d}{s}.shared.b16 {{", .{ n, if (transpose) ".trans" else "" }) ++
        (if (n == 4) "$0, $1, $2, $3}, [$4];" else "$0, $1}, [$2];");
    const result = b.emit(llvm.inline_asm(b.ctx, assembly, &.{address.inner}, .{ .result_type = llvm.structType(b.ctx, &@as([n]*const mlir.Type, @splat(.int(b.ctx, .i32)))), .constraints = if (n == 4) "=r,=r,=r,=r,r" else "=r,=r,r", .has_side_effects = true }, b.loc()));
    return unpackStruct(b, n, result, .i32);
}

fn mma(b: *B, comptime dtype: cute.DType, a: [4]V, k: [2]V, acc: [4]V) [4]V {
    const assembly = "mma.sync.aligned.m16n8k16.row.col.f32." ++ @tagName(dtype) ++ "." ++ @tagName(dtype) ++ ".f32 " ++
        "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};";
    const result = b.emit(llvm.inline_asm(b.ctx, assembly, &.{ a[0].inner, a[1].inner, a[2].inner, a[3].inner, k[0].inner, k[1].inner, acc[0].inner, acc[1].inner, acc[2].inner, acc[3].inner }, .{ .result_type = llvm.structType(b.ctx, &@as([4]*const mlir.Type, @splat(.float(b.ctx, .f32)))), .constraints = "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f" }, b.loc()));
    return unpackStruct(b, 4, result, .f32);
}

fn run(b: *B, c: Config) cute.FinishError!void {
    switch (c.head_dim) {
        inline 32, 64, 128, 256 => |d| switch (c.warps) {
            inline 1, 2, 4 => |w| try runDimension(b, c, d, w),
            else => unreachable,
        },
        512 => {
            const d = 512;
            std.debug.assert(c.warps == 4);
            try runDimension(b, c, d, 4);
        },
        else => unreachable,
    }
}

fn runDimension(b: *B, c: Config, comptime d: usize, comptime w: usize) cute.FinishError!void {
    switch (c.kv_tile) {
        16 => if (w <= 2) switch (c.dtype) {
            inline .bf16, .f16 => |dt| try runTile(b, c, d, w, 16, dt),
            else => unreachable,
        } else unreachable,
        inline 32, 64 => |n| switch (c.dtype) {
            inline .bf16, .f16 => |dt| try runTile(b, c, d, w, n, dt),
            else => unreachable,
        },
        else => unreachable,
    }
}

const TileContext = struct {
    b: *B,
    c: Config,
    q: V,
    k: V,
    v: V,
    table: V,
    indices: V,
    sink: V,
    out: V,
    shared: V,
    tid: V,
    lane: V,
    warp: V,
    seq: V,
    head: V,
    first_row: V,
    token: V,
    klen: V,
    split: V,

    // Resolve each selected slot once, then share its physical page across
    // every head, feature vector, and both K/V loads. Invalid logical indices
    // never dereference the page table; cp.async zero-fills their K/V rows.
    fn gather(t: TileContext, key_base: V) void {
        const b = t.b;
        const c = t.c;
        var selected = b.openFor(t.tid, c.kv_tile, c.warps * 32, .{});
        {
            const address = t.shared.add(c.indexBase()).add(selected.iv.mul(8));
            b.storeWords(1, address, .shared, .{b.constant(.i32, -1)});
            b.storeWords(1, address.add(4), .shared, .{b.constant(.i32, 0)});
            const slot = key_base.add(selected.iv);
            var in_range = b.openIf(slot.lt(c.topk));
            {
                const key = loadI32(b, t.indices, t.token.to(.i64).mul(c.topk).add(slot.to(.i64)));
                var valid = b.openIf(key.ge(0).bitAnd(key.lt(t.klen)));
                {
                    const page = loadI32(b, t.table, t.seq.mul(c.pages_per_seq).add(key.div(c.page_size)));
                    b.storeWords(1, address, .shared, .{page});
                    b.storeWords(1, address.add(4), .shared, .{key.rem(c.page_size)});
                    valid.close();
                }
                in_range.close();
            }
            selected.close();
        }
        b.syncThreads();
    }

    fn validSlot(t: TileContext, row: V) V {
        return t.b.loadWords(1, t.shared.add(t.c.indexBase()).add(row.mul(8)), .shared, .{})[0].ge(0);
    }

    fn loadKV(t: TileContext, source: V) void {
        const b = t.b;
        const c = t.c;
        var copy = b.openFor(t.tid, c.kv_tile * @divExact(c.head_dim, 8), c.warps * 32, .{});
        {
            const row = copy.iv.div(@divExact(c.head_dim, 8));
            const col = copy.iv.rem(@divExact(c.head_dim, 8)).mul(8);
            const address = t.shared.add(c.indexBase()).add(row.mul(8));
            const mapping = b.loadWords(2, address, .shared, .{});
            const page = mapping[0];
            const within = mapping[1];
            const valid = page.ge(0);
            const safe_page = b.select(valid, page, b.constant(.i32, 0));
            const offset = safe_page.to(.i64).mul(c.page_size).add(within.to(.i64)).mul(c.kv_heads).add(t.head.to(.i64)).mul(c.head_dim).add(col.to(.i64));
            b.cpAsync16(t.shared.add(c.kvBase()).add(swizzle(row, col, c.head_dim)), source.add(offset.mul(2)), valid);
            copy.close();
        }
    }

    fn publishReduction(t: TileContext, row: V, value: V, comptime maximum: bool) void {
        const b = t.b;
        var v = value;
        inline for (.{ 2, 1 }) |step| v = if (maximum) v.maximum(b.shuffleXor(v, step)) else v.add(b.shuffleXor(v, step));
        const address = t.shared.add(t.c.redBase() + (if (maximum) @as(i64, 0) else 256)).add(t.warp.mul(16).add(row).mul(4));
        var leader = b.openIf(t.lane.rem(4).eq(0));
        {
            b.storeWords(1, address, .shared, .{v.bitCast(.i32)});
            leader.close();
        }
        // Callers synchronize once both row halves have been published.
    }

    fn readReduction(t: TileContext, comptime w: usize, row: V, comptime maximum: bool) V {
        const b = t.b;
        const base = t.shared.add(t.c.redBase() + (if (maximum) @as(i64, 0) else 256)).add(t.warp.div(w).mul(w * 16).add(row).mul(4));
        var result = b.loadWords(1, base, .shared, .{})[0].bitCast(.f32);
        for (1..w) |i| {
            const v = b.loadWords(1, base.add(i * 16 * 4), .shared, .{})[0].bitCast(.f32);
            result = if (maximum) result.maximum(v) else result.add(v);
        }
        return result;
    }

    fn store(t: TileContext, row: V, dim: V, value: V) void {
        const c = t.c;
        var offset = t.token.to(.i64).mul(c.kv_heads).add(t.head.to(.i64)).mul(c.group).add(row.to(.i64));
        if (c.splits > 1) {
            offset = t.split.to(.i64).mul(c.tokens * c.kv_heads * c.group).add(offset).mul(c.head_dim + 2).add(dim.to(.i64));
            t.b.storeWords(1, t.out.add(offset.mul(4)), .global, .{value.bitCast(.i32)});
        } else {
            offset = offset.mul(c.head_dim).add(dim.to(.i64));
            t.b.storeHalf(t.out.add(offset.mul(2)), .global, value.to(c.dtype).bitCast(.i16).to(.i32));
        }
    }
};

fn runTile(b: *B, c: Config, comptime d: usize, comptime w: usize, comptime n: usize, comptime dt: cute.DType) cute.FinishError!void {
    std.debug.assert(c.sharedBytes() <= 99 * 1024);
    const a = try b.declareArgs(.{
        .q = .{ .ptr = dt },
        .k = .{ .ptr = dt },
        .v = .{ .ptr = dt },
        .table = .{ .ptr = cute.DType.i32 },
        .lengths = .{ .ptr = cute.DType.i32 },
        .starts = .{ .ptr = cute.DType.i32 },
        .sink = .{ .ptr = cute.DType.f32 },
        .indices = .{ .ptr = cute.DType.i32 },
        .out = .{ .ptr = if (c.splits > 1) cute.DType.f32 else dt },
    });
    b.setFunctionAttribute(.named(b.ctx, "cu_attrs", .dict(b.ctx, &.{.named(b.ctx, "max_dynamic_shared_size_bytes", b.devMaxSharedMemoryOptin())})));
    const token = b.blockIdx().x;
    const first_row = b.blockIdx().y.rem(c.headTiles()).mul(c.rows());
    const head = b.blockIdx().y.div(c.headTiles());
    const starts = b.ptrToInt(a.starts, .i64);
    const active_tokens = loadI32(b, starts, b.constant(.i32, c.batch));
    // Unused capacity is observable in the public tensor and must be zero.
    var inactive = b.openIf(token.ge(active_tokens));
    {
        const out = b.ptrToInt(a.out, .i64);
        var dims = b.openFor(b.threadIdx().x, c.rows() * (c.head_dim + (if (c.splits > 1) @as(i64, 2) else 0)), c.warps * 32, .{});
        {
            const width = c.head_dim + (if (c.splits > 1) @as(i64, 2) else 0);
            const row = first_row.add(dims.iv.div(width));
            const dim = dims.iv.rem(width);
            var valid = b.openIf(row.lt(c.group));
            {
                var offset = token.to(.i64).mul(c.kv_heads).add(head.to(.i64)).mul(c.group).add(row.to(.i64));
                if (c.splits > 1) {
                    offset = b.blockIdx().z.to(.i64).mul(c.tokens * c.kv_heads * c.group).add(offset).mul(width).add(dim.to(.i64));
                    const value = b.select(dim.eq(c.head_dim), b.constant(.f32, -std.math.inf(f32)), b.constant(.f32, 0));
                    b.storeWords(1, out.add(offset.mul(4)), .global, .{value.bitCast(.i32)});
                } else {
                    b.storeHalf(out.add(offset.mul(width).add(dim.to(.i64)).mul(2)), .global, b.constant(.i32, 0));
                }
                valid.close();
            }
            dims.close();
        }
        inactive.close();
    }
    var active = b.openIf(token.lt(active_tokens));
    {
        const tid = b.threadIdx().x;
        // Upper-bound lookup also handles inactive sequences with equal starts.
        var sequence = b.openWhile(.{ b.constant(.i32, 0), b.constant(.i32, c.batch) }, .{ .i32, .i32 });
        {
            sequence.yieldBefore(sequence.before_carried[0].lt(sequence.before_carried[1]), sequence.before_carried);
        }
        {
            const left = sequence.after_carried[0];
            const right = sequence.after_carried[1];
            const mid = left.add(right).div(2);
            const before = loadI32(b, starts, mid).le(token);
            sequence.yieldAfter(.{ b.select(before, mid.add(1), left), b.select(before, right, mid) });
        }
        const seq = sequence.results[0].sub(1);
        const t: TileContext = .{
            .b = b,
            .c = c,
            .q = b.ptrToInt(a.q, .i64),
            .k = b.ptrToInt(a.k, .i64),
            .v = b.ptrToInt(a.v, .i64),
            .table = b.ptrToInt(a.table, .i64),
            .indices = b.ptrToInt(a.indices, .i64),
            .sink = b.ptrToInt(a.sink, .i64),
            .out = b.ptrToInt(a.out, .i64),
            .shared = b.ptrToInt(b.allocSmem(.i8, c.sharedBytes(), 128), .i32),
            .tid = tid,
            .lane = tid.rem(32),
            .warp = tid.div(32),
            .seq = seq,
            .head = head,
            .first_row = first_row,
            .token = token,
            .klen = loadI32(b, b.ptrToInt(a.lengths, .i64), seq),
            .split = b.blockIdx().z,
        };
        var copy_q = b.openFor(tid, c.rows() * (d / 8), c.warps * 32, .{});
        {
            const row = copy_q.iv.div(d / 8);
            const col = copy_q.iv.rem(d / 8).mul(8);
            const packed_row = first_row.add(row);
            const valid = packed_row.lt(c.group);
            const safe_row = b.select(valid, packed_row, b.constant(.i32, 0));
            const offset = token.to(.i64).mul(c.kv_heads).add(t.head.to(.i64)).mul(c.group).add(safe_row.to(.i64)).mul(d).add(col.to(.i64));
            b.cpAsync16(t.shared.add(swizzle(row, col, d)), t.q.add(offset.mul(2)), valid);
            copy_q.close();
        }
        // Split the selected-slot list, not the logical KV timeline. Repeated
        // indices remain repeated softmax columns, including across splits.
        const tiles = @divTrunc(c.topk + n - 1, n);
        const lo = t.split.mul(tiles).div(c.splits);
        const hi = t.split.add(1).mul(tiles).div(c.splits);
        const row_base = t.warp.div(w).mul(16);
        const key_part = t.warp.rem(w).mul(n / w);
        const dim_part = t.warp.rem(w).mul(d / w);
        const row0 = t.lane.div(4);
        const col0 = t.lane.rem(4).mul(2);
        const zero = b.constant(.f32, 0);
        const neg_inf = b.constant(.f32, -std.math.inf(f32));
        const nf = n / w / 8;
        const of = d / w / 8;
        const count = 4 + of;
        var initial: [count]V = @splat(zero);
        initial[0] = neg_inf;
        initial[1] = neg_inf;
        for (initial[4..]) |*v| v.* = packAcc(b, .{ zero, zero, zero, zero });
        // Q can load while the first selected indices and pages are resolved.
        var loop = b.openFor(lo, hi, 1, initial);
        {
            const key_base = loop.iv.mul(n);
            t.gather(key_base);
            t.loadKV(t.k);
            b.cpAsyncWaitAll();
            b.syncThreads();
            var scores: [nf][4]V = @splat(@splat(zero));
            // Keep only MMA operands live in registers; Q remains in shared
            // memory so D512 does not consume the entire per-thread register file.
            var qk = b.openFor(0, d, 16, @as([nf * 4]V, @splat(zero)));
            {
                const qa = matrixLoad(b, 4, false, t.shared.add(swizzle(row_base.add(t.lane.rem(16)), qk.iv.add(t.lane.div(16).mul(8)), d)));
                for (0..nf) |i| {
                    const kb = matrixLoad(b, 2, false, t.shared.add(c.kvBase()).add(swizzle(key_part.add(i * 8).add(t.lane.rem(8)), qk.iv.add(t.lane.div(8).rem(2).mul(8)), d)));
                    scores[i] = mma(b, dt, qa, kb, qk.carried[i * 4 ..][0..4].*);
                }
                qk.yield(flatten(nf, scores));
            }
            for (0..nf) |i| scores[i] = qk.results[i * 4 ..][0..4].*;
            var local_max: [2]V = @splat(neg_inf);
            for (0..nf) |i| {
                for (0..4) |j| {
                    const row = first_row.add(row_base).add(row0).add((j / 2) * 8);
                    const slot = key_part.add(i * 8).add(col0).add(j % 2);
                    const valid = row.lt(c.group).bitAnd(t.validSlot(slot));
                    scores[i][j] = b.select(valid, scores[i][j].mul(c.scale * std.math.log2e), neg_inf);
                    local_max[j / 2] = local_max[j / 2].maximum(scores[i][j]);
                }
            }
            for (0..2) |h| t.publishReduction(row0.add(h * 8), local_max[h], true);
            b.syncThreads();
            // All K reads have completed. Reuse its allocation for V, and
            // overlap these global loads with softmax and probability stores.
            t.loadKV(t.v);
            var m: [2]V = undefined;
            var alpha: [2]V = undefined;
            var local_sum: [2]V = @splat(zero);
            for (0..2) |h| {
                m[h] = loop.carried[h].maximum(t.readReduction(w, row0.add(h * 8), true));
                // Empty/fully masked row tiles have m=-inf and denominator 0.
                alpha[h] = b.exp2Approx(loop.carried[h].sub(m[h].maximum(-std.math.floatMax(f32))));
            }
            for (0..nf) |i| {
                for (0..4) |j| {
                    scores[i][j] = b.exp2Approx(scores[i][j].sub(m[j / 2].maximum(-std.math.floatMax(f32))));
                    local_sum[j / 2] = local_sum[j / 2].add(scores[i][j]);
                }
                for (0..2) |h| {
                    const row = row_base.add(row0).add(h * 8);
                    const col = key_part.add(i * 8).add(col0);
                    b.storeWords(1, t.shared.add(c.pBase()).add(swizzle(row, col, n)), .shared, .{b.packHalf2(scores[i][h * 2], scores[i][h * 2 + 1], dt)});
                }
            }
            for (0..2) |h| t.publishReduction(row0.add(h * 8), local_sum[h], false);
            b.cpAsyncWaitAll();
            b.syncThreads();
            var next: [count]V = undefined;
            for (0..2) |h| {
                next[h] = m[h];
                next[h + 2] = loop.carried[h + 2].mul(alpha[h]).add(t.readReduction(w, row0.add(h * 8), false));
            }
            var acc: [of][4]V = undefined;
            for (0..of) |i| for (0..4) |j| {
                acc[i][j] = b.vectorExtract(loop.carried[4 + i], @intCast(j), .f32).mul(alpha[j / 2]);
            };
            var packed_acc: [of]V = undefined;
            for (0..of) |i| packed_acc[i] = packAcc(b, acc[i]);
            var pv = b.openFor(0, n, 16, packed_acc);
            {
                const pa = matrixLoad(b, 4, false, t.shared.add(c.pBase()).add(swizzle(row_base.add(t.lane.rem(16)), pv.iv.add(t.lane.div(16).mul(8)), n)));
                for (0..of) |i| {
                    const vb = matrixLoad(b, 2, true, t.shared.add(c.kvBase()).add(swizzle(pv.iv.add(t.lane.rem(16)), dim_part.add(i * 8), d)));
                    var prev: [4]V = undefined;
                    for (0..4) |j| prev[j] = b.vectorExtract(pv.carried[i], @intCast(j), .f32);
                    acc[i] = mma(b, dt, pa, vb, prev);
                    packed_acc[i] = packAcc(b, acc[i]);
                }
                pv.yield(packed_acc);
            }
            @memcpy(next[4..], &pv.results);
            b.syncThreads();
            loop.yield(next);
        }
        if (c.splits > tiles) b.cpAsyncWaitAll();
        for (0..2) |h| {
            const row = first_row.add(row_base).add(row0).add(h * 8);
            var valid = b.openIf(row.lt(c.group));
            {
                const m = loop.results[h];
                const l = loop.results[h + 2];
                var norm = b.select(l.gt(0), b.rcpApprox(l), zero);
                if (c.splits == 1 and c.sink) {
                    const sink = loadI32(b, t.sink, t.head.mul(c.group).add(row)).bitCast(.f32).mul(std.math.log2e);
                    const maximum = m.maximum(sink).maximum(-std.math.floatMax(f32));
                    const weight = b.exp2Approx(m.sub(maximum));
                    const denominator = l.mul(weight).add(b.exp2Approx(sink.sub(maximum)));
                    norm = weight.mul(b.select(denominator.gt(0), b.rcpApprox(denominator), zero));
                }
                for (0..of) |i| for (0..2) |j| {
                    const value = b.vectorExtract(loop.results[4 + i], @intCast(h * 2 + j), .f32);
                    t.store(row, dim_part.add(i * 8).add(col0).add(j), if (c.splits > 1) value else value.mul(norm));
                };
                if (c.splits > 1) {
                    var leader = b.openIf(t.warp.rem(w).eq(0).bitAnd(t.lane.rem(4).eq(0)));
                    {
                        t.store(row, b.constant(.i32, d), m);
                        t.store(row, b.constant(.i32, d + 1), l);
                        leader.close();
                    }
                }
                valid.close();
            }
        }
        active.close();
    }
}

pub const Merge = cute.Kernel(Config, .{
    .name = "paged_attention_cute_indexed_grouped_merge",
    .inputs = &.{ "partials", "starts", "sink" },
    .outputs = &.{"out"},
    .run = merge,
});

fn merge(b: *B, c: Config) cute.FinishError!void {
    const a = try b.declareArgs(.{
        .partials = .{ .tensor = .{ .dtype = .f32, .shape = &.{ c.splits, c.tokens, c.kv_heads * c.group, c.head_dim + 2 } } },
        .starts = .{ .tensor = .{ .dtype = .i32, .shape = &.{c.batch + 1} } },
        .sink = .{ .tensor = .{ .dtype = .f32, .shape = &.{c.kv_heads * c.group} } },
        .out = .{ .tensor = .{ .dtype = c.dtype, .shape = &.{ c.tokens, c.kv_heads * c.group, c.head_dim } } },
    });
    const tile = @min(@as(i64, 64), c.head_dim);
    const dim_base = b.blockIdx().z.mul(tile);
    const token = b.blockIdx().x;
    const head = b.blockIdx().y;
    var inactive = b.openIf(token.ge(a.starts.get(.{c.batch})));
    {
        var dims = b.openFor(b.threadIdx().x, tile, 64, .{});
        {
            a.out.set(.{ token, head, dim_base.add(dims.iv) }, b.constant(c.dtype, 0));
            dims.close();
        }
        inactive.close();
    }
    var valid = b.openIf(token.lt(a.starts.get(.{c.batch})));
    {
        const zero = b.constant(.f32, 0);
        const sink = if (c.sink) a.sink.get(.{head}).mul(std.math.log2e) else b.constant(.f32, -std.math.inf(f32));
        var maximum = b.openFor(b.threadIdx().x.rem(32), c.splits, 32, .{sink});
        {
            maximum.yield(.{maximum.carried[0].maximum(a.partials.get(.{ maximum.iv, token, head, c.head_dim }))});
        }
        var maximum_value = maximum.results[0];
        inline for (.{ 16, 8, 4, 2, 1 }) |step| maximum_value = maximum_value.maximum(b.shuffleXor(maximum_value, step));
        const m = maximum_value.maximum(-std.math.floatMax(f32));
        var dims = b.openFor(b.threadIdx().x, tile, 64, .{});
        {
            // Expose independent loads and a balanced sum within each chunk.
            // A bounded chunk avoids register spills for long selections.
            const chunk = @min(@as(i64, 16), c.splits);
            var sum = b.openFor(0, c.splits, chunk, .{ zero, if (c.sink) b.exp2Approx(sink.sub(m)) else zero });
            {
                var values: [16]V = @splat(zero);
                var denominators: [16]V = @splat(zero);
                for (0..@as(usize, @intCast(chunk))) |i| {
                    const part = sum.iv.add(i);
                    const valid_part = part.lt(c.splits);
                    const safe_part = part.minimum(c.splits - 1);
                    const weight = b.select(valid_part, b.exp2Approx(a.partials.get(.{ safe_part, token, head, c.head_dim }).sub(m)), zero);
                    values[i] = weight.mul(a.partials.get(.{ safe_part, token, head, dim_base.add(dims.iv) }));
                    denominators[i] = weight.mul(a.partials.get(.{ safe_part, token, head, c.head_dim + 1 }));
                }
                var stride: usize = 1;
                while (stride < 16) : (stride *= 2) {
                    var i: usize = 0;
                    while (i < 16) : (i += 2 * stride) {
                        values[i] = values[i].add(values[i + stride]);
                        denominators[i] = denominators[i].add(denominators[i + stride]);
                    }
                }
                sum.yield(.{ sum.carried[0].add(values[0]), sum.carried[1].add(denominators[0]) });
            }
            a.out.set(.{ token, head, dim_base.add(dims.iv) }, sum.results[0].mul(b.select(sum.results[1].gt(0), b.rcpApprox(sum.results[1]), zero)).to(c.dtype));
            dims.close();
        }
        valid.close();
    }
}

test "indexed grouped attention emits all supported head dimensions and dtypes" {
    for ([_]i64{ 32, 64, 128, 256, 512 }) |head_dim| for ([_]cute.DType{ .bf16, .f16 }) |dtype| {
        const cfg: Config = .{
            .head_dim = head_dim,
            .tokens = 7,
            .batch = 3,
            .kv_heads = 2,
            .group = 19,
            .pages_per_seq = 64,
            .page_size = 16,
            .topk = 37,
            .dtype = dtype,
            .kv_tile = if (head_dim == 512) 32 else 64,
            .scale = 1 / @sqrt(@as(f32, @floatFromInt(head_dim))),
            .sink = false,
        };
        try std.testing.expectEqual(@as(i64, 2), cfg.headTiles());
        try std.testing.expect(cfg.sharedBytes() <= 99 * 1024);
        const ir = try Kernel.emit(std.testing.allocator, cfg, .{ 128, 1, 1 });
        defer std.testing.allocator.free(ir);
        try std.testing.expect(std.mem.indexOf(u8, ir, if (dtype == .bf16) "mma.sync.aligned.m16n8k16.row.col.f32.bf16" else "mma.sync.aligned.m16n8k16.row.col.f32.f16") != null);
        try std.testing.expect(std.mem.indexOf(u8, ir, "cp.async") != null);
        try std.testing.expect(std.mem.indexOf(u8, ir, "scf.if") != null);
    };
}

test "indexed grouped attention emits split softmax and sink merge" {
    for ([_]cute.DType{ .bf16, .f16 }) |dtype| {
        const cfg: Config = .{
            .head_dim = 512,
            .tokens = 7,
            .batch = 3,
            .kv_heads = 2,
            .group = 19,
            .pages_per_seq = 64,
            .page_size = 16,
            .topk = 257,
            .dtype = dtype,
            .kv_tile = 32,
            .scale = 0.125,
            .sink = true,
            .splits = 3,
        };
        const ir = try Kernel.emit(std.testing.allocator, cfg, .{ 128, 1, 1 });
        defer std.testing.allocator.free(ir);
        const merge_ir = try Merge.emit(std.testing.allocator, cfg, .{ 128, 1, 1 });
        defer std.testing.allocator.free(merge_ir);
        try std.testing.expect(std.mem.indexOf(u8, ir, "ldmatrix.sync.aligned.m8n8.x2.trans") != null);
        try std.testing.expect(std.mem.indexOf(u8, merge_ir, "scf.for") != null);
    }
}

test "indexed grouped attention emits single warp D32 D64 D128" {
    for ([_]i64{ 32, 64, 128 }) |head_dim| for ([_]cute.DType{ .bf16, .f16 }) |dtype| for ([_]i64{ 16, 32, 64 }) |tile| {
        const cfg: Config = .{
            .head_dim = head_dim,
            .tokens = 7,
            .batch = 3,
            .kv_heads = 2,
            .group = 19,
            .pages_per_seq = 64,
            .page_size = 16,
            .topk = 65,
            .dtype = dtype,
            .kv_tile = tile,
            .warps = 1,
            .scale = 0.125,
            .sink = true,
            .splits = 3,
        };
        const ir = try Kernel.emit(std.testing.allocator, cfg, .{ 32, 1, 1 });
        defer std.testing.allocator.free(ir);
        try std.testing.expect(std.mem.indexOf(u8, ir, "mma.sync.aligned.m16n8k16") != null);
        try std.testing.expect(std.mem.indexOf(u8, ir, "ldmatrix.sync.aligned.m8n8.x2.trans") != null);
    };
}
