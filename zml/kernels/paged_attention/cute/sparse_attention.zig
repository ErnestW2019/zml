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
    decode: bool = false,
    merge_parts: i64 = 8,
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
    pub fn partialWidth(c: Config) i64 {
        // Align each split to a 32-byte sector, including max/denominator.
        return c.head_dim + 8;
    }
    fn pBase(c: Config) i64 {
        return c.kvBase() + c.kv_tile * c.head_dim * 2;
    }
    fn redBase(c: Config) i64 {
        const key_warps = @min(c.warps, @min(@divTrunc(c.kv_tile, 8), 4));
        const feature_parts = @divExact(c.warps, key_warps);
        // Wide-head QK partials share this allocation with the later FP16 P.
        return c.pBase() + c.rows() * c.kv_tile * (if (feature_parts > 1) feature_parts * 4 else 2);
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

// Small MHA selections use SIMT: four warps evaluate independent selected
// slots, then reduce their weighted values within one CTA and one launch.
pub const Tiny = cute.Kernel(Config, .{
    .name = "paged_attention_cute_indexed_tiny",
    .inputs = &.{ "q", "k", "v", "table", "lengths", "starts", "sink", "indices" },
    .outputs = &.{"out"},
    .run = runTiny,
});

// In a decode-only batch each sequence owns zero or one query. Equal total
// active tokens and sequence count proves token == sequence; spare capacity
// and inactive sequences retain the general upper-bound lookup.
fn findSequence(b: *B, c: Config, starts: V, token: V, active_tokens: V) V {
    var fast = b.openIfElse(if (c.decode) active_tokens.eq(c.batch) else b.constant(.i1, 0), .{.i32});
    fast.yieldThen(.{token});
    var sequence = b.openWhile(.{ b.constant(.i32, 0), b.constant(.i32, c.batch) }, .{ .i32, .i32 });
    sequence.yieldBefore(sequence.before_carried[0].lt(sequence.before_carried[1]), sequence.before_carried);
    const left = sequence.after_carried[0];
    const right = sequence.after_carried[1];
    const mid = left.add(right).div(2);
    const before = loadI32(b, starts, mid).le(token);
    sequence.yieldAfter(.{ b.select(before, mid.add(1), left), b.select(before, right, mid) });
    fast.yieldElse(.{sequence.results[0].sub(1)});
    return fast.results[0];
}

fn loadHalf(b: *B, address: V, comptime dt: cute.DType) V {
    return b.emit(llvm.load(b.ctx, b.addressPtr(address, .global).inner, dt.toMlir(b.ctx), .{ .alignment = 2 }, b.loc())).to(.f32);
}

fn runTiny(b: *B, c: Config) cute.FinishError!void {
    std.debug.assert(c.group == 1 and c.topk <= 8);
    switch (c.head_dim) {
        inline 32, 64, 128, 256, 512 => |d| switch (c.dtype) {
            inline .bf16, .f16 => |dt| try runTinyDimension(b, c, d, dt),
            else => unreachable,
        },
        else => unreachable,
    }
}

fn runTinyDimension(b: *B, c: Config, comptime d: usize, comptime dt: cute.DType) cute.FinishError!void {
    const chunks = d / 32;
    const warps: usize = if (c.topk > 4) 8 else 4;
    const slots: usize = 8 / warps;
    const a = try b.declareArgs(.{
        .q = .{ .ptr = dt },
        .k = .{ .ptr = dt },
        .v = .{ .ptr = dt },
        .table = .{ .ptr = cute.DType.i32 },
        .lengths = .{ .ptr = cute.DType.i32 },
        .starts = .{ .ptr = cute.DType.i32 },
        .sink = .{ .ptr = cute.DType.f32 },
        .indices = .{ .ptr = cute.DType.i32 },
        .out = .{ .ptr = dt },
    });
    const token = b.blockIdx().x;
    const head = b.blockIdx().y;
    const tid = b.threadIdx().x;
    const lane = tid.rem(32);
    const warp = tid.div(32);
    const zero = b.constant(.f32, 0);
    const neg_inf = b.constant(.f32, -std.math.inf(f32));
    const starts = b.ptrToInt(a.starts, .i64);
    const active_tokens = loadI32(b, starts, b.constant(.i32, c.batch));
    const out = b.ptrToInt(a.out, .i64).add(token.to(.i64).mul(c.kv_heads).add(head.to(.i64)).mul(d * 2));
    var inactive = b.openIf(token.ge(active_tokens));
    {
        var dims = b.openFor(tid, d, warps * 32, .{});
        b.storeHalf(out.add(dims.iv.to(.i64).mul(2)), .global, b.constant(.i32, 0));
        dims.close();
        inactive.close();
    }
    var active = b.openIf(token.lt(active_tokens));
    {
        const seq = findSequence(b, c, starts, token, active_tokens);
        const len = loadI32(b, b.ptrToInt(a.lengths, .i64), seq);
        const q = b.ptrToInt(a.q, .i64).add(token.to(.i64).mul(c.kv_heads).add(head.to(.i64)).mul(d * 2));
        const shared = b.ptrToInt(b.allocSmem(.i8, @intCast(32 + warps * d * 4), 16), .i32);
        var query: [chunks]V = undefined;
        for (&query, 0..) |*value, i| value.* = loadHalf(b, q.add(lane.add(i * 32).to(.i64).mul(2)), dt);
        var rows: [2]V = undefined;
        var valid: [2]V = undefined;
        for (0..slots) |i| {
            const slot = warp.add(i * warps);
            var selected = b.openIfElse(slot.lt(c.topk), .{.i32});
            selected.yieldThen(.{loadI32(b, b.ptrToInt(a.indices, .i64), token.to(.i64).mul(c.topk).add(slot.to(.i64)))});
            selected.yieldElse(.{b.constant(.i32, -1)});
            const key = selected.results[0];
            valid[i] = key.ge(0).bitAnd(key.lt(len));
            var entry = b.openIfElse(valid[i], .{ .f32, .i64 });
            {
                const page = loadI32(b, b.ptrToInt(a.table, .i64), seq.mul(c.pages_per_seq).add(key.div(c.page_size)));
                const row = page.to(.i64).mul(c.page_size).add(key.rem(c.page_size).to(.i64)).mul(c.kv_heads).add(head.to(.i64)).mul(d * 2);
                var score = zero;
                for (query, 0..) |value, j| score = score.add(value.mul(loadHalf(b, b.ptrToInt(a.k, .i64).add(row).add(lane.add(j * 32).to(.i64).mul(2)), dt)));
                inline for (.{ 16, 8, 4, 2, 1 }) |step| score = score.add(b.shuffleXor(score, step));
                entry.yieldThen(.{ score.mul(c.scale * std.math.log2e), row });
            }
            entry.yieldElse(.{ neg_inf, b.constant(.i64, 0) });
            rows[i] = entry.results[1];
            var leader = b.openIf(lane.eq(0));
            b.storeWords(1, shared.add(slot.mul(4)), .shared, .{entry.results[0].bitCast(.i32)});
            leader.close();
        }
        b.syncThreads();
        const sink = if (c.sink) b.loadWords(1, b.ptrToInt(a.sink, .i64).add(head.to(.i64).mul(4)), .global, .{})[0].bitCast(.f32).mul(std.math.log2e) else neg_inf;
        var maximum = sink;
        var scores: [8]V = undefined;
        for (&scores, 0..) |*score, i| {
            score.* = b.loadWords(1, shared.add(i * 4), .shared, .{})[0].bitCast(.f32);
            maximum = maximum.maximum(score.*);
        }
        maximum = maximum.maximum(b.constant(.f32, -std.math.floatMax(f32)));
        var denominator = if (c.sink) b.exp2Approx(sink.sub(maximum)) else zero;
        for (scores) |score| denominator = denominator.add(b.exp2Approx(score.sub(maximum)));
        const norm = b.select(denominator.gt(0), b.rcpApprox(denominator), zero);
        var output: [chunks]V = @splat(zero);
        for (0..slots) |i| {
            const score = b.loadWords(1, shared.add(warp.add(i * warps).mul(4)), .shared, .{})[0].bitCast(.f32);
            const weight = b.exp2Approx(score.sub(maximum)).mul(norm);
            var entry = b.openIfElse(valid[i], @as([chunks]V, @splat(zero)));
            var values: [chunks]V = undefined;
            for (&values, 0..) |*value, j| value.* = output[j].add(weight.mul(loadHalf(b, b.ptrToInt(a.v, .i64).add(rows[i]).add(lane.add(j * 32).to(.i64).mul(2)), dt)));
            entry.yieldThen(values);
            entry.yieldElse(output);
            output = entry.results;
        }
        for (output, 0..) |value, i| b.storeWords(1, shared.add(32).add(warp.mul(d).add(lane.add(i * 32)).mul(4)), .shared, .{value.bitCast(.i32)});
        b.syncThreads();
        var dims = b.openFor(tid, d, warps * 32, .{});
        var value = zero;
        for (0..warps) |i| value = value.add(b.loadWords(1, shared.add(32 + i * d * 4).add(dims.iv.mul(4)), .shared, .{})[0].bitCast(.f32));
        b.storeHalf(out.add(dims.iv.to(.i64).mul(2)), .global, value.to(dt).bitCast(.i16).to(.i32));
        dims.close();
        active.close();
    }
}

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
            8 => if (d >= 64) try runDimension(b, c, d, 8) else unreachable,
            else => unreachable,
        },
        512 => {
            const d = 512;
            switch (c.warps) {
                4 => try runDimension(b, c, d, 4),
                8 => try runDimension(b, c, d, 8),
                else => unreachable,
            }
        },
        else => unreachable,
    }
}

fn runDimension(b: *B, c: Config, comptime d: usize, comptime w: usize) cute.FinishError!void {
    switch (c.kv_tile) {
        16 => switch (c.dtype) {
            inline .bf16, .f16 => |dt| try runTile(b, c, d, w, 16, dt),
            else => unreachable,
        },
        inline 32, 64, 128 => |n| switch (c.dtype) {
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

    fn sync(t: TileContext) void {
        if (t.c.warps == 1) {
            t.b.syncWarp();
        } else t.b.syncThreads();
    }

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
        t.sync();
    }

    fn validSlot(t: TileContext, row: V) V {
        return t.b.loadWords(1, t.shared.add(t.c.indexBase()).add(row.mul(8)), .shared, .{})[0].ge(0);
    }

    fn loadKV(t: TileContext, source: V) void {
        const vectors = @divExact(t.c.kv_tile * t.c.head_dim, 8);
        const threads = t.c.warps * 32;
        for (0..@intCast(@divTrunc(vectors + threads - 1, threads))) |i| {
            const iv = t.tid.add(i * @as(usize, @intCast(threads)));
            if (@mod(vectors, threads) == 0) {
                t.copyKV(source, iv);
            } else {
                var valid = t.b.openIf(iv.lt(vectors));
                t.copyKV(source, iv);
                valid.close();
            }
        }
    }

    fn copyKV(t: TileContext, source: V, iv: V) void {
        const b = t.b;
        const c = t.c;
        const row = iv.div(@divExact(c.head_dim, 8));
        const col = iv.rem(@divExact(c.head_dim, 8)).mul(8);
        const address = t.shared.add(c.indexBase()).add(row.mul(8));
        const mapping = b.loadWords(2, address, .shared, .{});
        const page = mapping[0];
        const within = mapping[1];
        const valid = page.ge(0);
        const safe_page = b.select(valid, page, b.constant(.i32, 0));
        const offset = safe_page.to(.i64).mul(c.page_size).add(within.to(.i64)).mul(c.kv_heads).add(t.head.to(.i64)).mul(c.head_dim).add(col.to(.i64));
        b.cpAsync16(t.shared.add(c.kvBase()).add(swizzle(row, col, c.head_dim)), source.add(offset.mul(2)), valid);
    }

    fn loadQuery(t: TileContext) void {
        const vectors = @divExact(t.c.rows() * t.c.head_dim, 8);
        const threads = t.c.warps * 32;
        for (0..@intCast(@divTrunc(vectors + threads - 1, threads))) |i| {
            const iv = t.tid.add(i * @as(usize, @intCast(threads)));
            if (@mod(vectors, threads) == 0) {
                t.copyQuery(iv);
            } else {
                var valid = t.b.openIf(iv.lt(vectors));
                t.copyQuery(iv);
                valid.close();
            }
        }
    }

    fn copyQuery(t: TileContext, iv: V) void {
        const c = t.c;
        const row = iv.div(@divExact(c.head_dim, 8));
        const col = iv.rem(@divExact(c.head_dim, 8)).mul(8);
        const packed_row = t.first_row.add(row);
        const valid = packed_row.lt(c.group);
        const safe_row = t.b.select(valid, packed_row, t.b.constant(.i32, 0));
        const offset = t.token.to(.i64).mul(c.kv_heads).add(t.head.to(.i64)).mul(c.group).add(safe_row.to(.i64)).mul(c.head_dim).add(col.to(.i64));
        t.b.cpAsync16(t.shared.add(swizzle(row, col, c.head_dim)), t.q.add(offset.mul(2)), valid);
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
        const base = t.shared.add(t.c.redBase() + (if (maximum) @as(i64, 0) else 256)).add(row.mul(4));
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
            offset = offset.mul(c.splits).add(t.split.to(.i64)).mul(c.partialWidth()).add(dim.to(.i64));
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
    if (c.sharedBytes() > 48 * 1024) b.setFunctionAttribute(.named(b.ctx, "cu_attrs", .dict(b.ctx, &.{.named(b.ctx, "max_dynamic_shared_size_bytes", .int(b.ctx, .i32, c.sharedBytes()))})));
    const token = b.blockIdx().x;
    const first_row = b.blockIdx().y.rem(c.headTiles()).mul(c.rows());
    const head = b.blockIdx().y.div(c.headTiles());
    const starts = b.ptrToInt(a.starts, .i64);
    const active_tokens = loadI32(b, starts, b.constant(.i32, c.batch));
    // Unused capacity is observable in the public tensor and must be zero.
    var inactive = b.openIf(token.ge(active_tokens));
    {
        const out = b.ptrToInt(a.out, .i64);
        var dims = b.openFor(b.threadIdx().x, c.rows() * (if (c.splits > 1) c.partialWidth() else c.head_dim), c.warps * 32, .{});
        {
            const width = if (c.splits > 1) c.partialWidth() else c.head_dim;
            const row = first_row.add(dims.iv.div(width));
            const dim = dims.iv.rem(width);
            var valid = b.openIf(row.lt(c.group));
            {
                var offset = token.to(.i64).mul(c.kv_heads).add(head.to(.i64)).mul(c.group).add(row.to(.i64));
                if (c.splits > 1) {
                    offset = offset.mul(c.splits).add(b.blockIdx().z.to(.i64)).mul(width).add(dim.to(.i64));
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
        const seq = findSequence(b, c, starts, token, active_tokens);
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
        t.loadQuery();
        // Split the selected-slot list, not the logical KV timeline. Repeated
        // indices remain repeated softmax columns, including across splits.
        const tiles = @divTrunc(c.topk + n - 1, n);
        const lo = t.split.mul(tiles).div(c.splits);
        const hi = t.split.add(1).mul(tiles).div(c.splits);
        const row_base = t.warp.div(w).mul(16);
        const dim_part = t.warp.rem(w).mul(d / w);
        const row0 = t.lane.div(4);
        const col0 = t.lane.rem(4).mul(2);
        const zero = b.constant(.f32, 0);
        const neg_inf = b.constant(.f32, -std.math.inf(f32));
        const of = d / w / 8;
        const count = 4 + of;
        var initial: [count]V = @splat(zero);
        initial[0] = neg_inf;
        initial[1] = neg_inf;
        for (initial[4..]) |*v| v.* = packAcc(b, .{ zero, zero, zero, zero });
        // A one-tile partition needs neither online rescaling nor a tile loop.
        const result = if (c.splits == tiles)
            advanceTile(t, d, w, n, dt, true, t.split.mul(n), initial)
        else blk: {
            var loop = b.openFor(lo, hi, 1, initial);
            loop.yield(advanceTile(t, d, w, n, dt, false, loop.iv.mul(n), loop.carried));
            break :blk loop.results;
        };
        if (c.splits > tiles) b.cpAsyncWaitAll();
        for (0..2) |h| {
            const row = first_row.add(row_base).add(row0).add(h * 8);
            var valid = b.openIf(row.lt(c.group));
            {
                const m = result[h];
                const l = result[h + 2];
                var norm = b.select(l.gt(0), b.rcpApprox(l), zero);
                if (c.splits == 1 and c.sink) {
                    const sink = loadI32(b, t.sink, t.head.mul(c.group).add(row)).bitCast(.f32).mul(std.math.log2e);
                    const maximum = m.maximum(sink).maximum(-std.math.floatMax(f32));
                    const weight = b.exp2Approx(m.sub(maximum));
                    const denominator = l.mul(weight).add(b.exp2Approx(sink.sub(maximum)));
                    norm = weight.mul(b.select(denominator.gt(0), b.rcpApprox(denominator), zero));
                }
                for (0..of) |i| for (0..2) |j| {
                    const value = b.vectorExtract(result[4 + i], @intCast(h * 2 + j), .f32);
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

fn advanceTile(t: TileContext, comptime d: usize, comptime w: usize, comptime n: usize, comptime dt: cute.DType, comptime single: bool, key_base: V, previous: [4 + d / w / 8]V) [4 + d / w / 8]V {
    const b = t.b;
    const c = t.c;
    const first_row = t.first_row;
    const row_base = t.warp.div(w).mul(16);
    const key_warps = @min(w, @min(n / 8, 4));
    const feature_parts = w / key_warps;
    const feature_part = t.warp.div(key_warps).mul(d / feature_parts);
    const key_part = t.warp.rem(key_warps).mul(n / key_warps);
    const dim_part = t.warp.rem(w).mul(d / w);
    const row0 = t.lane.div(4);
    const col0 = t.lane.rem(4).mul(2);
    const zero = b.constant(.f32, 0);
    const neg_inf = b.constant(.f32, -std.math.inf(f32));
    const nf = n / key_warps / 8;
    const of = d / w / 8;
    const count = 4 + of;
    t.gather(key_base);
    t.loadKV(t.k);
    b.cpAsyncWaitAll();
    t.sync();
    var scores: [nf][4]V = @splat(@splat(zero));
    // Keep only MMA operands live in registers; Q remains in shared
    // memory so D512 does not consume the entire per-thread register file.
    for (0..d / feature_parts / 16) |ki| {
        const qa = matrixLoad(b, 4, false, t.shared.add(swizzle(row_base.add(t.lane.rem(16)), feature_part.add(ki * 16).add(t.lane.div(16).mul(8)), d)));
        for (0..nf) |i| {
            const kb = matrixLoad(b, 2, false, t.shared.add(c.kvBase()).add(swizzle(key_part.add(i * 8).add(t.lane.rem(8)), feature_part.add(ki * 16).add(t.lane.div(8).rem(2).mul(8)), d)));
            scores[i] = mma(b, dt, qa, kb, scores[i]);
        }
    }
    if (feature_parts > 1) {
        // Store each MMA register as a contiguous warp vector; row-major
        // FP32 score tiles alias shared-memory banks across query rows.
        // The max-reduction barrier protects these reads before reuse as P.
        const bank = t.warp.div(key_warps);
        for (0..nf) |i| for (0..4) |j| {
            const register = t.warp.rem(key_warps).mul(nf * 128).add(i * 128 + j * 32).add(t.lane);
            const offset = bank.mul(16 * n).add(register).mul(4);
            b.storeWords(1, t.shared.add(c.pBase()).add(offset), .shared, .{scores[i][j].bitCast(.i32)});
        };
        t.sync();
        for (0..nf) |i| for (0..4) |j| {
            const register = t.warp.rem(key_warps).mul(nf * 128).add(i * 128 + j * 32).add(t.lane);
            for (1..feature_parts) |peer| {
                const offset = bank.add(peer).rem(feature_parts).mul(16 * n).add(register).mul(4);
                scores[i][j] = scores[i][j].add(b.loadWords(1, t.shared.add(c.pBase()).add(offset), .shared, .{})[0].bitCast(.f32));
            }
        };
    }
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
    if (w == 1) {
        for (&local_max) |*value| inline for (.{ 2, 1 }) |step| {
            value.* = value.*.maximum(b.shuffleXor(value.*, step));
        };
    } else {
        var unique = b.openIf(t.warp.lt(key_warps));
        for (0..2) |h| t.publishReduction(row0.add(h * 8), local_max[h], true);
        unique.close();
        t.sync();
    }
    // All K reads have completed. Reuse its allocation for V, and
    // overlap these global loads with softmax and probability stores.
    t.loadKV(t.v);
    var m: [2]V = undefined;
    var alpha: [2]V = undefined;
    var local_sum: [2]V = @splat(zero);
    for (0..2) |h| {
        const tile_max = if (w == 1) local_max[h] else t.readReduction(key_warps, row0.add(h * 8), true);
        m[h] = if (single) tile_max else previous[h].maximum(tile_max);
        // Empty/fully masked row tiles have m=-inf and denominator 0.
        alpha[h] = if (single) zero else b.exp2Approx(previous[h].sub(m[h].maximum(-std.math.floatMax(f32))));
    }
    for (0..nf) |i| {
        for (0..4) |j| {
            scores[i][j] = b.exp2Approx(scores[i][j].sub(m[j / 2].maximum(-std.math.floatMax(f32))));
            local_sum[j / 2] = local_sum[j / 2].add(scores[i][j]);
        }
        var unique = b.openIf(t.warp.lt(key_warps));
        for (0..2) |h| {
            const row = row_base.add(row0).add(h * 8);
            const col = key_part.add(i * 8).add(col0);
            b.storeWords(1, t.shared.add(c.pBase()).add(swizzle(row, col, n)), .shared, .{b.packHalf2(scores[i][h * 2], scores[i][h * 2 + 1], dt)});
        }
        unique.close();
    }
    if (w == 1) {
        for (&local_sum) |*value| inline for (.{ 2, 1 }) |step| {
            value.* = value.*.add(b.shuffleXor(value.*, step));
        };
    } else {
        var unique = b.openIf(t.warp.lt(key_warps));
        for (0..2) |h| t.publishReduction(row0.add(h * 8), local_sum[h], false);
        unique.close();
    }
    b.cpAsyncWaitAll();
    t.sync();
    var next: [count]V = undefined;
    for (0..2) |h| {
        next[h] = m[h];
        const tile_sum = if (w == 1) local_sum[h] else t.readReduction(key_warps, row0.add(h * 8), false);
        next[h + 2] = if (single) tile_sum else previous[h + 2].mul(alpha[h]).add(tile_sum);
    }
    var acc: [of][4]V = undefined;
    for (0..of) |i| for (0..4) |j| {
        acc[i][j] = if (single) zero else b.vectorExtract(previous[4 + i], @intCast(j), .f32).mul(alpha[j / 2]);
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
    if (!single) t.sync();
    return next;
}

pub const Merge = cute.Kernel(Config, .{
    .name = "paged_attention_cute_indexed_grouped_merge",
    .inputs = &.{ "partials", "starts", "sink" },
    .outputs = &.{"out"},
    .run = merge,
});

// Lanes assigned to a feature own disjoint split partitions. Shuffles combine
// their softmax state, shortening the serial global-load chain at small batch.
fn merge(b: *B, c: Config) cute.FinishError!void {
    std.debug.assert(c.splits > 0 and c.splits <= 128 and (c.merge_parts == 4 or c.merge_parts == 8 or c.merge_parts == 16 or c.merge_parts == 32));
    const a = try b.declareArgs(.{
        .partials = .{ .tensor = .{ .dtype = .f32, .shape = &.{ c.tokens, c.kv_heads * c.group, c.splits, c.partialWidth() } } },
        .starts = .{ .tensor = .{ .dtype = .i32, .shape = &.{c.batch + 1} } },
        .sink = .{ .tensor = .{ .dtype = .f32, .shape = &.{c.kv_heads * c.group} } },
        .out = .{ .tensor = .{ .dtype = c.dtype, .shape = &.{ c.tokens, c.kv_heads * c.group, c.head_dim } } },
    });
    const tid = b.threadIdx().x;
    const cols = @divExact(32, c.merge_parts);
    const part_lane = tid.rem(32).div(cols);
    const dim = b.blockIdx().z.mul(4 * cols).add(tid.div(32).mul(cols)).add(tid.rem(cols));
    const token = b.blockIdx().x;
    const head = b.blockIdx().y;
    const zero = b.constant(.f32, 0);
    var inactive = b.openIf(token.ge(a.starts.get(.{c.batch})).bitAnd(part_lane.eq(0)));
    a.out.set(.{ token, head, dim }, b.constant(c.dtype, 0));
    inactive.close();
    var valid = b.openIf(token.lt(a.starts.get(.{c.batch})));
    {
        const sink = if (c.sink) a.sink.get(.{head}).mul(std.math.log2e) else b.constant(.f32, -std.math.inf(f32));
        var maxima: [32]V = @splat(sink);
        const groups = @divTrunc(c.splits + c.merge_parts - 1, c.merge_parts);
        for (0..@as(usize, @intCast(groups))) |i| {
            const part = part_lane.add(i * @as(usize, @intCast(c.merge_parts)));
            maxima[i] = b.select(part.lt(c.splits), a.partials.get(.{ token, head, part.minimum(c.splits - 1), c.head_dim }), sink);
        }
        var max_stride: usize = 1;
        while (max_stride < 32) : (max_stride *= 2) {
            var i: usize = 0;
            while (i < 32) : (i += 2 * max_stride) maxima[i] = maxima[i].maximum(maxima[i + max_stride]);
        }
        var m = maxima[0].maximum(sink);
        var step = cols;
        while (step < 32) : (step *= 2) m = m.maximum(b.shuffleXor(m, @intCast(step)));
        m = m.maximum(-std.math.floatMax(f32));
        const chunk = @min(@as(i64, 32), @divTrunc(c.splits + c.merge_parts - 1, c.merge_parts));
        const sink_weight = if (c.sink) b.select(part_lane.eq(0), b.exp2Approx(sink.sub(m)), zero) else zero;
        var sum = b.openFor(0, c.splits, chunk * c.merge_parts, .{ zero, sink_weight });
        {
            var values: [32]V = @splat(zero);
            var denominators: [32]V = @splat(zero);
            for (0..@as(usize, @intCast(chunk))) |i| {
                const part = sum.iv.add(part_lane).add(i * @as(usize, @intCast(c.merge_parts)));
                const safe_part = part.minimum(c.splits - 1);
                const weight = b.select(part.lt(c.splits), b.exp2Approx(a.partials.get(.{ token, head, safe_part, c.head_dim }).sub(m)), zero);
                values[i] = weight.mul(a.partials.get(.{ token, head, safe_part, dim }));
                denominators[i] = weight.mul(a.partials.get(.{ token, head, safe_part, c.head_dim + 1 }));
            }
            var stride: usize = 1;
            while (stride < 32) : (stride *= 2) {
                var i: usize = 0;
                while (i < 32) : (i += 2 * stride) {
                    values[i] = values[i].add(values[i + stride]);
                    denominators[i] = denominators[i].add(denominators[i + stride]);
                }
            }
            sum.yield(.{ sum.carried[0].add(values[0]), sum.carried[1].add(denominators[0]) });
        }
        var numerator = sum.results[0];
        var denominator = sum.results[1];
        step = cols;
        while (step < 32) : (step *= 2) {
            numerator = numerator.add(b.shuffleXor(numerator, @intCast(step)));
            denominator = denominator.add(b.shuffleXor(denominator, @intCast(step)));
        }
        var leader = b.openIf(part_lane.eq(0));
        a.out.set(.{ token, head, dim }, numerator.mul(b.select(denominator.gt(0), b.rcpApprox(denominator), zero)).to(c.dtype));
        leader.close();
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
