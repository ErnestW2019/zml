const reference = @import("reference.zig");
const std = @import("std");
const zml = @import("zml");
const cases = @import("cases.zig");
const fixtures = @import("fixtures.zig");
const Inputs = fixtures.Inputs;

test "CPU reference self checks" {
    _ = @import("reference.zig");
}

test "paged attention correctness" {
    const platform = zml.testing.env();
    if (platform.target != .cuda) return error.CudaRequired;

    const a = std.testing.allocator;
    const io = std.testing.io;

    const cc = zml.platform.cuda.computeCapability(platform) orelse return error.UnknownComputeCapability;
    std.debug.print("PAGED_ATTENTION dtype=bf16 page_size=16 seed=42 cc={d}.{d}\n{f}\n", .{
        cc.major, cc.minor, platform.fmtVerbose(),
    });

    var failures: usize = 0;
    for (cases.correctness) |c| {
        std.debug.print("CASE name={s} d={d} b={d} hkv={d} g={d} prefill={d} chunk={d} kv={d} mask={s} backend=cute\n", .{
            c.name, c.head_dim, c.batch, c.kv_heads, c.group, c.prefill, if (c.prefill > 0) c.chunk else 1, c.kv_len, @tagName(c.mask),
        });

        runCorrectness(a, io, platform, c, .cute, .bf16) catch |err| {
            failures += 1;
            std.debug.print("FAIL error={s}\n", .{@errorName(err)});
        };
    }

    try std.testing.expectEqual(@as(usize, 0), failures);
}

test "FP16 paged attention" {
    const platform = zml.testing.env();
    if (platform.target != .cuda) return error.CudaRequired;
    for ([_]cases.Case{
        .{ .name = "fp16_decode", .head_dim = 64, .kv_len = 257 },
        .{ .name = "fp16_mixed", .prefill = 1, .mask = .non_causal },
        .{ .name = "fp16_window_sink", .head_dim = 256, .prefill = 1, .mask = .window, .sink = true },
        .{ .name = "fp16_split_sink", .head_dim = 512, .batch = 1, .kv_len = 2049, .sink = true, .shuffle_pages = true },
    }) |c| try runCorrectness(std.testing.allocator, std.testing.io, platform, c, .cute, .f16);
}

test "indexed paged attention BF16 all backends" {
    try runSparseCorrectness(.bf16);
}

test "indexed paged attention FP16 all backends" {
    try runSparseCorrectness(.f16);
}

fn runSparseCorrectness(dtype: zml.DataType) !void {
    const platform = zml.testing.env();
    if (platform.target != .cuda) return error.CudaRequired;
    var failures: usize = 0;
    for (cases.sparseCorrectness) |c| {
        for (std.enums.values(fixtures.Backend)) |backend| {
            std.debug.print("CASE name={s} d={d} b={d} hkv={d} g={d} prefill={d} chunk={d} kv={d} topk={d} backend={s} compute_dtype={s}\n", .{
                c.name, c.head_dim, c.batch, c.kv_heads, c.group, c.prefill, if (c.prefill > 0) c.chunk else 1, c.kv_len, c.topk, @tagName(backend), @tagName(dtype),
            });
            runCorrectness(std.testing.allocator, std.testing.io, platform, c, backend, dtype) catch |err| {
                failures += 1;
                std.debug.print("FAIL error={s}\n", .{@errorName(err)});
            };
        }
    }
    try std.testing.expectEqual(@as(usize, 0), failures);
}

fn runCorrectness(a: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, c: cases.Case, backend: fixtures.Backend, dtype: zml.DataType) !void {
    const inputs = Inputs.initDType(c, dtype);
    const exe = try platform.compileFn(a, io, fixtures.forwardBackend, .{ inputs, c, backend }, .{ .program_name = "paged_attention_harness" });
    defer exe.deinit();

    var runner = try exe.runner(a);
    defer runner.deinit(a);

    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();

    const fixture = try fixtures.Fixture.init(arena.allocator(), c);
    if (dtype == .f16) fixture.quantizeF16();

    var buffers = try fixtures.upload(a, io, platform, inputs, fixture);
    defer zml.Buffer.deinitAll(Inputs, &buffers);

    var output: zml.Buffer = undefined;
    runner.run(io, .{buffers}, .{&output}, .{ .wait = true });
    defer output.deinit();

    var host = try output.toSliceAlloc(a, io);
    defer host.free(a);

    const max_error = if (dtype == .f16)
        try reference.check(a, c, fixture, host.items(f16))
    else
        try reference.check(a, c, fixture, host.items(zml.floats.BFloat16));
    if (backend != .triton) {
        if (dtype == .f16) {
            try reference.checkZeroTail(c, fixture, host.items(f16));
        } else {
            try reference.checkZeroTail(c, fixture, host.items(zml.floats.BFloat16));
        }
    }
    std.debug.print("RESULT name={s} d={d} b={d} hkv={d} g={d} prefill={d} chunk={d} kv={d} topk={d} backend={s} compute_dtype={s} check=all_active_outputs inactive_tail={s} max_abs={d:.6}\n", .{
        c.name, c.head_dim, c.batch, c.kv_heads, c.group, c.prefill, if (c.prefill > 0) c.chunk else 1, c.kv_len, c.topk, @tagName(backend), @tagName(dtype), if (backend == .triton) "unspecified" else "zero_checked", max_error,
    });
}
