const std = @import("std");
const zml = @import("zml");
const cases = @import("cases.zig");
const fixtures = @import("fixtures.zig");
const reference = @import("reference.zig");
const Inputs = fixtures.Inputs;

const Args = struct {
    profile_dir: ?[]const u8 = null,
    paired: bool = false,
    tuning: zml.paged_attention.cute.SparseTuning = .{},
    sparse: bool = false,
    stress: bool = false,
    all_backends: bool = false,
    backend: ?fixtures.Backend = null,
    case: ?[]const u8 = null,
    dtype: zml.DataType = .bf16,
    warmup: usize = 10,
    samples: usize = 101,
};

pub fn main(init: std.process.Init) !void {
    const args = try parseArgs(init);
    const platform = try zml.Platform.auto(init.gpa, init.io, .{});
    defer platform.deinit(init.gpa, init.io);
    if (platform.target != .cuda) return error.CudaRequired;

    const mesh = try platform.registerMesh("tp", .mesh(.{ .model = .high_bandwidth }));
    const cc = zml.platform.cuda.computeCapability(platform) orelse return error.UnknownComputeCapability;
    std.debug.print("PAGED_ATTENTION dtype={s} page_size=16 seed=42 cc={d}.{d}\n{f}\n", .{
        @tagName(args.dtype), cc.major, cc.minor, platform.fmtVerbose(),
    });
    std.debug.print("timing=synchronized_invocation_us devices={d} hkv=per_device (KV heads sharded across all visible GPUs; includes runtime, output allocation, attention, merge and wait; excludes compile, upload, reference validation and output destruction)\n", .{platform.devices.len});

    std.debug.print("TUNING {any}\n", .{args.tuning});
    const suite: []const cases.Case = if (args.stress) &cases.sparseStressBenchmarks else if (args.sparse) &cases.sparseBenchmarks else &cases.benchmarks;
    var failures: usize = 0;
    var selected: usize = 0;
    for (suite, 0..) |c, index| {
        if (args.case) |filter| {
            const wanted_index: ?usize = std.fmt.parseInt(usize, filter, 10) catch null;
            if (wanted_index) |wanted| {
                if (wanted != index) continue;
            } else if (!std.mem.eql(u8, filter, c.name)) continue;
        }
        selected += 1;
        if (args.paired) {
            try runPaired(init.gpa, init.io, platform, mesh, c, index, args);
            continue;
        }
        const backends = std.enums.values(fixtures.Backend);
        for (0..backends.len) |rank| {
            const backend = backends[(rank + index) % backends.len];
            if (args.backend) |wanted| {
                if (wanted != backend) continue;
            } else if (!args.sparse and !args.all_backends and backend != .cute) continue;
            // Triton's GB300 dot accumulator has a documented 16K-element
            // tile limit. Keep optional stress runs from faulting the device.
            if (backend == .triton and c.mask == .indices and cc.major == 10 and cc.minor == 3 and c.group * c.head_dim > 16 * 1024) {
                std.debug.print("SKIP index={d} name={s} backend=triton reason=gb300_triton_head_tile_limit\n", .{ index, c.name });
                continue;
            }
            std.debug.print("CASE index={d} name={s} d={d} b={d} hkv={d} g={d} prefill={d} chunk={d} kv={d} topk={d} mask={s} backend={s}\n", .{
                index, c.name, c.head_dim, c.batch, c.kv_heads, c.group, c.prefill, if (c.prefill > 0) c.chunk else 1, c.kv_len, c.topk, @tagName(c.mask), @tagName(backend),
            });
            runBenchmark(init.gpa, init.io, platform, mesh, c, index, backend, args) catch |err| {
                failures += 1;
                std.debug.print("FAIL error={s}\n", .{@errorName(err)});
            };
        }
    }
    if (selected == 0) return error.UnknownCase;
    if (failures != 0) return error.BenchmarkFailed;
}

fn parseArgs(init: std.process.Init) !Args {
    var result: Args = .{};
    var it = init.minimal.args.iterate();
    _ = it.next();
    while (it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--sparse")) {
            result.sparse = true;
        } else if (std.mem.eql(u8, arg, "--paired")) {
            result.paired = true;
        } else if (std.mem.eql(u8, arg, "--stress")) {
            result.sparse = true;
            result.stress = true;
        } else if (std.mem.startsWith(u8, arg, "--tune-head-rows=")) {
            result.tuning.headRows = try std.fmt.parseInt(i64, arg["--tune-head-rows=".len..], 10);
        } else if (std.mem.startsWith(u8, arg, "--tune-warps=")) {
            result.tuning.warps = try std.fmt.parseInt(i64, arg["--tune-warps=".len..], 10);
        } else if (std.mem.startsWith(u8, arg, "--tune-tile=")) {
            result.tuning.kvTile = try std.fmt.parseInt(i64, arg["--tune-tile=".len..], 10);
        } else if (std.mem.startsWith(u8, arg, "--tune-splits=")) {
            result.tuning.splits = try std.fmt.parseInt(i64, arg["--tune-splits=".len..], 10);
        } else if (std.mem.startsWith(u8, arg, "--tune-merge-lanes=")) {
            result.tuning.mergeLanes = try std.fmt.parseInt(i64, arg["--tune-merge-lanes=".len..], 10);
        } else if (std.mem.startsWith(u8, arg, "--tune-merge-vector=")) {
            result.tuning.mergeVector = try std.fmt.parseInt(i64, arg["--tune-merge-vector=".len..], 10);
        } else if (std.mem.startsWith(u8, arg, "--tune-merge-parts=")) {
            result.tuning.mergeParts = try std.fmt.parseInt(i64, arg["--tune-merge-parts=".len..], 10);
        } else if (std.mem.startsWith(u8, arg, "--backend=")) {
            const value = arg["--backend=".len..];
            result.all_backends = std.mem.eql(u8, value, "all");
            result.backend = if (std.mem.eql(u8, value, "all")) null else std.meta.stringToEnum(fixtures.Backend, value) orelse return error.InvalidBackend;
        } else if (std.mem.startsWith(u8, arg, "--case=")) {
            result.case = arg["--case=".len..];
        } else if (std.mem.startsWith(u8, arg, "--dtype=")) {
            result.dtype = std.meta.stringToEnum(zml.DataType, arg["--dtype=".len..]) orelse return error.InvalidDType;
            if (result.dtype != .bf16 and result.dtype != .f16) return error.InvalidDType;
        } else if (std.mem.startsWith(u8, arg, "--profile-dir=")) {
            result.profile_dir = arg["--profile-dir=".len..];
        } else if (std.mem.startsWith(u8, arg, "--warmup=")) {
            result.warmup = try std.fmt.parseInt(usize, arg["--warmup=".len..], 10);
        } else if (std.mem.startsWith(u8, arg, "--samples=")) {
            result.samples = try std.fmt.parseInt(usize, arg["--samples=".len..], 10);
            if (result.samples == 0) return error.InvalidSamples;
        } else if (std.mem.eql(u8, arg, "--help")) {
            std.debug.print("paged attention benchmark [--sparse|--stress] [--paired] [--backend=cute|cute_generic|triton|all] [--case=INDEX|NAME] [--dtype=bf16|f16] [--warmup=10] [--samples=101] [--profile-dir=PATH] [--tune-head-rows=N --tune-warps=N --tune-tile=N --tune-splits=N --tune-merge-parts=N --tune-merge-vector=N --tune-merge-lanes=N ]\nSparse mode compares all three backends on identical seed-42 inputs; the default dense suite uses CuTe. Every measured case is first checked against the CPU reference. Use --stress --backend=NAME to isolate large-group backend resource limits.\n", .{});
            std.process.exit(0);
        } else return error.UnknownArgument;
    }
    if (result.tuning.headRows) |v| if (v != 8 and v != 16) return error.InvalidTuning;
    if (result.tuning.splits) |v| if (v < 1 or v > 128) return error.InvalidTuning;
    if (result.tuning.mergeLanes) |v| if (std.mem.indexOfScalar(i64, &.{ 8, 16, 32 }, v) == null) return error.InvalidTuning;
    if (result.tuning.mergeVector) |v| if (std.mem.indexOfScalar(i64, &.{ 0, 4, 8, 16 }, v) == null) return error.InvalidTuning;
    if (result.tuning.warps) |v| if (std.mem.indexOfScalar(i64, &.{ 1, 2, 4, 8 }, v) == null) return error.InvalidTuning;
    if (result.tuning.kvTile) |v| if (std.mem.indexOfScalar(i64, &.{ 16, 32, 64, 128 }, v) == null) return error.InvalidTuning;
    if (result.tuning.mergeParts) |v| if (std.mem.indexOfScalar(i64, &.{ 4, 8, 16, 32 }, v) == null) return error.InvalidTuning;
    if (result.paired and (result.stress or result.backend != null)) return error.InvalidPairedOptions;
    return result;
}

fn runBenchmark(a: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, mesh: *const zml.Mesh, c: cases.Case, index: usize, backend: fixtures.Backend, args: Args) !void {
    // Keep the number of KV heads per GPU constant.
    var global_case = c;
    global_case.kv_heads *= platform.devices.len;
    var inputs = Inputs.initDType(global_case, args.dtype);
    inputs.q = inputs.q.withPartitioning(mesh, .{ .hkv = .model });
    inputs.k = inputs.k.withPartitioning(mesh, .{ .hkv = .model });
    inputs.v = inputs.v.withPartitioning(mesh, .{ .hkv = .model });
    inputs.sinks = inputs.sinks.withPartitioning(mesh, .{ ._0 = .model });

    const compile_start: std.Io.Timestamp = .now(io, .awake);
    const exe = try platform.compileFn(a, io, forward, .{ inputs, global_case, mesh, backend, args.tuning }, .{
        .program_name = "paged_attention_harness",
        .meshes = &.{mesh},
    });
    defer exe.deinit();
    const compile_us = elapsedUs(compile_start, io);
    var runner = try exe.runner(a);
    defer runner.deinit(a);
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const fixture = try fixtures.Fixture.init(arena.allocator(), global_case);
    if (args.dtype == .f16) fixture.quantizeF16();
    var buffers = try fixtures.upload(a, io, platform, inputs, fixture);
    defer zml.Buffer.deinitAll(Inputs, &buffers);

    var output: zml.Buffer = undefined;
    const max_error = validate: {
        runner.run(io, .{buffers}, .{&output}, .{ .wait = true });
        defer output.deinit();
        var host = try output.toSliceAlloc(a, io);
        defer host.free(a);
        break :validate if (args.dtype == .f16)
            try reference.check(a, global_case, fixture, host.items(f16))
        else
            try reference.check(a, global_case, fixture, host.items(zml.floats.BFloat16));
    };
    for (0..args.warmup) |_| {
        runner.run(io, .{buffers}, .{&output}, .{ .wait = true });
        output.deinit();
    }
    const session = try std.fmt.allocPrint(a, "{s}-{d}-{s}", .{ @tagName(args.dtype), index, @tagName(backend) });
    defer a.free(session);
    var profiler = if (args.profile_dir) |path| try platform.profiler(a, io, .{
        .repository_path = path,
        .session_id = session,
        .host_tracer_level = 0,
        .device_tracer_level = 1,
        .python_tracer_level = 0,
        .raise_error_on_start_failure = true,
    }) else null;
    defer if (profiler) |*p| p.deinit();
    if (profiler) |*p| try p.start();
    const samples = try a.alloc(f64, args.samples);
    defer a.free(samples);
    for (samples) |*sample| {
        const call_start: std.Io.Timestamp = .now(io, .awake);
        runner.run(io, .{buffers}, .{&output}, .{ .wait = true });
        sample.* = elapsedUs(call_start, io);
        output.deinit();
    }
    if (profiler) |*p| _ = try p.stop();
    std.mem.sort(f64, samples, {}, std.sort.asc(f64));
    const median = if (samples.len % 2 == 0)
        (samples[samples.len / 2 - 1] + samples[samples.len / 2]) / 2
    else
        samples[samples.len / 2];
    std.debug.print("RESULT index={d} name={s} d={d} b={d} hkv={d} g={d} prefill={d} chunk={d} kv={d} topk={d} backend={s} compute_dtype={s} check=all_outputs max_abs={d:.6} compile_ms={d:.3} n={d} median_us={d:.3} p10_us={d:.3} p90_us={d:.3}\n", .{
        index, c.name, c.head_dim, c.batch, c.kv_heads, c.group, c.prefill, if (c.prefill > 0) c.chunk else 1, c.kv_len, c.topk, @tagName(backend), @tagName(args.dtype), max_error, compile_us / 1000, samples.len, median, samples[samples.len / 10], samples[(samples.len - 1) * 9 / 10],
    });
}

fn forward(inputs: Inputs, c: cases.Case, mesh: *const zml.Mesh, backend: fixtures.Backend, tuning: zml.paged_attention.cute.SparseTuning) zml.Tensor {
    const Call = struct {
        inputs: Inputs,
        case: cases.Case,
        backend: fixtures.Backend,
        tuning: zml.paged_attention.cute.SparseTuning,

        fn local(self: @This(), _: zml.Shape) zml.Tensor {
            return fixtures.forwardBackendTuned(self.inputs, self.case, self.backend, self.tuning);
        }
    };
    const heads: zml.Sharding.Partitioning.Logical = .parse(.{ .hkv = .model });
    return zml.ops.manualComputation(Call.local, .{ .inputs = inputs, .case = c, .backend = backend, .tuning = tuning }, inputs.q.shape(), mesh, .{
        .in_specs = .{ .inputs = .{
            .q = heads,
            .k = heads,
            .v = heads,
            .table = .replicated,
            .lengths = .replicated,
            .starts = .replicated,
            .sinks = .parse(.{ ._0 = .model }),
            .indices = .replicated,
        } },
        .out_specs = heads,
    });
}

fn elapsedUs(start: std.Io.Timestamp, io: std.Io) f64 {
    return @as(f64, @floatFromInt(start.untilNow(io, .awake).toNanoseconds())) / 1000;
}

// Keep all executables and inputs alive and rotate order for each sample round.
fn runPaired(a: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, mesh: *const zml.Mesh, c: cases.Case, index: usize, args: Args) !void {
    var global_case = c;
    global_case.kv_heads *= platform.devices.len;
    var inputs = Inputs.initDType(global_case, args.dtype);
    inputs.q = inputs.q.withPartitioning(mesh, .{ .hkv = .model });
    inputs.k = inputs.k.withPartitioning(mesh, .{ .hkv = .model });
    inputs.v = inputs.v.withPartitioning(mesh, .{ .hkv = .model });
    inputs.sinks = inputs.sinks.withPartitioning(mesh, .{ ._0 = .model });
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const fixture = try fixtures.Fixture.init(arena.allocator(), global_case);
    if (args.dtype == .f16) fixture.quantizeF16();
    var buffers = try fixtures.upload(a, io, platform, inputs, fixture);
    defer zml.Buffer.deinitAll(Inputs, &buffers);
    const backends = std.enums.values(fixtures.Backend);
    var executables: [3]zml.Exe = undefined;
    var runners: [3]zml.Exe.Runner = undefined;
    var compiled: usize = 0;
    var initialized: usize = 0;
    defer for (executables[0..compiled]) |e| e.deinit();
    defer for (runners[0..initialized]) |*r| r.deinit(a);
    var errors: [3]f32 = undefined;
    var compile_us: [3]f64 = undefined;
    var samples: [3][]f64 = undefined;
    for (&samples) |*v| v.* = try arena.allocator().alloc(f64, args.samples);
    for (backends, 0..) |backend, i| {
        const start: std.Io.Timestamp = .now(io, .awake);
        const programName = try std.fmt.allocPrint(a, "paired_attention_{s}", .{@tagName(backend)});
        defer a.free(programName);
        executables[i] = try platform.compileFn(a, io, forward, .{ inputs, global_case, mesh, backend, args.tuning }, .{ .program_name = programName, .meshes = &.{mesh} });
        compiled += 1;
        compile_us[i] = elapsedUs(start, io);
        runners[i] = try executables[i].runner(a);
        initialized += 1;
        var output: zml.Buffer = undefined;
        runners[i].run(io, .{buffers}, .{&output}, .{ .wait = true });
        defer output.deinit();
        var host = try output.toSliceAlloc(a, io);
        defer host.free(a);
        errors[i] = if (args.dtype == .f16) try reference.check(a, global_case, fixture, host.items(f16)) else try reference.check(a, global_case, fixture, host.items(zml.floats.BFloat16));
    }
    const session = try std.fmt.allocPrint(a, "{s}-{d}-paired", .{ @tagName(args.dtype), index });
    defer a.free(session);
    var profiler = if (args.profile_dir) |path| try platform.profiler(a, io, .{
        .repository_path = path,
        .session_id = session,
        .host_tracer_level = 0,
        .device_tracer_level = 1,
        .python_tracer_level = 0,
        .raise_error_on_start_failure = true,
    }) else null;
    defer if (profiler) |*p| p.deinit();
    for (0..args.warmup + args.samples) |round| {
        if (round == args.warmup) if (profiler) |*p| try p.start();
        for (0..backends.len) |rank| {
            const i = (rank + round) % backends.len;
            var output: zml.Buffer = undefined;
            const start: std.Io.Timestamp = .now(io, .awake);
            runners[i].run(io, .{buffers}, .{&output}, .{ .wait = true });
            const us = elapsedUs(start, io);
            if (round >= args.warmup) samples[i][round - args.warmup] = us;
            output.deinit();
        }
    }
    if (profiler) |*p| _ = try p.stop();
    const deltas = try arena.allocator().alloc(f64, args.samples);
    const ratios = try arena.allocator().alloc(f64, args.samples);
    const cute_index = @backingInt(fixtures.Backend.cute);
    const triton_index = @backingInt(fixtures.Backend.triton);
    for (deltas, ratios, 0..) |*delta, *ratio, i| {
        delta.* = samples[cute_index][i] - samples[triton_index][i];
        ratio.* = samples[triton_index][i] / samples[cute_index][i];
    }
    std.mem.sort(f64, deltas, {}, std.sort.asc(f64));
    std.mem.sort(f64, ratios, {}, std.sort.asc(f64));
    std.debug.print("PAIR index={d} compute_dtype={s} n={d} cute_minus_triton_median_us={d:.3} triton_over_cute_median={d:.6} delta_p10_us={d:.3} delta_p90_us={d:.3}\n", .{ index, @tagName(args.dtype), deltas.len, deltas[deltas.len / 2], ratios[ratios.len / 2], deltas[deltas.len / 10], deltas[(deltas.len - 1) * 9 / 10] });
    for (backends, 0..) |backend, i| {
        const values = samples[i];
        std.mem.sort(f64, values, {}, std.sort.asc(f64));
        const median = if (values.len % 2 == 0) (values[values.len / 2 - 1] + values[values.len / 2]) / 2 else values[values.len / 2];
        std.debug.print("RESULT index={d} name={s} d={d} b={d} hkv={d} g={d} prefill={d} chunk={d} kv={d} topk={d} backend={s} compute_dtype={s} check=all_outputs paired=true max_abs={d:.6} compile_ms={d:.3} n={d} median_us={d:.3} p10_us={d:.3} p90_us={d:.3}\n", .{ index, c.name, c.head_dim, c.batch, c.kv_heads, c.group, c.prefill, if (c.prefill > 0) c.chunk else 1, c.kv_len, c.topk, @tagName(backend), @tagName(args.dtype), errors[i], compile_us[i] / 1000, values.len, median, values[values.len / 10], values[(values.len - 1) * 9 / 10] });
    }
}
