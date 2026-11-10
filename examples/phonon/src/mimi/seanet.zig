const std = @import("std");
const zml = @import("zml");
const convolution = @import("convolution.zig");
const test_data = @import("seanet_test_data.zig");

test {
    std.testing.refAllDecls(@This());
}

/// SEANet's complete convolution history. All tensors use [.b,.d,.s] and begin
/// at zero. Conv histories store preceding inputs; upsample histories store
/// unfinished bias-free outputs. No history is shared between batch rows.
pub const Cache = struct {
    initial: zml.Tensor,
    upsample: [3]zml.Tensor,
    residual: [3]zml.Tensor,
    final: zml.Tensor,
};
pub const Cached_Output = struct { hidden: zml.Tensor, cache: Cache };

/// Converts 200 Hz transformer features into a 24 kHz mono waveform. A K=7
/// convolution is followed by three ELU/upsample/residual stages (6x,5x,4x),
/// then ELU and a K=3 output convolution. Channels descend 512 -> 256 -> 128 -> 64 -> 1.
/// Parameters come from `decoder.decoder_seanet.layers`; convolutions are causal.
pub const SEANet = struct {
    initial: convolution.Conv1d,
    upsample: [3]convolution.ConvTranspose1d,
    residual: [3]convolution.Resnet_Block,
    final: convolution.Conv1d,

    pub fn init(store: zml.io.TensorStore.View) SEANet {
        const here = @typeName(@This()) ++ ".init";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        return .{
            .initial = .init(store.withPrefix("layers.0")),
            .upsample = .{
                .init(store.withPrefix("layers.2"), 6, 1),
                .init(store.withPrefix("layers.5"), 5, 1),
                .init(store.withPrefix("layers.8"), 4, 1),
            },
            .residual = .{
                .init(store.withPrefix("layers.3")),
                .init(store.withPrefix("layers.6")),
                .init(store.withPrefix("layers.9")),
            },
            .final = .init(store.withPrefix("layers.11")),
        };
    }

    pub fn bufferize(self: SEANet, allocator: std.mem.Allocator) !zml.Bufferized(SEANet) {
        const here = @typeName(@This()) ++ ".bufferize";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        var buffers: zml.Bufferized(SEANet) = .{
            .initial = try zml.mem.bufferize(allocator, convolution.Conv1d, &self.initial),
            .upsample = undefined,
            .residual = undefined,
            .final = try zml.mem.bufferize(allocator, convolution.Conv1d, &self.final),
        };
        for (&self.upsample, &buffers.upsample) |*layer, *target| target.* = try zml.mem.bufferize(allocator, convolution.ConvTranspose1d, layer);
        for (&self.residual, &buffers.residual) |*layer, *target| target.* = try zml.mem.bufferize(allocator, convolution.Resnet_Block, layer);
        return buffers;
    }

    /// `input`: floating [B, 512, S], transformer output transposed to channels
    /// first. Returns [B, 1, S*120] audio samples in input.dtype(), with no output
    /// clipping. Tags .b independent waveforms, .d channels, .s time at each rate.
    pub fn forward_uncached(self: SEANet, input: zml.Tensor) zml.Tensor {
        const here = @typeName(@This()) ++ ".forward_uncached";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        var hidden = self.initial.forward_uncached(input);
        for (self.upsample, self.residual) |upsampler, residual_block| {
            const activated = zml.nn.elu(hidden, 1);
            const upsampled = upsampler.forward_uncached(activated);
            hidden = residual_block.forward_uncached(upsampled);
        }
        return self.final.forward_uncached(zml.nn.elu(hidden, 1));
    }

    /// `input`: floating [B,512,Q], new contiguous transformer features.
    /// `cache`: histories from init_cache, initially zero then carried between
    /// chunks. Returns [B,1,Q*120] finalized samples and all updated histories.
    /// Output follows input.dtype(); each history retains its own floating dtype.
    /// Tags .b batch, .d channels, .s time; history lengths are kernel/stride dependent.
    pub fn forward_cached(self: SEANet, input: zml.Tensor, cache: Cache) Cached_Output {
        const here = @typeName(@This()) ++ ".forward_cached";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        const initial = self.initial.forward_cached(input, cache.initial);
        var hidden = initial.hidden;
        var history = cache;
        history.initial = initial.cache;

        for (self.upsample, self.residual, 0..) |upsampler, residual_block, index| {
            const activated = zml.nn.elu(hidden, 1);
            const upsampled = upsampler.forward_cached(activated, cache.upsample[index]);
            const refined = residual_block.forward_cached(upsampled.hidden, cache.residual[index]);
            hidden = refined.hidden;
            history.upsample[index] = upsampled.cache;
            history.residual[index] = refined.cache;
        }

        const waveform = self.final.forward_cached(zml.nn.elu(hidden, 1), cache.final);
        history.final = waveform.cache;
        return .{ .hidden = waveform.hidden, .cache = history };
    }

    pub fn init_cache(self: SEANet, batch: i64, dtype: zml.DataType) Cache {
        const here = @typeName(@This()) ++ ".init_cache";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        var cache: Cache = .{
            .initial = .fromShape(self.initial.cache_shape(batch, dtype)),
            .upsample = undefined,
            .residual = undefined,
            .final = .fromShape(self.final.cache_shape(batch, dtype)),
        };
        for (self.upsample, self.residual, 0..) |up, res, index| {
            cache.upsample[index] = .fromShape(up.cache_shape(batch, dtype));
            cache.residual[index] = .fromShape(res.conv1.cache_shape(batch, dtype));
        }
        return cache;
    }
};

test "SEANet" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    var rng: std.Random.DefaultPrng = .init(42);
    const random = rng.random();
    const layer: SEANet = .{
        .initial = .{
            .weight = .fromShape(.init(.{ .out = 512, .in = 512, .k = 7 }, .f32)),
            .bias = .fromShape(.init(.{ .d = 512 }, .f32)),
        },
        .upsample = .{
            .{
                .weight = .fromShape(.init(.{ .in = 512, .out = 256, .k = 12 }, .f32)),
                .bias = .fromShape(.init(.{ .d = 256 }, .f32)),
                .stride = 6,
                .groups = 1,
            },
            .{
                .weight = .fromShape(.init(.{ .in = 256, .out = 128, .k = 10 }, .f32)),
                .bias = .fromShape(.init(.{ .d = 128 }, .f32)),
                .stride = 5,
                .groups = 1,
            },
            .{
                .weight = .fromShape(.init(.{ .in = 128, .out = 64, .k = 8 }, .f32)),
                .bias = .fromShape(.init(.{ .d = 64 }, .f32)),
                .stride = 4,
                .groups = 1,
            },
        },
        .residual = .{
            .{
                .conv1 = .{
                    .weight = .fromShape(.init(.{ .out = 128, .in = 256, .k = 3 }, .f32)),
                    .bias = .fromShape(.init(.{ .d = 128 }, .f32)),
                },
                .conv2 = .{
                    .weight = .fromShape(.init(.{ .out = 256, .in = 128, .k = 1 }, .f32)),
                    .bias = .fromShape(.init(.{ .d = 256 }, .f32)),
                },
            },
            .{
                .conv1 = .{
                    .weight = .fromShape(.init(.{ .out = 64, .in = 128, .k = 3 }, .f32)),
                    .bias = .fromShape(.init(.{ .d = 64 }, .f32)),
                },
                .conv2 = .{
                    .weight = .fromShape(.init(.{ .out = 128, .in = 64, .k = 1 }, .f32)),
                    .bias = .fromShape(.init(.{ .d = 128 }, .f32)),
                },
            },
            .{
                .conv1 = .{
                    .weight = .fromShape(.init(.{ .out = 32, .in = 64, .k = 3 }, .f32)),
                    .bias = .fromShape(.init(.{ .d = 32 }, .f32)),
                },
                .conv2 = .{
                    .weight = .fromShape(.init(.{ .out = 64, .in = 32, .k = 1 }, .f32)),
                    .bias = .fromShape(.init(.{ .d = 64 }, .f32)),
                },
            },
        },
        .final = .{
            .weight = .fromShape(.init(.{ .out = 1, .in = 64, .k = 3 }, .f32)),
            .bias = .fromShape(.init(.{ .d = 1 }, .f32)),
        },
    };
    const input_shape: zml.Shape = .init(.{ .b = 2, .d = 512, .s = 3 }, .f32);
    var input: [2][512][3]f32 = undefined;
    for (std.mem.bytesAsSlice(f32, std.mem.asBytes(&input))) |*value| value.* = random.float(f32) * 2 - 1;
    var input_buffer = try zml.Buffer.fromBytes(io, platform, input_shape, std.mem.asBytes(&input));
    defer input_buffer.deinit();
    var weight_buffers = try layer.bufferize(allocator);
    const tensors = try zml.meta.collectValues(zml.Tensor, allocator, &layer);
    defer allocator.free(tensors);
    const buffers = try zml.meta.collectMutPtrs(zml.Buffer, allocator, &weight_buffers);
    defer allocator.free(buffers);
    var loaded: usize = 0;
    defer for (buffers[0..loaded]) |buffer| buffer.deinit();
    for (tensors, buffers) |tensor, buffer| {
        const values = try zml.Slice.alloc(allocator, tensor.shape());
        defer values.free(allocator);
        for (values.items(f32)) |*value| value.* = (random.float(f32) * 2 - 1) * 0.05;
        buffer.* = try zml.Buffer.fromSlice(io, platform, values);
        loaded += 1;
    }
    var exe = try platform.compile(allocator, io, layer, .forward_uncached, .{zml.Tensor.fromShape(input_shape)}, .{});
    defer exe.deinit();
    var output = try exe.eval(allocator, io, .{ weight_buffers, input_buffer });
    defer output.deinit();
    const output_shape: zml.Shape = .init(.{ .b = 2, .d = 1, .s = 360 }, .f32);
    try std.testing.expect(output.shape().eqlWithTags(output_shape));
    const expected = test_data.seanet;
    try zml.testing.expectClose(io, zml.Slice.init(output_shape, std.mem.asBytes(&expected)), output, .{
        .absolute_tolerance = 1e-5,
        .relative_tolerance = 1e-4,
        .minimum_close_fraction = 1,
    });
    for ([_]zml.DataType{ .f16, .bf16, .f64 }) |dtype| {
        var typed_exe = try platform.compile(allocator, io, layer, .forward_uncached, .{zml.Tensor.fromShape(input_shape.withDtype(dtype))}, .{});
        defer typed_exe.deinit();
        try std.testing.expect(typed_exe.output_shapes[0].eqlWithTags(output_shape.withDtype(dtype)));
    }
}

test "SEANet: cached" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    var rng: std.Random.DefaultPrng = .init(42);
    const random = rng.random();
    const layer: SEANet = .{
        .initial = .{
            .weight = .fromShape(.init(.{ .out = 512, .in = 512, .k = 7 }, .f32)),
            .bias = .fromShape(.init(.{ .d = 512 }, .f32)),
        },
        .upsample = .{
            .{
                .weight = .fromShape(.init(.{ .in = 512, .out = 256, .k = 12 }, .f32)),
                .bias = .fromShape(.init(.{ .d = 256 }, .f32)),
                .stride = 6,
                .groups = 1,
            },
            .{
                .weight = .fromShape(.init(.{ .in = 256, .out = 128, .k = 10 }, .f32)),
                .bias = .fromShape(.init(.{ .d = 128 }, .f32)),
                .stride = 5,
                .groups = 1,
            },
            .{
                .weight = .fromShape(.init(.{ .in = 128, .out = 64, .k = 8 }, .f32)),
                .bias = .fromShape(.init(.{ .d = 64 }, .f32)),
                .stride = 4,
                .groups = 1,
            },
        },
        .residual = .{
            .{
                .conv1 = .{
                    .weight = .fromShape(.init(.{ .out = 128, .in = 256, .k = 3 }, .f32)),
                    .bias = .fromShape(.init(.{ .d = 128 }, .f32)),
                },
                .conv2 = .{
                    .weight = .fromShape(.init(.{ .out = 256, .in = 128, .k = 1 }, .f32)),
                    .bias = .fromShape(.init(.{ .d = 256 }, .f32)),
                },
            },
            .{
                .conv1 = .{
                    .weight = .fromShape(.init(.{ .out = 64, .in = 128, .k = 3 }, .f32)),
                    .bias = .fromShape(.init(.{ .d = 64 }, .f32)),
                },
                .conv2 = .{
                    .weight = .fromShape(.init(.{ .out = 128, .in = 64, .k = 1 }, .f32)),
                    .bias = .fromShape(.init(.{ .d = 128 }, .f32)),
                },
            },
            .{
                .conv1 = .{
                    .weight = .fromShape(.init(.{ .out = 32, .in = 64, .k = 3 }, .f32)),
                    .bias = .fromShape(.init(.{ .d = 32 }, .f32)),
                },
                .conv2 = .{
                    .weight = .fromShape(.init(.{ .out = 64, .in = 32, .k = 1 }, .f32)),
                    .bias = .fromShape(.init(.{ .d = 64 }, .f32)),
                },
            },
        },
        .final = .{
            .weight = .fromShape(.init(.{ .out = 1, .in = 64, .k = 3 }, .f32)),
            .bias = .fromShape(.init(.{ .d = 1 }, .f32)),
        },
    };
    const input_shape: zml.Shape = .init(.{ .b = 2, .d = 512, .s = 3 }, .f32);
    var input: [2][512][3]f32 = undefined;
    for (std.mem.bytesAsSlice(f32, std.mem.asBytes(&input))) |*value| value.* = random.float(f32) * 2 - 1;
    var input_buffer = try zml.Buffer.fromBytes(io, platform, input_shape, std.mem.asBytes(&input));
    defer input_buffer.deinit();
    var weight_buffers = try layer.bufferize(allocator);
    const tensors = try zml.meta.collectValues(zml.Tensor, allocator, &layer);
    defer allocator.free(tensors);
    const buffers = try zml.meta.collectMutPtrs(zml.Buffer, allocator, &weight_buffers);
    defer allocator.free(buffers);
    var loaded: usize = 0;
    defer for (buffers[0..loaded]) |buffer| buffer.deinit();
    for (tensors, buffers) |tensor, buffer| {
        const values = try zml.Slice.alloc(allocator, tensor.shape());
        defer values.free(allocator);
        for (values.items(f32)) |*value| value.* = (random.float(f32) * 2 - 1) * 0.05;
        buffer.* = try zml.Buffer.fromSlice(io, platform, values);
        loaded += 1;
    }

    var uncached_exe = try platform.compile(allocator, io, layer, .forward_uncached, .{zml.Tensor.fromShape(input_shape)}, .{});
    defer uncached_exe.deinit();
    var uncached_output = try uncached_exe.eval(allocator, io, .{ weight_buffers, input_buffer });
    defer uncached_output.deinit();
    const cache: Cache = layer.init_cache(2, .f32);
    var cache_buffers: zml.Bufferized(Cache) = undefined;
    const cache_tensors = try zml.meta.collectValues(zml.Tensor, allocator, &cache);
    defer allocator.free(cache_tensors);
    const cache_targets = try zml.meta.collectMutPtrs(zml.Buffer, allocator, &cache_buffers);
    defer allocator.free(cache_targets);
    var initialized: usize = 0;
    defer for (cache_targets[0..initialized]) |buffer| buffer.deinit();
    for (cache_tensors, cache_targets) |tensor, buffer| {
        const zeros = try zml.Slice.alloc(allocator, tensor.shape());
        defer zeros.free(allocator);
        @memset(zeros.data(), 0);
        buffer.* = try .fromSlice(io, platform, zeros);
        initialized += 1;
    }
    var cached_output: [2][1][360]f32 = undefined;
    inline for (.{ .{ .start = 0, .end = 1 }, .{ .start = 1, .end = 3 } }) |step| {
        const count = step.end - step.start;
        const chunk_shape: zml.Shape = .init(.{ .b = 2, .d = 512, .s = count }, .f32);
        var chunk: [2][512][count]f32 = undefined;
        for (&chunk, input) |*batch, source| {
            for (batch, source) |*channel, values| @memcpy(channel, values[step.start..step.end]);
        }
        var chunk_buffer = try zml.Buffer.fromBytes(io, platform, chunk_shape, std.mem.asBytes(&chunk));
        defer chunk_buffer.deinit();
        var exe = try platform.compile(allocator, io, layer, .forward_cached, .{ zml.Tensor.fromShape(chunk_shape), cache }, .{});
        defer exe.deinit();
        var runner = try exe.runner(allocator);
        defer runner.deinit(allocator);
        var result: zml.Bufferized(Cached_Output) = undefined;
        runner.run(io, .{ weight_buffers, chunk_buffer, cache_buffers }, .{&result}, .{ .wait = true });
        defer result.hidden.deinit();
        zml.Buffer.deinitAll(Cache, &cache_buffers);
        cache_buffers = result.cache;
        const values = try result.hidden.getValue([2][1][count * 120]f32, io);
        for (&cached_output, values) |*batch, source| {
            for (batch, source) |*channel, samples| @memcpy(channel[step.start * 120 .. step.end * 120], &samples);
        }
    }
    const output_shape: zml.Shape = .init(.{ .b = 2, .d = 1, .s = 360 }, .f32);
    try std.testing.expect(uncached_output.shape().eqlWithTags(output_shape));
    try zml.testing.expectClose(io, uncached_output, zml.Slice.init(output_shape, std.mem.asBytes(&cached_output)), .{
        .absolute_tolerance = 1e-5,
        .relative_tolerance = 1e-4,
        .minimum_close_fraction = 1,
    });
}
