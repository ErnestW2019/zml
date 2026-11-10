const std = @import("std");
const zml = @import("zml");
const test_data = @import("convolution_test_data.zig");

test {
    std.testing.refAllDecls(@This());
}

/// Features for the new chunk and the history needed by the next chunk.
/// Both use channels-first tags [.b, .d, .s]; cache retains its storage dtype.
pub const Cached_Output = struct { hidden: zml.Tensor, cache: zml.Tensor };

/// Mimi's causal, zero-padded, stride-one convolution. Checkpoint weights use
/// [output channels, input channels, kernel taps]; all decoder convolutions have
/// dilation one. The first sample sees zeros to its left, never future samples.
pub const Conv1d = struct {
    weight: zml.Tensor,
    bias: zml.Tensor,

    /// Expects a PocketTTSAudioTokenizerConv1d checkpoint view (above `conv`).
    pub fn init(store: zml.io.TensorStore.View) Conv1d {
        const here = @typeName(@This()) ++ ".init";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        return .{
            .weight = store.createReplicatedTensor("conv.weight", .{ .out, .in, .k }),
            .bias = store.createReplicatedTensor("conv.bias", .{.d}),
        };
    }

    /// `input`: floating [B, C_in, S], from the preceding SEANet activation.
    /// Returns [B, C_out, S] in input.dtype(). Tags: .b independent sequences,
    /// .d channels, .s time; weight .in is contracted, .out becomes .d, .k taps.
    pub fn forward_uncached(self: Conv1d, input: zml.Tensor) zml.Tensor {
        const here = @typeName(@This()) ++ ".forward_uncached";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        const features = input.withTags(.{ .b, .d, .s });
        std.debug.assert(features.dtype().isFloat());
        const padded = features.pad(0, .{ .s = zml.Tensor.Pad{ .low = self.weight.dim(.k) - 1 } });
        return self.convolve(padded);
    }

    /// `input`: floating [B, C_in, Q], new contiguous SEANet features.
    /// `cache`: floating [B, C_in, K-1], the preceding input samples, initially
    /// zero. Returns Q outputs and the last K-1 inputs for the next call. Input
    /// controls compute/output dtype; cache preserves its own dtype. Tags .b/.d/.s.
    pub fn forward_cached(self: Conv1d, input: zml.Tensor, cache: zml.Tensor) Cached_Output {
        const here = @typeName(@This()) ++ ".forward_cached";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        const features = input.withTags(.{ .b, .d, .s });
        std.debug.assert(features.dtype().isFloat() and cache.dtype().isFloat());
        std.debug.assert(cache.dim(.s) == self.weight.dim(.k) - 1);

        const with_history = zml.Tensor.concatenate(&.{ cache.convert(features.dtype()), features }, .s);
        const output = self.convolve(with_history);
        const next_history = with_history.slice(.s, .{ .start = features.dim(.s) }).convert(cache.dtype());
        return .{ .hidden = output, .cache = next_history };
    }

    fn convolve(self: Conv1d, input: zml.Tensor) zml.Tensor {
        const y = input.conv1d(self.weight.convert(input.dtype()), .{}).withTags(.{ .b, .d, .s });
        return y.add(self.bias.convert(input.dtype()).broad(y.shape()));
    }

    pub fn cache_shape(self: Conv1d, batch: i64, dtype: zml.DataType) zml.Shape {
        const here = @typeName(@This()) ++ ".cache_shape";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        return .init(.{ .b = batch, .d = self.weight.dim(.in), .s = self.weight.dim(.k) - 1 }, dtype);
    }
};

/// Causal transposed convolution: inserts stride-1 zeros between input samples,
/// convolves with the reversed kernel, and trims K-stride samples from the right.
/// Used for depthwise 16x latent upsampling and dense 6x, 5x, 4x SEANet upsampling.
/// Checkpoint layout is [C_in, C_out/groups, K], unlike Conv1d's weight layout.
pub const ConvTranspose1d = struct {
    weight: zml.Tensor,
    bias: ?zml.Tensor,
    stride: i64,
    groups: i64 = 1,

    pub fn init(store: zml.io.TensorStore.View, stride: i64, groups: i64) ConvTranspose1d {
        const here = @typeName(@This()) ++ ".init";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        return .{
            .weight = store.createReplicatedTensor("conv.weight", .{ .in, .out, .k }),
            .bias = if (store.hasKey("conv.bias")) store.createReplicatedTensor("conv.bias", .{.d}) else null,
            .stride = stride,
            .groups = groups,
        };
    }

    /// `input`: floating [B, C_in, S], from latent projection or a SEANet ELU.
    /// Returns [B, C_out, S*stride] in input.dtype(). Tags .b batch, .d channels,
    /// .s time; weight .in/.out describe channels and .k the kernel's time taps.
    pub fn forward_uncached(self: ConvTranspose1d, input: zml.Tensor) zml.Tensor {
        const here = @typeName(@This()) ++ ".forward_uncached";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        const features = input.withTags(.{ .b, .d, .s });
        const raw_output = self.convolve(features);
        const finalized_length = features.dim(.s) * self.stride;
        const finalized = raw_output.slice(.s, .{ .end = finalized_length });
        return self.add_bias(finalized);
    }

    /// `input`: floating [B, C_in, Q], new contiguous input samples.
    /// `cache`: floating [B, C_out, K-stride], the previous chunk's unfinished
    /// convolution tail, initially zero. It excludes bias so overlap is added
    /// exactly once. Returns Q*stride finalized samples and the next tail.
    /// Compute/output follow input.dtype(); cache retains its dtype; tags .b/.d/.s.
    pub fn forward_cached(self: ConvTranspose1d, input: zml.Tensor, cache: zml.Tensor) Cached_Output {
        const here = @typeName(@This()) ++ ".forward_cached";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        const features = input.withTags(.{ .b, .d, .s });
        const finalized_length = features.dim(.s) * self.stride;
        std.debug.assert(cache.dtype().isFloat() and cache.dim(.s) == self.weight.dim(.k) - self.stride);

        const raw_output = self.convolve(features);
        const previous_overlap = cache.convert(features.dtype()).pad(0, .{ .s = zml.Tensor.Pad{ .high = finalized_length } });
        const combined = raw_output.add(previous_overlap);
        const finalized = combined.slice(.s, .{ .end = finalized_length });
        const next_overlap = combined.slice(.s, .{ .start = finalized_length }).convert(cache.dtype());
        return .{ .hidden = self.add_bias(finalized), .cache = next_overlap };
    }

    fn convolve(self: ConvTranspose1d, input: zml.Tensor) zml.Tensor {
        std.debug.assert(input.dtype().isFloat() and input.dim(.s) > 0);
        std.debug.assert(self.stride > 0 and self.weight.dim(.k) >= self.stride);
        const kernel = self.convolution_kernel(input.dtype());
        const padding = self.weight.dim(.k) - 1;
        return input.conv1d(kernel, .{
            .lhs_dilation = self.stride,
            .padding = &.{ padding, padding },
            .window_reversal = true,
            .feature_group_count = self.groups,
        }).withTags(.{ .b, .d, .s });
    }

    /// PyTorch [C_in,C_out/groups,K] -> StableHLO [C_out,C_in/groups,K].
    /// Swap channel axes within each group so groups remain independent.
    fn convolution_kernel(self: ConvTranspose1d, dtype: zml.DataType) zml.Tensor {
        const channels_per_group = @divExact(self.weight.dim(.in), self.groups);
        const grouped = self.weight.convert(dtype).splitAxis(.in, .{ .g = self.groups, .ci = channels_per_group });
        const output_first = grouped.transpose(.{ .g, .out, .ci, .k });
        return output_first.merge(.{ .co = .{ .g, .out } });
    }

    fn add_bias(self: ConvTranspose1d, input: zml.Tensor) zml.Tensor {
        return if (self.bias) |bias| input.add(bias.convert(input.dtype()).broad(input.shape())) else input;
    }

    pub fn cache_shape(self: ConvTranspose1d, batch: i64, dtype: zml.DataType) zml.Shape {
        const here = @typeName(@This()) ++ ".cache_shape";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        return .init(.{ .b = batch, .d = self.weight.dim(.out) * self.groups, .s = self.weight.dim(.k) - self.stride }, dtype);
    }
};

/// SEANet residual branch: ELU, causal K=3 convolution reducing channels by two,
/// ELU, K=1 convolution restoring channels, then the identity skip connection.
pub const Resnet_Block = struct {
    conv1: Conv1d,
    conv2: Conv1d,

    pub fn init(store: zml.io.TensorStore.View) Resnet_Block {
        const here = @typeName(@This()) ++ ".init";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        return .{ .conv1 = .init(store.withPrefix("block.1")), .conv2 = .init(store.withPrefix("block.3")) };
    }

    /// `input`: floating SEANet features [B, C, S] from an upsampling convolution.
    /// Returns the same shape and dtype, assigning tags .b batch, .d channels, .s time.
    pub fn forward_uncached(self: Resnet_Block, input: zml.Tensor) zml.Tensor {
        const here = @typeName(@This()) ++ ".forward_uncached";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        const residual = input.withTags(.{ .b, .d, .s });
        const reduced = self.conv1.forward_uncached(zml.nn.elu(residual, 1));
        const restored = self.conv2.forward_uncached(zml.nn.elu(reduced, 1));
        return residual.add(restored);
    }

    /// `input`: floating [B, C, Q], new features; `cache`: floating [B, C, 2],
    /// preceding post-ELU inputs to conv1, initially zero. K=1 conv2 needs no
    /// history. Returns the residual output and updated cache, tags .b/.d/.s.
    pub fn forward_cached(self: Resnet_Block, input: zml.Tensor, cache: zml.Tensor) Cached_Output {
        const here = @typeName(@This()) ++ ".forward_cached";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        const residual = input.withTags(.{ .b, .d, .s });
        const reduced = self.conv1.forward_cached(zml.nn.elu(residual, 1), cache);
        const restored = self.conv2.forward_uncached(zml.nn.elu(reduced.hidden, 1));
        return .{ .hidden = residual.add(restored), .cache = reduced.cache };
    }
};

test "Conv1d" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    var rng: std.Random.DefaultPrng = .init(42);
    const random = rng.random();
    const layer: Conv1d = .{
        .weight = .fromShape(.init(.{ .out = 3, .in = 4, .k = 7 }, .f32)),
        .bias = .fromShape(.init(.{ .d = 3 }, .f32)),
    };
    const input_shape: zml.Shape = .init(.{ .b = 2, .d = 4, .s = 3 }, .f32);
    var input: [2][4][3]f32 = undefined;
    for (std.mem.bytesAsSlice(f32, std.mem.asBytes(&input))) |*value| value.* = random.float(f32) * 2 - 1;
    var input_buffer = try zml.Buffer.fromBytes(io, platform, input_shape, std.mem.asBytes(&input));
    defer input_buffer.deinit();
    var weight_buffers = try zml.mem.bufferize(allocator, Conv1d, &layer);
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
    const output_shape: zml.Shape = .init(.{ .b = 2, .d = 3, .s = 3 }, .f32);
    try std.testing.expect(output.shape().eqlWithTags(output_shape));
    const expected = test_data.conv1d;
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

test "ConvTranspose1d: dense" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    var rng: std.Random.DefaultPrng = .init(42);
    const random = rng.random();
    const layer: ConvTranspose1d = .{
        .weight = .fromShape(.init(.{ .in = 4, .out = 3, .k = 12 }, .f32)),
        .bias = .fromShape(.init(.{ .d = 3 }, .f32)),
        .stride = 6,
    };
    const input_shape: zml.Shape = .init(.{ .b = 2, .d = 4, .s = 3 }, .f32);
    var input: [2][4][3]f32 = undefined;
    for (std.mem.bytesAsSlice(f32, std.mem.asBytes(&input))) |*value| value.* = random.float(f32) * 2 - 1;
    var input_buffer = try zml.Buffer.fromBytes(io, platform, input_shape, std.mem.asBytes(&input));
    defer input_buffer.deinit();
    var weight_buffers = try zml.mem.bufferize(allocator, ConvTranspose1d, &layer);
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
    const output_shape: zml.Shape = .init(.{ .b = 2, .d = 3, .s = 18 }, .f32);
    try std.testing.expect(output.shape().eqlWithTags(output_shape));
    const expected = test_data.transpose_dense;
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

test "ConvTranspose1d: depthwise" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    var rng: std.Random.DefaultPrng = .init(42);
    const random = rng.random();
    const layer: ConvTranspose1d = .{
        .weight = .fromShape(.init(.{ .in = 4, .out = 1, .k = 32 }, .f32)),
        .bias = null,
        .stride = 16,
        .groups = 4,
    };
    const input_shape: zml.Shape = .init(.{ .b = 2, .d = 4, .s = 3 }, .f32);
    var input: [2][4][3]f32 = undefined;
    for (std.mem.bytesAsSlice(f32, std.mem.asBytes(&input))) |*value| value.* = random.float(f32) * 2 - 1;
    var input_buffer = try zml.Buffer.fromBytes(io, platform, input_shape, std.mem.asBytes(&input));
    defer input_buffer.deinit();
    var weight_buffers = try zml.mem.bufferize(allocator, ConvTranspose1d, &layer);
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
    const output_shape: zml.Shape = .init(.{ .b = 2, .d = 4, .s = 48 }, .f32);
    try std.testing.expect(output.shape().eqlWithTags(output_shape));
    const expected = test_data.transpose_depthwise;
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

test "Resnet_Block" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    var rng: std.Random.DefaultPrng = .init(42);
    const random = rng.random();
    const layer: Resnet_Block = .{
        .conv1 = .{
            .weight = .fromShape(.init(.{ .out = 2, .in = 4, .k = 3 }, .f32)),
            .bias = .fromShape(.init(.{ .d = 2 }, .f32)),
        },
        .conv2 = .{
            .weight = .fromShape(.init(.{ .out = 4, .in = 2, .k = 1 }, .f32)),
            .bias = .fromShape(.init(.{ .d = 4 }, .f32)),
        },
    };
    const input_shape: zml.Shape = .init(.{ .b = 2, .d = 4, .s = 3 }, .f32);
    var input: [2][4][3]f32 = undefined;
    for (std.mem.bytesAsSlice(f32, std.mem.asBytes(&input))) |*value| value.* = random.float(f32) * 2 - 1;
    var input_buffer = try zml.Buffer.fromBytes(io, platform, input_shape, std.mem.asBytes(&input));
    defer input_buffer.deinit();
    var weight_buffers = try zml.mem.bufferize(allocator, Resnet_Block, &layer);
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
    const output_shape: zml.Shape = .init(.{ .b = 2, .d = 4, .s = 3 }, .f32);
    try std.testing.expect(output.shape().eqlWithTags(output_shape));
    const expected = test_data.resnet;
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

test "Conv1d: cached" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    var rng: std.Random.DefaultPrng = .init(42);
    const random = rng.random();
    const layer: Conv1d = .{
        .weight = .fromShape(.init(.{ .out = 3, .in = 4, .k = 7 }, .f32)),
        .bias = .fromShape(.init(.{ .d = 3 }, .f32)),
    };
    const input_shape: zml.Shape = .init(.{ .b = 2, .d = 4, .s = 3 }, .f32);
    var input: [2][4][3]f32 = undefined;
    for (std.mem.bytesAsSlice(f32, std.mem.asBytes(&input))) |*value| value.* = random.float(f32) * 2 - 1;
    var input_buffer = try zml.Buffer.fromBytes(io, platform, input_shape, std.mem.asBytes(&input));
    defer input_buffer.deinit();
    var weight_buffers = try zml.mem.bufferize(allocator, Conv1d, &layer);
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
    const cache: zml.Tensor = .fromShape(layer.cache_shape(2, .f32));
    var cache_buffers: zml.Bufferized(zml.Tensor) = undefined;
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
    var cached_output: [2][3][3]f32 = undefined;
    inline for (.{ .{ .start = 0, .end = 1 }, .{ .start = 1, .end = 3 } }) |step| {
        const count = step.end - step.start;
        const chunk_shape: zml.Shape = .init(.{ .b = 2, .d = 4, .s = count }, .f32);
        var chunk: [2][4][count]f32 = undefined;
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
        zml.Buffer.deinitAll(zml.Tensor, &cache_buffers);
        cache_buffers = result.cache;
        const values = try result.hidden.getValue([2][3][count * 1]f32, io);
        for (&cached_output, values) |*batch, source| {
            for (batch, source) |*channel, samples| @memcpy(channel[step.start * 1 .. step.end * 1], &samples);
        }
    }
    const output_shape: zml.Shape = .init(.{ .b = 2, .d = 3, .s = 3 }, .f32);
    try std.testing.expect(uncached_output.shape().eqlWithTags(output_shape));
    try zml.testing.expectClose(io, uncached_output, zml.Slice.init(output_shape, std.mem.asBytes(&cached_output)), .{
        .absolute_tolerance = 1e-5,
        .relative_tolerance = 1e-4,
        .minimum_close_fraction = 1,
    });
}

test "ConvTranspose1d: dense: cached" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    var rng: std.Random.DefaultPrng = .init(42);
    const random = rng.random();
    const layer: ConvTranspose1d = .{
        .weight = .fromShape(.init(.{ .in = 4, .out = 3, .k = 12 }, .f32)),
        .bias = .fromShape(.init(.{ .d = 3 }, .f32)),
        .stride = 6,
    };
    const input_shape: zml.Shape = .init(.{ .b = 2, .d = 4, .s = 3 }, .f32);
    var input: [2][4][3]f32 = undefined;
    for (std.mem.bytesAsSlice(f32, std.mem.asBytes(&input))) |*value| value.* = random.float(f32) * 2 - 1;
    var input_buffer = try zml.Buffer.fromBytes(io, platform, input_shape, std.mem.asBytes(&input));
    defer input_buffer.deinit();
    var weight_buffers = try zml.mem.bufferize(allocator, ConvTranspose1d, &layer);
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
    const cache: zml.Tensor = .fromShape(layer.cache_shape(2, .f32));
    var cache_buffers: zml.Bufferized(zml.Tensor) = undefined;
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
    var cached_output: [2][3][18]f32 = undefined;
    inline for (.{ .{ .start = 0, .end = 1 }, .{ .start = 1, .end = 3 } }) |step| {
        const count = step.end - step.start;
        const chunk_shape: zml.Shape = .init(.{ .b = 2, .d = 4, .s = count }, .f32);
        var chunk: [2][4][count]f32 = undefined;
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
        zml.Buffer.deinitAll(zml.Tensor, &cache_buffers);
        cache_buffers = result.cache;
        const values = try result.hidden.getValue([2][3][count * 6]f32, io);
        for (&cached_output, values) |*batch, source| {
            for (batch, source) |*channel, samples| @memcpy(channel[step.start * 6 .. step.end * 6], &samples);
        }
    }
    const output_shape: zml.Shape = .init(.{ .b = 2, .d = 3, .s = 18 }, .f32);
    try std.testing.expect(uncached_output.shape().eqlWithTags(output_shape));
    try zml.testing.expectClose(io, uncached_output, zml.Slice.init(output_shape, std.mem.asBytes(&cached_output)), .{
        .absolute_tolerance = 1e-5,
        .relative_tolerance = 1e-4,
        .minimum_close_fraction = 1,
    });
}

test "ConvTranspose1d: depthwise: cached" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    var rng: std.Random.DefaultPrng = .init(42);
    const random = rng.random();
    const layer: ConvTranspose1d = .{
        .weight = .fromShape(.init(.{ .in = 4, .out = 1, .k = 32 }, .f32)),
        .bias = null,
        .stride = 16,
        .groups = 4,
    };
    const input_shape: zml.Shape = .init(.{ .b = 2, .d = 4, .s = 3 }, .f32);
    var input: [2][4][3]f32 = undefined;
    for (std.mem.bytesAsSlice(f32, std.mem.asBytes(&input))) |*value| value.* = random.float(f32) * 2 - 1;
    var input_buffer = try zml.Buffer.fromBytes(io, platform, input_shape, std.mem.asBytes(&input));
    defer input_buffer.deinit();
    var weight_buffers = try zml.mem.bufferize(allocator, ConvTranspose1d, &layer);
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
    const cache: zml.Tensor = .fromShape(layer.cache_shape(2, .f32));
    var cache_buffers: zml.Bufferized(zml.Tensor) = undefined;
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
    var cached_output: [2][4][48]f32 = undefined;
    inline for (.{ .{ .start = 0, .end = 1 }, .{ .start = 1, .end = 3 } }) |step| {
        const count = step.end - step.start;
        const chunk_shape: zml.Shape = .init(.{ .b = 2, .d = 4, .s = count }, .f32);
        var chunk: [2][4][count]f32 = undefined;
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
        zml.Buffer.deinitAll(zml.Tensor, &cache_buffers);
        cache_buffers = result.cache;
        const values = try result.hidden.getValue([2][4][count * 16]f32, io);
        for (&cached_output, values) |*batch, source| {
            for (batch, source) |*channel, samples| @memcpy(channel[step.start * 16 .. step.end * 16], &samples);
        }
    }
    const output_shape: zml.Shape = .init(.{ .b = 2, .d = 4, .s = 48 }, .f32);
    try std.testing.expect(uncached_output.shape().eqlWithTags(output_shape));
    try zml.testing.expectClose(io, uncached_output, zml.Slice.init(output_shape, std.mem.asBytes(&cached_output)), .{
        .absolute_tolerance = 1e-5,
        .relative_tolerance = 1e-4,
        .minimum_close_fraction = 1,
    });
}

test "Resnet_Block: cached" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    var rng: std.Random.DefaultPrng = .init(42);
    const random = rng.random();
    const layer: Resnet_Block = .{
        .conv1 = .{
            .weight = .fromShape(.init(.{ .out = 2, .in = 4, .k = 3 }, .f32)),
            .bias = .fromShape(.init(.{ .d = 2 }, .f32)),
        },
        .conv2 = .{
            .weight = .fromShape(.init(.{ .out = 4, .in = 2, .k = 1 }, .f32)),
            .bias = .fromShape(.init(.{ .d = 4 }, .f32)),
        },
    };
    const input_shape: zml.Shape = .init(.{ .b = 2, .d = 4, .s = 3 }, .f32);
    var input: [2][4][3]f32 = undefined;
    for (std.mem.bytesAsSlice(f32, std.mem.asBytes(&input))) |*value| value.* = random.float(f32) * 2 - 1;
    var input_buffer = try zml.Buffer.fromBytes(io, platform, input_shape, std.mem.asBytes(&input));
    defer input_buffer.deinit();
    var weight_buffers = try zml.mem.bufferize(allocator, Resnet_Block, &layer);
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
    const cache: zml.Tensor = .fromShape(layer.conv1.cache_shape(2, .f32));
    var cache_buffers: zml.Bufferized(zml.Tensor) = undefined;
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
    var cached_output: [2][4][3]f32 = undefined;
    inline for (.{ .{ .start = 0, .end = 1 }, .{ .start = 1, .end = 3 } }) |step| {
        const count = step.end - step.start;
        const chunk_shape: zml.Shape = .init(.{ .b = 2, .d = 4, .s = count }, .f32);
        var chunk: [2][4][count]f32 = undefined;
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
        zml.Buffer.deinitAll(zml.Tensor, &cache_buffers);
        cache_buffers = result.cache;
        const values = try result.hidden.getValue([2][4][count * 1]f32, io);
        for (&cached_output, values) |*batch, source| {
            for (batch, source) |*channel, samples| @memcpy(channel[step.start * 1 .. step.end * 1], &samples);
        }
    }
    const output_shape: zml.Shape = .init(.{ .b = 2, .d = 4, .s = 3 }, .f32);
    try std.testing.expect(uncached_output.shape().eqlWithTags(output_shape));
    try zml.testing.expectClose(io, uncached_output, zml.Slice.init(output_shape, std.mem.asBytes(&cached_output)), .{
        .absolute_tolerance = 1e-5,
        .relative_tolerance = 1e-4,
        .minimum_close_fraction = 1,
    });
}
