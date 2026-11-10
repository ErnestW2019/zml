const std = @import("std");
const zml = @import("zml");
const util = @import("../util.zig");
const model = @import("model.zig");
const Model = model.Pocket_TTS_Audio_Tokenizer_Model;

test {
    std.testing.refAllDecls(@This());
    _ = @import("convolution.zig");
    _ = @import("transformer.zig");
    _ = @import("seanet.zig");
}

pub const sampling_rate = 24_000;
pub const samples_per_frame = 1920;

/// Owns Mimi's parameters and the matching Phonon latent normalization statistics.
/// The platform is borrowed and must outlive these buffers.
const Weights = struct {
    module: Model,
    buffers: zml.Bufferized(Model),

    mean: [32]f32,
    stddev: [32]f32,

    fn init(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform) !Weights {
        const here = @typeName(@This()) ++ ".init";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        const path = try util.assetPath(allocator, util.gradium_phonon_mimi);
        defer allocator.free(path);
        var registry = try zml.safetensors.TensorRegistry.fromPath(allocator, io, path);
        defer registry.deinit();
        var store: zml.io.TensorStore = .fromRegistry(allocator, &registry, &.{});
        defer store.deinit();

        const module: Model = .init(store.view());
        var buffers = try module.load(allocator, io, platform, &store);
        errdefer Model.unloadBuffers(&buffers);

        const phonon_path = try util.assetPath(allocator, util.gradium_phonon_31m_en);
        defer allocator.free(phonon_path);
        var phonon_registry = try zml.safetensors.TensorRegistry.fromPath(allocator, io, phonon_path);
        defer phonon_registry.deinit();

        return .{
            .module = module,
            .buffers = buffers,
            .mean = try read_statistic(allocator, io, &phonon_registry, "flow_lm.emb_mean"),
            .stddev = try read_statistic(allocator, io, &phonon_registry, "flow_lm.emb_std"),
        };
    }

    fn deinit(self: *Weights) void {
        Model.unloadBuffers(&self.buffers);
    }

    fn read_statistic(
        allocator: std.mem.Allocator,
        io: std.Io,
        registry: *const zml.safetensors.TensorRegistry,
        name: []const u8,
    ) ![32]f32 {
        const here = @typeName(@This()) ++ ".read_statistic";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        const tensor = registry.tensors.get(name) orelse return error.MissingLatentStatistics;
        if (tensor.shape.rank() != 1 or tensor.shape.dim(0) != 32) return error.InvalidLatentStatistics;
        const values = try util.readTensorSlice(allocator, io, tensor);
        defer values.free(allocator);
        var result: [32]f32 = undefined;
        switch (values.dtype()) {
            .f32 => @memcpy(&result, values.constItems(f32)),
            .bf16 => for (&result, values.constItems(zml.floats.BFloat16)) |*dst, src| {
                dst.* = src.toF32();
            },
            .f16 => for (&result, values.constItems(f16)) |*dst, src| {
                dst.* = @floatCast(src);
            },
            .f64 => for (&result, values.constItems(f64)) |*dst, src| {
                dst.* = @floatCast(src);
            },
            else => return error.InvalidLatentStatistics,
        }
        return result;
    }

    /// Converts normalized TTS [B,S,32] host f32 values into Mimi's codec space.
    /// The returned allocation belongs to the caller. No batch/time axes mix.
    fn denormalize(self: *const Weights, allocator: std.mem.Allocator, frame: zml.Slice) !zml.Slice {
        if (frame.dtype() != .f32 or frame.shape.rank() != 3 or frame.shape.dim(0) <= 0 or frame.shape.dim(1) <= 0 or frame.shape.dim(2) != 32 or !frame.isContiguous()) return error.InvalidLatents;

        const here = @typeName(@This()) ++ ".denormalize";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        const output = try zml.Slice.alloc(allocator, frame.shape.withTags(.{ .b, .s, .ld }));
        for (output.items(f32), frame.constItems(f32), 0..) |*dst, src, index| dst.* = src * self.stddev[index % 32] + self.mean[index % 32];

        return output;
    }
};

/// Host driver for complete-sequence audio decoding. Owns checkpoint buffers;
/// each decode compiles for the supplied sequence shape and returns owned audio.
pub const Audio_Decoder_Uncached = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    weights: Weights,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        batch: i64,
        chunk_frames: i64,
        max_frames: i64,
    ) !Audio_Decoder_Uncached {
        _ = batch; // autofix
        _ = chunk_frames; // autofix
        _ = max_frames; // autofix

        const here = @typeName(@This()) ++ ".init";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        return .{
            .allocator = allocator,
            .io = io,
            .platform = platform,
            .weights = try .init(allocator, io, platform),
        };
    }
    pub fn deinit(self: *Audio_Decoder_Uncached) void {
        self.weights.deinit();
    }

    /// `input`: contiguous normalized f32 TTS latents [B,S,32], from generation.
    /// Returns owned [B,1,S*1920] f32 samples at 24 kHz; caller frees the Slice.
    pub fn decode(self: *const Audio_Decoder_Uncached, frame: zml.Slice) !zml.Slice {
        const here = @typeName(@This()) ++ ".decode";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        const latents = try self.weights.denormalize(self.allocator, frame);
        defer latents.free(self.allocator);

        var buffer = try zml.Buffer.fromSlice(self.io, self.platform, latents);
        defer buffer.deinit();
        var exe = try self.platform.compile(
            self.allocator,
            self.io,
            self.weights.module,
            .forward_uncached,
            .{zml.Tensor.fromShape(latents.shape)},
            .{
                .program_name = @typeName(@This()),
            },
        );
        defer exe.deinit();
        var output = try exe.eval(self.allocator, self.io, .{ self.weights.buffers, buffer });
        defer output.deinit();
        return output.toSliceAlloc(self.allocator, self.io);
    }
};

/// Host driver for contiguous fixed-size latent chunks. Compiles once and owns
/// convolution overlap plus both transformer layers' K/V history. Its capacity
/// is measured in latent frames; all rows advance together. No flush is needed:
/// each decode returns exactly chunk_frames*1920 finalized samples per row.
pub const Audio_Decoder_Cached = struct {
    allocator: std.mem.Allocator,
    io: std.Io,

    platform: *const zml.Platform,
    executable: zml.Exe,

    weights: Weights,
    cache: zml.Bufferized(model.Decoder_Cache),

    batch: i64,
    chunk_frames: i64,
    max_frames: i64,

    frames: i64 = 0,
    failed: bool = false,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        batch: i64,
        chunk_frames: i64,
        max_frames: i64,
    ) !Audio_Decoder_Cached {
        if (batch <= 0 or chunk_frames <= 0 or max_frames < chunk_frames or max_frames > @divTrunc(std.math.maxInt(i32), 16))
            return error.InvalidDecoderCapacity;

        var weights: Weights = try .init(allocator, io, platform);
        errdefer weights.deinit();
        const cache = weights.module.decoder.init_cache(batch, max_frames * 16, .f32);
        var cache_buffers = try cache.initBuffers(allocator, io, platform);
        errdefer zml.Buffer.deinitAll(model.Decoder_Cache, &cache_buffers);
        const shape: zml.Shape = .init(.{ .b = batch, .s = chunk_frames, .ld = 32 }, .f32);
        var executable = try platform.compile(allocator, io, weights.module, .forward_cached, .{
            zml.Tensor.fromShape(shape), zml.Tensor.fromShape(.scalar(.i32)), cache,
        }, .{
            .program_name = @typeName(@This()),
        });
        errdefer executable.deinit();
        return .{
            .allocator = allocator,
            .io = io,
            .platform = platform,
            .weights = weights,
            .cache = cache_buffers,
            .executable = executable,
            .batch = batch,
            .chunk_frames = chunk_frames,
            .max_frames = max_frames,
        };
    }

    pub fn deinit(self: *Audio_Decoder_Cached) void {
        self.executable.deinit();
        zml.Buffer.deinitAll(model.Decoder_Cache, &self.cache);
        self.weights.deinit();
    }

    /// `input`: normalized f32 [batch,chunk_frames,32], the next contiguous chunk
    /// from TTS. Returns owned [batch,1,chunk_frames*1920] f32 waveform samples.
    /// Caller frees the Slice. Capacity/shape errors leave history unchanged;
    /// failures after execution invalidate this decoder to prevent unsafe retries.
    pub fn decode(self: *Audio_Decoder_Cached, frame: zml.Slice) !zml.Slice {
        const here = @typeName(@This()) ++ ".decode";

        if (self.failed) return error.DecoderFailed;
        if (self.frames + self.chunk_frames > self.max_frames) return error.DecoderCapacityExceeded;
        if (frame.shape.rank() != 3 or frame.shape.dim(0) != self.batch or frame.shape.dim(1) != self.chunk_frames) return error.InvalidLatents;

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        const latents = try self.weights.denormalize(self.allocator, frame);
        defer latents.free(self.allocator);

        var buffer = try zml.Buffer.fromSlice(self.io, self.platform, latents);
        defer buffer.deinit();

        const position: i32 = @intCast(self.frames * 16);
        var position_buffer = try zml.Buffer.fromBytes(self.io, self.platform, .scalar(.i32), std.mem.asBytes(&position));
        defer position_buffer.deinit();

        var runner = try self.executable.runner(self.allocator);
        defer runner.deinit(self.allocator);

        var result: zml.Bufferized(model.Cached_Audio) = undefined;
        runner.run(self.io, .{ self.weights.buffers, buffer, position_buffer, self.cache }, .{&result}, .{ .wait = true });
        defer result.audio.deinit();
        zml.Buffer.deinitAll(model.Decoder_Cache, &self.cache);

        self.cache = result.cache;
        self.frames += self.chunk_frames;
        errdefer self.failed = true;

        return result.audio.toSliceAlloc(self.allocator, self.io);
    }
};

test Audio_Decoder_Uncached {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    var decoder = try Audio_Decoder_Uncached.init(allocator, io, platform, 1, 23, 23);
    defer decoder.deinit();
    const latents = @import("../phonon/main_test_data.zig").audio_latents;
    const input = zml.Slice.init(.init(.{ .b = 1, .s = 23, .ld = 32 }, .f32), std.mem.asBytes(&latents));
    const output = try decoder.decode(input);
    defer output.free(allocator);
    const output_shape: zml.Shape = .init(.{ .b = 1, .d = 1, .s = 44160 }, .f32);
    try std.testing.expect(output.shape.eqlWithTags(output_shape));
    const expected = @import("root_test_data.zig").audio;
    try zml.testing.expectClose(io, zml.Slice.init(output_shape, std.mem.asBytes(&expected)), output, .{
        .absolute_tolerance = 1e-4,
        .relative_tolerance = 1e-4,
        .minimum_close_fraction = 1,
    });
}

test Audio_Decoder_Cached {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    var uncached = try Audio_Decoder_Uncached.init(allocator, io, platform, 1, 23, 23);
    defer uncached.deinit();
    const latents = @import("../phonon/main_test_data.zig").audio_latents;
    const input = zml.Slice.init(.init(.{ .b = 1, .s = 23, .ld = 32 }, .f32), std.mem.asBytes(&latents));
    const expected = try uncached.decode(input);
    defer expected.free(allocator);
    var cached = try Audio_Decoder_Cached.init(allocator, io, platform, 1, 1, 23);
    defer cached.deinit();
    const actual = try zml.Slice.alloc(allocator, expected.shape);
    defer actual.free(allocator);
    try std.testing.expectError(error.InvalidLatents, cached.decode(input));
    try std.testing.expectEqual(0, cached.frames);
    for (latents, 0..) |frame, index| {
        const chunk = zml.Slice.init(.init(.{ .b = 1, .s = 1, .ld = 32 }, .f32), std.mem.asBytes(&frame));
        const output = try cached.decode(chunk);
        defer output.free(allocator);
        try std.testing.expectEqual(samples_per_frame, output.shape.dim(.s));
        @memcpy(actual.items(f32)[index * samples_per_frame ..][0..samples_per_frame], output.constItems(f32));
    }
    try std.testing.expectEqual(23, cached.frames);
    try std.testing.expectError(error.DecoderCapacityExceeded, cached.decode(input));
    try zml.testing.expectClose(io, expected, actual, .{
        .absolute_tolerance = 1e-4,
        .relative_tolerance = 1e-4,
        .minimum_close_fraction = 1,
    });
}
