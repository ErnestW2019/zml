const std = @import("std");
const zml = @import("zml");

const KV_Cache = @import("transformer.zig").KV_Cache;
const Model = @import("model.zig").Gradium_Phonon_TTS_For_Conditional_Generation;
const Tokenizer = @import("Tokenizer.zig");
const util = @import("../util.zig");

test {
    std.testing.refAllDecls(@This());
}

// ---

/// Generates normalized f32 audio latents for one text prompt without a K/V cache.
///
/// Owns loaded model/voice buffers, prompt IDs, generated history, and RNG state.
/// The caller lends the platform and drives `step` until `done`, then consumes
/// `latents` as `[1, frames(), 32]`. Each step recomputes the full sequence using
/// `Model.forward_uncached` and applies the checkpoint's one-step Euler sampler.
/// Latents still need denormalization and Mimi decoding to become waveform audio.
pub const Text_To_Speech_Uncached = struct {
    io: std.Io,
    gpa: std.mem.Allocator,
    rng: std.Random.DefaultPrng = .init(42),

    platform: *const zml.Platform,

    model: Model,
    weights: zml.Bufferized(Model),

    voice: zml.Buffer,
    start_time: zml.Buffer,
    end_time: zml.Buffer,

    ids_list: std.ArrayList(i64),
    latents_list: std.ArrayList(f32),

    /// All IDs past, present and future.
    ///
    /// Future IDs are all zeroes.
    ids: zml.Buffer,

    /// Starts at num_tokens, ends at max_tokens.
    active_length_host: i32,
    /// Gets swapped with the reusable buffer after each execution.
    active_length: zml.Buffer,
    /// Storage space for the generated latent frame. Written to by Phonon, read from by Mimi.
    //
    // TODO: double-buffering this would make TSS<>Decoding pipelining possible -- no idea if that's useful though.
    frame: zml.Buffer,
    noise: zml.Buffer,

    num_tokens: usize,
    max_tokens: usize,
    max_frames: usize,
    frames_after_eos: usize,

    remaining: ?usize = null,
    done: bool = false,
    stopped_by_eos: bool = false,

    pub fn init(
        io: std.Io,
        gpa: std.mem.Allocator,
        arena: std.mem.Allocator,
        platform: *const zml.Platform,
        text: []const u8,
    ) !Text_To_Speech_Uncached {
        const here = @typeName(@This()) ++ ".init";

        var arena_local_impl: std.heap.ArenaAllocator = .init(gpa);
        defer arena_local_impl.deinit();
        const arena_local = arena_local_impl.allocator();

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        const tokenizer_path = try util.assetPath(arena_local, util.gradium_phonon_31m_en_tokenizer);
        var tokenizer = try Tokenizer.init(arena_local, io, tokenizer_path);
        defer tokenizer.deinit();

        const voice_path = try util.assetPath(arena, util.gradium_phonon_voice);
        const voice = try load_voice(gpa, io, voice_path);
        defer voice.free(gpa);

        const tokens = try tokenizer.encode_prompt(gpa, text, @intCast(voice.shape.dim(.p)));
        defer tokens.free(gpa);

        var max_frames: usize = 0;
        var frames_after_eos: usize = 0;
        { // TODO: this is most definitely silly

            var span_count_frames = zml.tracer.span_at(@src(), here ++ "/count_frames", .{});
            defer span_count_frames.end();

            // Preserve the checkpoint's text-length budget and short-text EOS tail.
            var text_ids: std.ArrayList(u32) = .empty;
            defer text_ids.deinit(gpa);

            for (tokens.constItems(i64)) |id| {
                if (id == 4 or id == 5 or id == 6) continue;
                try text_ids.append(gpa, @intCast(id));
            }

            var decoder = try tokenizer.inner.decoder();
            defer decoder.deinit();
            var decoded = try decoder.decodeAlloc(gpa, text_ids.items);
            defer decoded.deinit(gpa);
            max_frames = @intFromFloat(@ceil((@as(f64, @floatFromInt(text_ids.items.len)) / 3 + 2) * 12.5));

            var words = std.mem.tokenizeAny(u8, decoded.items, " \t\r\n\x0b\x0c");
            var word_count: usize = 0;
            while (words.next() != null) word_count += 1;
            if (word_count <= 4) frames_after_eos = 3;
        }

        var span_load_registry = zml.tracer.span_at(@src(), here ++ "/load_registry", .{});
        {}
        const model_path = try util.assetPath(gpa, util.gradium_phonon_31m_en);
        defer gpa.free(model_path);
        var registry = try zml.safetensors.TensorRegistry.fromPath(gpa, io, model_path);
        defer registry.deinit();
        var store: zml.io.TensorStore = .fromRegistry(gpa, &registry, &.{});
        defer store.deinit();
        {}
        span_load_registry.end();

        const model: Model = .init(store.view());
        var weights = try model.load(gpa, io, platform, &store);
        errdefer Model.unload(&weights);

        var voice_buffer = blk: {
            var span_upload_voice = zml.tracer.span_at(@src(), here ++ "/upload_voice", .{});
            defer span_upload_voice.end();

            break :blk try zml.Buffer.fromSlice(io, platform, voice);
        };
        errdefer voice_buffer.deinit();

        const time_shape: zml.Shape = .init(.{ .b = 1, .t = 1 }, .f32);
        var start_time = try zml.Buffer.fromBytes(io, platform, time_shape, std.mem.asBytes(&@as([1]f32, .{0})));
        errdefer start_time.deinit();
        var end_time = try zml.Buffer.fromBytes(io, platform, time_shape, std.mem.asBytes(&@as([1]f32, .{1})));
        errdefer end_time.deinit();

        const num_tokens: usize = tokens.constItems(i64).len;
        const max_tokens: usize = try std.math.add(usize, num_tokens, max_frames);

        // Reserve the complete budget so committing a completed frame cannot allocate.
        var ids_list = try std.ArrayList(i64).initCapacity(gpa, max_tokens);
        errdefer ids_list.deinit(gpa);
        ids_list.appendSliceAssumeCapacity(tokens.constItems(i64));

        var latents_list = try std.ArrayList(f32).initCapacity(gpa, try std.math.mul(usize, max_frames, 32));
        errdefer latents_list.deinit(gpa);

        const active_length_host = num_tokens;
        const active_length = try zml.Buffer.scalar(io, platform, num_tokens, .i32);

        const ids = blk: {
            const shape = zml.Shape.init(.{ .b = 1, .s = max_tokens }, .i64);

            const zeroes = try zml.Slice.alloc(gpa, shape);
            defer zeroes.free(gpa);
            @memset(zeroes.items(i64), 0);
            @memcpy(zeroes.items(i64)[0..num_tokens], tokens.constItems(i64));

            break :blk try zml.Buffer.fromSlice(io, platform, zeroes);
        };

        const frame = try zml.Buffer.uninitialized(io, platform, .init(.{ .b = 1, .s = 1, .ld = 32 }, .f32), .{});

        // const ids = zml.Slice.init(.init(.{ .b = 1, .s = self.ids.items.len }, .i64), std.mem.sliceAsBytes(self.ids.items));
        // const history = zml.Slice.init(.init(.{ .b = 1, .a = self.frames(), .ld = 32 }, .f32), std.mem.sliceAsBytes(self.latents.items));
        // const candidate = zml.Slice.init(.init(.{ .b = 1, .ld = 32 }, .f32), std.mem.asBytes(&noise));
        //
        // var ids_buffer = try zml.Buffer.fromSlice(io, self.platform, ids);
        // defer ids_buffer.deinit();
        // var history_buffer = try zml.Buffer.fromSlice(io, self.platform, history);
        // defer history_buffer.deinit();
        // var candidate_buffer = try zml.Buffer.fromSlice(io, self.platform, candidate);
        // defer candidate_buffer.deinit();

        // var exe = try platform.compile(gpa, io, model, .forward_uncached, .{
        //     zml.Tensor.fromShape(ids_list.shape()),
        //     zml.Tensor.fromShape(voice.shape()),
        //     zml.Tensor.fromShape(history.shape()),
        //     zml.Tensor.fromShape(candidate.shape()),
        //     zml.Tensor.fromShape(start_time.shape()),
        //     zml.Tensor.fromShape(end_time.shape()),
        // }, .{
        //     .program_name = @typeName(@This()),
        // });
        // defer exe.deinit();

        return .{
            .gpa = gpa,
            .io = io,
            .platform = platform,
            .model = model,
            .weights = weights,
            .voice = voice_buffer,
            .start_time = start_time,
            .end_time = end_time,
            .ids_list = ids_list,
            .latents_list = latents_list,
            .num_tokens = num_tokens,
            .max_tokens = max_tokens,
            .max_frames = max_frames,
            .frames_after_eos = frames_after_eos,

            // FIX: them
            .ids = ids,
            .active_length_host = @intCast(active_length_host),
            .active_length = active_length,
            .frame = frame,
            .noise = undefined,
        };
    }

    pub fn deinit(self: *Text_To_Speech_Uncached) void {
        const here = @typeName(@This()) ++ ".deinit";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        self.latents_list.deinit(self.gpa);
        self.ids_list.deinit(self.gpa);
        self.end_time.deinit();
        self.start_time.deinit();
        self.voice.deinit();
        Model.unload(&self.weights);

        self.ids.deinit();
        self.active_length.deinit();
        self.frame.deinit();
    }

    pub fn frames(self: *const Text_To_Speech_Uncached) usize {
        return self.latents_list.items.len / 32;
    }

    pub const Step_Result = struct {
        eos_score: f32,
        frame: zml.Buffer,
    };

    /// Generates and retains one normalized frame, returning its raw EOS score.
    pub fn step(self: *Text_To_Speech_Uncached) !Step_Result {
        if (self.done) return error.GenerationFinished;

        const here = @typeName(@This()) ++ ".step";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        const gpa = self.gpa;
        const io = self.io;

        // Commit RNG state together with the completed frame, after successful execution.
        var rng = self.rng;
        const random = rng.random();
        var noise: [32]f32 = undefined;
        for (&noise) |*value| value.* = random.floatNorm(f32) * @sqrt(@as(f32, 0.3));

        const ids = zml.Slice.init(.init(.{ .b = 1, .s = self.ids_list.items.len }, .i64), std.mem.sliceAsBytes(self.ids_list.items));
        const history = zml.Slice.init(.init(.{ .b = 1, .a = self.frames(), .ld = 32 }, .f32), std.mem.sliceAsBytes(self.latents_list.items));
        const candidate = zml.Slice.init(.init(.{ .b = 1, .ld = 32 }, .f32), std.mem.asBytes(&noise));

        var ids_buffer = try zml.Buffer.fromSlice(io, self.platform, ids);
        defer ids_buffer.deinit();
        var history_buffer = try zml.Buffer.fromSlice(io, self.platform, history);
        defer history_buffer.deinit();
        var candidate_buffer = try zml.Buffer.fromSlice(io, self.platform, candidate);
        defer candidate_buffer.deinit();

        var exe = try self.platform.compile(gpa, io, self.model, .forward_uncached, .{
            zml.Tensor.fromShape(ids.shape),
            zml.Tensor.fromShape(self.voice.shape()),
            zml.Tensor.fromShape(history.shape),
            zml.Tensor.fromShape(candidate.shape),
            zml.Tensor.fromShape(self.start_time.shape()),
            zml.Tensor.fromShape(self.end_time.shape()),
            zml.Tensor.fromShape(self.active_length.shape()),
            zml.Tensor.fromShape(self.frame.shape()),
        }, .{
            .program_name = @typeName(@This()),
        });
        defer exe.deinit();

        var runner = try exe.runner(gpa);
        defer runner.deinit(gpa);

        var result: zml.Bufferized(Model.Output) = undefined;
        runner.run(io, .{
            self.weights,
            ids_buffer,
            self.voice,
            history_buffer,
            candidate_buffer,
            self.start_time,
            self.end_time,
            self.active_length,
            self.frame,
        }, .{&result}, .{ .wait = true }); // TODO: don't wait!

        self.active_length_host += 1;
        self.active_length = result.active_length;
        self.frame = result.frame;
        defer result.eos_score.deinit();
        // defer zml.Buffer.deinitAll(Model.Output, &result);

        const score = try result.eos_score.getValue(f32, io);

        // const velocity = try result.velocity.getValue([32]f32, io);
        // var frame: [32]f32 = undefined;
        // for (&frame, noise, velocity) |*value, initial, update| value.* = initial + update;

        const frame = try self.frame.getValue([32]f32, io);
        self.latents_list.appendSliceAssumeCapacity(&frame);
        self.ids_list.appendAssumeCapacity(6);
        self.rng = rng;

        // Retain the triggering frame and the configured tail before stopping.
        if (self.remaining == null and score > 4.0) self.remaining = self.frames_after_eos;
        if (self.remaining) |remaining| {
            if (remaining == 0) {
                self.done = true;
                self.stopped_by_eos = true;
            } else self.remaining = remaining - 1;
        }
        if (self.frames() >= self.max_frames) self.done = true;

        return .{
            .eos_score = score,
            .frame = self.frame,
        };
    }
};

/// Generates normalized f32 audio latents for one prompt using cached attention.
///
/// Owns model/voice buffers, two compiled executables, runtime K/V storage, prompt
/// IDs, generated history, and RNG state. The caller lends the platform and drives
/// `step` until `done`. The first call prefills the complete prompt; later calls
/// decode only the previous frame at its absolute sequence position.
/// Cache buffers are donated and replaced on every call. A failure after execution
/// begins makes the session unusable because its cache may already have changed.
pub const Text_To_Speech_Cached = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    weights: zml.Bufferized(Model),
    voice: zml.Buffer,
    start_time: zml.Buffer,
    end_time: zml.Buffer,
    prefill: zml.Exe,
    decode: zml.Exe,
    cache: zml.Bufferized(KV_Cache),
    prompt: zml.Buffer,
    prompt_length: usize,
    ids: std.ArrayList(i64),
    latents: std.ArrayList(f32),
    rng: std.Random.DefaultPrng = .init(42),
    max_frames: usize,
    frames_after_eos: usize,
    remaining: ?usize = null,
    done: bool = false,
    stopped_by_eos: bool = false,
    failed: bool = false,

    pub fn init(io: std.Io, gpa: std.mem.Allocator, arena: std.mem.Allocator, platform: *const zml.Platform, text: []const u8) !Text_To_Speech_Cached {
        _ = arena; // autofix

        const tokenizer_path = try util.assetPath(gpa, util.gradium_phonon_31m_en_tokenizer);
        defer gpa.free(tokenizer_path);
        var tokenizer = try Tokenizer.init(gpa, io, tokenizer_path);
        defer tokenizer.deinit();
        const voice_path = try util.assetPath(gpa, util.gradium_phonon_voice);
        defer gpa.free(voice_path);
        const voice = try load_voice(gpa, io, voice_path);
        defer voice.free(gpa);
        const tokens = try tokenizer.encode_prompt(gpa, text, @intCast(voice.shape.dim(.p)));
        defer tokens.free(gpa);

        // Preserve the checkpoint's text-length budget and short-text EOS tail.
        var text_ids: std.ArrayList(u32) = .empty;
        defer text_ids.deinit(gpa);
        for (tokens.constItems(i64)) |id| {
            if (id == 4 or id == 5 or id == 6) continue;
            try text_ids.append(gpa, @intCast(id));
        }
        var decoder = try tokenizer.inner.decoder();
        defer decoder.deinit();
        var decoded = try decoder.decodeAlloc(gpa, text_ids.items);
        defer decoded.deinit(gpa);
        var words = std.mem.tokenizeAny(u8, decoded.items, " \t\r\n\x0b\x0c");
        var word_count: usize = 0;
        while (words.next() != null) word_count += 1;
        const max_frames: usize = @intFromFloat(@ceil((@as(f64, @floatFromInt(text_ids.items.len)) / 3 + 2) * 12.5));

        const model_path = try util.assetPath(gpa, util.gradium_phonon_31m_en);
        defer gpa.free(model_path);
        var registry = try zml.safetensors.TensorRegistry.fromPath(gpa, io, model_path);
        defer registry.deinit();
        var store: zml.io.TensorStore = .fromRegistry(gpa, &registry, &.{});
        defer store.deinit();
        const model: Model = .init(store.view());
        var weights = try model.load(gpa, io, platform, &store);
        errdefer Model.unload(&weights);
        var voice_buffer = try zml.Buffer.fromSlice(io, platform, voice);
        errdefer voice_buffer.deinit();
        const time_shape: zml.Shape = .init(.{ .b = 1, .t = 1 }, .f32);
        var start_time = try zml.Buffer.fromBytes(io, platform, time_shape, std.mem.asBytes(&@as([1]f32, .{0})));
        errdefer start_time.deinit();
        var end_time = try zml.Buffer.fromBytes(io, platform, time_shape, std.mem.asBytes(&@as([1]f32, .{1})));
        errdefer end_time.deinit();

        const prompt_length = tokens.constItems(i64).len;
        const capacity = try std.math.add(usize, prompt_length, max_frames);
        if (capacity > std.math.maxInt(i32)) return error.CacheCapacityExceeded;
        const symbolic_cache = KV_Cache.init(1, @intCast(capacity));
        var cache = try symbolic_cache.initBuffer(gpa, io, platform);
        errdefer KV_Cache.deinitBuffer(&cache);
        var prompt = try zml.Buffer.fromSlice(io, platform, tokens);
        errdefer prompt.deinit();
        const candidate = zml.Tensor.fromShape(.init(.{ .b = 1, .ld = 32 }, .f32));
        var prefill = try platform.compile(gpa, io, model, .forward_cached_prefill, .{
            zml.Tensor.fromShape(tokens.shape),
            zml.Tensor.fromShape(voice.shape),
            candidate,
            zml.Tensor.fromShape(time_shape),
            zml.Tensor.fromShape(time_shape),
            symbolic_cache,
        }, .{
            .program_name = @typeName(@This()) ++ "/prefill",
        });
        errdefer prefill.deinit();
        var decode = try platform.compile(gpa, io, model, .forward_cached_decode, .{
            zml.Tensor.fromShape(.init(.{ .b = 1, .a = 1, .ld = 32 }, .f32)),
            zml.Tensor.fromShape(.scalar(.i32)),
            candidate,
            zml.Tensor.fromShape(time_shape),
            zml.Tensor.fromShape(time_shape),
            symbolic_cache,
        }, .{
            .program_name = @typeName(@This()) ++ "/decode",
        });
        errdefer decode.deinit();

        // Reserve the complete budget so committing a completed frame cannot allocate.
        var ids = try std.ArrayList(i64).initCapacity(gpa, try std.math.add(usize, tokens.constItems(i64).len, max_frames));
        errdefer ids.deinit(gpa);
        ids.appendSliceAssumeCapacity(tokens.constItems(i64));
        var latents = try std.ArrayList(f32).initCapacity(gpa, try std.math.mul(usize, max_frames, 32));
        errdefer latents.deinit(gpa);
        return .{
            .allocator = gpa,
            .io = io,
            .platform = platform,
            .weights = weights,
            .voice = voice_buffer,
            .start_time = start_time,
            .end_time = end_time,
            .prefill = prefill,
            .decode = decode,
            .cache = cache,
            .prompt = prompt,
            .prompt_length = prompt_length,
            .ids = ids,
            .latents = latents,
            .max_frames = max_frames,
            .frames_after_eos = if (word_count <= 4) 3 else 1,
        };
    }

    pub fn deinit(self: *Text_To_Speech_Cached) void {
        self.latents.deinit(self.allocator);
        self.ids.deinit(self.allocator);
        self.decode.deinit();
        self.prefill.deinit();
        KV_Cache.deinitBuffer(&self.cache);
        self.prompt.deinit();
        self.end_time.deinit();
        self.start_time.deinit();
        self.voice.deinit();
        Model.unload(&self.weights);
    }

    pub fn frames(self: *const Text_To_Speech_Cached) usize {
        return self.latents.items.len / 32;
    }

    /// Generates and retains one normalized frame, returning its raw EOS score.
    pub fn step(self: *Text_To_Speech_Cached) !f32 {
        if (self.done) return error.GenerationFinished;
        if (self.failed) return error.InvalidGenerationState;
        const allocator = self.allocator;
        const io = self.io;
        var rng = self.rng;
        const random = rng.random();
        var noise: [32]f32 = undefined;
        for (&noise) |*value| value.* = random.floatNorm(f32) * @sqrt(@as(f32, 0.3));
        var candidate_buffer = try zml.Buffer.fromBytes(io, self.platform, .init(.{ .b = 1, .ld = 32 }, .f32), std.mem.asBytes(&noise));
        defer candidate_buffer.deinit();
        const frame_count = self.frames();
        const exe = if (frame_count == 0) &self.prefill else &self.decode;
        var runner = try exe.runner(allocator);
        defer runner.deinit(allocator);
        var result: zml.Bufferized(Model.Prediction_Output) = undefined;
        // Output destinations replace the donated cache handles in place.
        errdefer self.failed = true;
        if (frame_count == 0) {
            runner.run(io, .{ self.weights, self.prompt, self.voice, candidate_buffer, self.start_time, self.end_time, self.cache }, .{ &result, &self.cache }, .{ .wait = true });
        } else {
            const position = self.prompt_length + frame_count - 1;
            if (position >= self.cache.k.shape().dim(.k)) return error.CacheCapacityExceeded;
            const absolute_position: i32 = @intCast(position);
            var position_buffer = try zml.Buffer.fromBytes(io, self.platform, .scalar(.i32), std.mem.asBytes(&absolute_position));
            defer position_buffer.deinit();
            const frame: [32]f32 = self.latents.items[self.latents.items.len - 32 ..][0..32].*;
            var frame_buffer = try zml.Buffer.fromBytes(io, self.platform, .init(.{ .b = 1, .a = 1, .ld = 32 }, .f32), std.mem.asBytes(&frame));
            defer frame_buffer.deinit();
            runner.run(io, .{ self.weights, frame_buffer, position_buffer, candidate_buffer, self.start_time, self.end_time, self.cache }, .{ &result, &self.cache }, .{ .wait = true });
        }
        defer zml.Buffer.deinitAll(Model.Prediction_Output, &result);
        const velocity = try result.velocity.getValue([32]f32, io);
        const score = try result.eos_score.getValue(f32, io);
        var frame: [32]f32 = undefined;
        for (&frame, noise, velocity) |*value, initial, update| value.* = initial + update;
        self.latents.appendSliceAssumeCapacity(&frame);
        self.ids.appendAssumeCapacity(6);
        self.rng = rng;

        // Retain the triggering frame and the configured tail before stopping.
        if (self.remaining == null and score > 4.0) self.remaining = self.frames_after_eos;
        if (self.remaining) |remaining| {
            if (remaining == 0) {
                self.done = true;
                self.stopped_by_eos = true;
            } else self.remaining = remaining - 1;
        }
        if (self.frames() >= self.max_frames) self.done = true;
        return score;
    }
};

fn load_voice(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !zml.Slice {
    const here = @typeName(@This()) ++ ".load_voice";

    var span = zml.tracer.span_at(@src(), here, .{});
    defer span.end();

    var registry = blk: {
        var load_registry = zml.tracer.span_at(@src(), here ++ "/load_registry", .{});
        defer load_registry.end();

        break :blk try zml.safetensors.TensorRegistry.fromPath(allocator, io, path);
    };
    defer registry.deinit();

    const tensor = registry.tensors.get("speaker_wavs") orelse return error.VoiceNotFound;
    if (tensor.shape.rank() != 3 or tensor.shape.dim(0) != 1 or tensor.shape.dim(1) != 512 or !tensor.shape.dtype().isFloat())
        return error.InvalidVoiceShape;

    const raw = try zml.Slice.alloc(allocator, tensor.shape);
    defer raw.free(allocator);

    {
        var span_read_voice = zml.tracer.span_at(@src(), here ++ "/read_voice", .{});
        defer span_read_voice.end();

        var read_buffer: [8192]u8 = undefined;
        var reader = try tensor.reader(io, &read_buffer, .{});
        defer reader.deinit();
        try reader.interface.readSliceAll(raw.data());
    }

    var span_transpose_voice = zml.tracer.span_at(@src(), here ++ "/transpose_voice", .{});
    defer span_transpose_voice.end();

    const frames: usize = @intCast(tensor.shape.dim(2));
    const features: usize = @intCast(tensor.shape.dim(1));

    // Match Python's speaker_wavs[0].T: [1, features, frames] -> [1, frames, features].
    const voice = try zml.Slice.alloc(allocator, .init(.{ .b = 1, .p = frames, .ad = features }, .f32));
    const src = raw.constItems(f32);
    const dst = voice.items(f32);
    for (0..frames) |frame| {
        for (0..features) |feature| {
            dst[frame * features + feature] = src[feature * frames + frame];
        }
    }

    return voice;
}

test "Text_To_Speech_Uncached" {
    const test_data = @import("main_test_data.zig");

    const gpa = std.testing.allocator;
    const io = std.testing.io;

    var arena_impl: std.heap.ArenaAllocator = .init(gpa);
    defer arena_impl.deinit();
    const arena = arena_impl.allocator();

    const platform = zml.testing.env();

    var tts = try Text_To_Speech_Uncached.init(io, gpa, arena, platform, "Bonjour ZML!");
    defer tts.deinit();
    while (!tts.done) _ = try tts.step();

    const expected = test_data.audio_latents;
    try std.testing.expectEqual(expected.len, tts.frames());
    try std.testing.expect(tts.stopped_by_eos);
    const shape: zml.Shape = .init(.{ .b = 1, .a = expected.len, .ld = 32 }, .f32);
    // Free-running feedback compounds small f32 differences across frames.
    try zml.testing.expectClose(
        io,
        zml.Slice.init(shape, std.mem.asBytes(&expected)),
        zml.Slice.init(shape, std.mem.sliceAsBytes(tts.latents_list.items)),
        .{
            .absolute_tolerance = 1e-2,
            .relative_tolerance = 1e-4,
            .minimum_close_fraction = 1,
        },
    );
    // A per-element ceiling alone would allow the whole sequence to drift.
    const report = try zml.testing.compareFloats(gpa, f32, f32, std.mem.bytesAsSlice(f32, std.mem.asBytes(&expected)), tts.latents_list.items, .{});
    try std.testing.expect(report.mean_absolute_error <= 1e-4);
    try std.testing.expect(report.rmse <= 5e-4);
    // Check the error without formatting the borrowed device buffer in Step_Result.
    const stopped: anyerror!void = if (tts.step()) |_| {} else |err| err;
    try std.testing.expectError(error.GenerationFinished, stopped);
}

test "Text_To_Speech_Cached" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;

    var arena_impl: std.heap.ArenaAllocator = .init(gpa);
    defer arena_impl.deinit();
    const arena = arena_impl.allocator();

    const platform = zml.testing.env();

    // Both sessions contain large model/buffer structs; keep them off the test stack.
    const uncached = try gpa.create(Text_To_Speech_Uncached);
    defer gpa.destroy(uncached);
    uncached.* = try Text_To_Speech_Uncached.init(io, gpa, arena, platform, "Bonjour ZML!");
    defer uncached.deinit();
    while (!uncached.done) _ = try uncached.step();

    const cached = try gpa.create(Text_To_Speech_Cached);
    defer gpa.destroy(cached);
    cached.* = try Text_To_Speech_Cached.init(io, gpa, arena, platform, "Bonjour ZML!");
    defer cached.deinit();
    while (!cached.done) _ = try cached.step();

    try std.testing.expectEqual(uncached.frames(), cached.frames());
    try std.testing.expectEqual(uncached.stopped_by_eos, cached.stopped_by_eos);
    const shape: zml.Shape = .init(.{ .b = 1, .a = uncached.frames(), .ld = 32 }, .f32);
    // Independent feedback amplifies f32 differences between full-prefix and cached execution.
    try zml.testing.expectClose(
        io,
        zml.Slice.init(shape, std.mem.sliceAsBytes(uncached.latents_list.items)),
        zml.Slice.init(shape, std.mem.sliceAsBytes(cached.latents.items)),
        .{
            .absolute_tolerance = 6e-2,
            .relative_tolerance = 1e-4,
            .minimum_close_fraction = 1,
        },
    );
    // Keep a tight aggregate bound despite a few feedback-amplified outliers.
    const report = try zml.testing.compareFloats(gpa, f32, f32, uncached.latents_list.items, cached.latents.items, .{});
    try std.testing.expect(report.mean_absolute_error <= 1e-3);
    try std.testing.expect(report.rmse <= 4e-3);
    try std.testing.expectError(error.GenerationFinished, cached.step());
}
