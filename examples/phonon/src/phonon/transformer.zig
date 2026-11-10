const std = @import("std");
const zml = @import("zml");

test {
    std.testing.refAllDecls(@This());
}

/// Stores the attention history reused between speech-generation steps.
///
/// `k` and `v` are floating-point `[.layer=6, .b=B, .k=C, .h=8, .hd=64]`: one key
/// and value vector per layer, batch row, sequence slot, and attention head.
/// Keys already include their absolute-position RoPE rotation; values do not.
/// C is fixed prompt-plus-generation capacity, rather than the current length.
/// `init` and `initBuffer` allocate f32 history. Attention preserves each cache
/// tensor's dtype and converts selected keys/values to the hidden input's dtype.
///
/// The host allocates finite zero-filled buffers before prefill and carries the
/// updated cache between calls. Attention writes new positions and masks future
/// slots. These tensors are runtime history, separate from checkpoint parameters;
/// the host owns their buffers and manages donation at the prediction boundary.
pub const KV_Cache = struct {
    k: zml.Tensor,
    v: zml.Tensor,

    pub fn init(batch: i64, capacity: i64) KV_Cache {
        std.debug.assert(batch > 0 and capacity > 0);
        const shape = zml.Shape.init(.{ .layer = 6, .b = batch, .k = capacity, .h = 8, .hd = 64 }, .f32);
        return .{ .k = .fromShape(shape), .v = .fromShape(shape) };
    }

    pub fn initBuffer(self: KV_Cache, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform) !zml.Bufferized(KV_Cache) {
        const zeros = try zml.Slice.alloc(allocator, self.k.shape());
        defer zeros.free(allocator);
        @memset(zeros.items(f32), 0);
        var k = try zml.Buffer.fromSlice(io, platform, zeros);
        errdefer k.deinit();
        const v = try zml.Buffer.fromSlice(io, platform, zeros);
        return .{ .k = k, .v = v };
    }

    pub fn deinitBuffer(buffers: *zml.Bufferized(KV_Cache)) void {
        zml.Buffer.deinitAll(KV_Cache, buffers);
    }

    pub fn reuseBuffer(self: KV_Cache, origin: KV_Cache) KV_Cache {
        const k = self.k.reuseBuffer(origin.k); // [6,B,C,8,64] f32 -> same shape/dtype, donating origin K
        const v = self.v.reuseBuffer(origin.v); // [6,B,C,8,64] f32 -> same shape/dtype, donating origin V
        return .{ .k = k, .v = v };
    }
};

/// Returns transformer features together with their updated attention history.
///
/// `hidden` is floating-point `[.b=B, .s=Q, .d=512]`, one feature vector per newly
/// processed position. Q is the prompt length during prefill and 1 during decode.
/// Decoder layers return residual features; the model applies final normalization
/// before returning this type. `cache` retains the fixed-capacity K/V tensors
/// for all six layers, including the positions processed by this call. Attention
/// preserves the hidden input's dtype and each cache tensor's dtype. The model
/// uses the voice input's dtype; the generation wrapper currently supplies f32.
pub const CachedHidden = struct {
    hidden: zml.Tensor,
    cache: KV_Cache,
};

// --- Gradium_Phonon_TTS_Model ---
// The complete transformer, which runs multiple passes of the inner transformer block (aka Gradium_Phonon_TTS_Decoder_Layer).

/// Turns text, voice, and generated audio into contextual transformer features.
///
/// Token embeddings are replaced at special positions by projected voice frames,
/// a learned audio-start vector, or completed audio latents. Six decoder layers
/// mix context using causal attention, then final LayerNorm produces the features
/// used by the EOS and flow heads. Each position can see itself and earlier
/// positions in the same unpadded sequence.
///
/// `forward_uncached` processes the complete prefix and audio history on every call.
/// `forward_cached` processes only new positions and returns updated K/V history.
/// Both paths perform embedding lookup, input replacement, all six decoder layers,
/// and final normalization. The caller supplies matching token IDs, voice frames,
/// and audio latents for the positions being processed.
///
/// Inputs include integer IDs `[B, S]`, floating-point voice `[B, P, 512]`, and
/// normalized audio history `[B, A, 32]`. Voice determines the compute/output dtype;
/// audio history and parameters are converted to match. Output features are
/// `[B, S, 512]` for full-prefix execution or `[B, Q, 512]` for Q new cached positions.
/// Each K/V cache tensor preserves its storage dtype.
/// B counts sequences, P voice frames, and A completed generated frames.
///
/// Parameters come from the root checkpoint view: transcript embeddings, input
/// projections, `flow_lm.transformer.layers.0..5`, and `flow_lm.out_norm`.
/// Stored bf16 tensors are converted to the voice dtype. ZML evaluates normalization
/// statistics, RoPE trigonometry, softmax, and erf internally in f32, then converts
/// back. This port targets phonon-31m-en with 8 attention heads of 64 features,
/// RoPE theta 10000, and norm epsilon 1e-5.
///
/// Reference:
/// ```
/// model: GradiumPhononTTSModel
/// ├── embed_tokens: Embedding
/// │   └── weight: shape=(4001, 512), dtype=torch.float32
/// │       ├── loaded: model.embed_tokens.weight
/// │       └── checkpoint: flow_lm.condition_provider.conditioners.transcript_in_segment.embed.weight
/// ├── layers: ModuleList
/// │   └── 0..5: GradiumPhononTTSDecoderLayer
/// │       ├── self_attn: GradiumPhononTTSAttention
/// │       │   ├── q_proj: Linear
/// │       │   │   └── weight: shape=(512, 512), dtype=torch.float32
/// │       │   │       ├── loaded: model.layers.5.self_attn.q_proj.weight
/// │       │   │       └── checkpoint: flow_lm.transformer.layers.5.self_attn.in_proj.weight
/// │       │   ├── k_proj: Linear
/// │       │   │   └── weight: shape=(512, 512), dtype=torch.float32
/// │       │   │       ├── loaded: model.layers.5.self_attn.k_proj.weight
/// │       │   │       └── checkpoint: flow_lm.transformer.layers.5.self_attn.in_proj.weight
/// │       │   ├── v_proj: Linear
/// │       │   │   └── weight: shape=(512, 512), dtype=torch.float32
/// │       │   │       ├── loaded: model.layers.5.self_attn.v_proj.weight
/// │       │   │       └── checkpoint: flow_lm.transformer.layers.5.self_attn.in_proj.weight
/// │       │   └── o_proj: Linear
/// │       │       └── weight: shape=(512, 512), dtype=torch.float32
/// │       │           ├── loaded: model.layers.5.self_attn.o_proj.weight
/// │       │           └── checkpoint: flow_lm.transformer.layers.5.self_attn.out_proj.weight
/// │       ├── mlp: GradiumPhononTTSMLP
/// │       │   ├── activation_fn: GELUActivation
/// │       │   ├── fc1: Linear
/// │       │   │   └── weight: shape=(2048, 512), dtype=torch.float32
/// │       │   │       ├── loaded: model.layers.5.mlp.fc1.weight
/// │       │   │       └── checkpoint: flow_lm.transformer.layers.5.linear1.weight
/// │       │   └── fc2: Linear
/// │       │       └── weight: shape=(512, 2048), dtype=torch.float32
/// │       │           ├── loaded: model.layers.5.mlp.fc2.weight
/// │       │           └── checkpoint: flow_lm.transformer.layers.5.linear2.weight
/// │       ├── input_layernorm: LayerNorm
/// │       │   ├── weight: shape=(512,), dtype=torch.float32
/// │       │   │   ├── loaded: model.layers.5.input_layernorm.weight
/// │       │   │   └── checkpoint: flow_lm.transformer.layers.5.norm1.weight
/// │       │   └── bias: shape=(512,), dtype=torch.float32
/// │       │       ├── loaded: model.layers.5.input_layernorm.bias
/// │       │       └── checkpoint: flow_lm.transformer.layers.5.norm1.bias
/// │       └── post_attention_layernorm: LayerNorm
/// │           ├── weight: shape=(512,), dtype=torch.float32
/// │           │   ├── loaded: model.layers.5.post_attention_layernorm.weight
/// │           │   └── checkpoint: flow_lm.transformer.layers.5.norm2.weight
/// │           └── bias: shape=(512,), dtype=torch.float32
/// │               ├── loaded: model.layers.5.post_attention_layernorm.bias
/// │               └── checkpoint: flow_lm.transformer.layers.5.norm2.bias
/// ├── norm: LayerNorm
/// │   ├── weight: shape=(512,), dtype=torch.float32
/// │   │   ├── loaded: model.norm.weight
/// │   │   └── checkpoint: flow_lm.out_norm.weight
/// │   └── bias: shape=(512,), dtype=torch.float32
/// │       ├── loaded: model.norm.bias
/// │       └── checkpoint: flow_lm.out_norm.bias
/// ├── rotary_emb: GradiumPhononTTSRotaryEmbedding
/// ├── audio_prompt_proj: Linear
/// │   └── weight: shape=(512, 512), dtype=torch.float32
/// │       ├── loaded: model.audio_prompt_proj.weight
/// │       └── checkpoint: flow_lm.condition_provider.conditioners.speaker_wavs.output_proj.weight
/// ├── audio_latent_proj: Linear
/// │   └── weight: shape=(512, 32), dtype=torch.float32
/// │       ├── loaded: model.audio_latent_proj.weight
/// │       └── checkpoint: flow_lm.input_linear.weight
/// ├── num_speakers_embed: Embedding
/// │   └── weight: shape=(32, 16), dtype=torch.float32
/// │       ├── loaded: model.num_speakers_embed.weight
/// │       └── checkpoint: flow_lm.condition_provider.conditioners.num_speakers.embed.weight
/// ├── num_speakers_proj: Linear
/// │   └── weight: shape=(512, 16), dtype=torch.float32
/// │       ├── loaded: model.num_speakers_proj.weight
/// │       └── checkpoint: flow_lm.condition_provider.conditioners.num_speakers.output_proj.weight
/// ├── duration_delta_proj: Linear
/// │   └── weight: shape=(512, 128), dtype=torch.float32
/// │       ├── loaded: model.duration_delta_proj.weight
/// │       └── checkpoint: flow_lm.condition_provider.conditioners.duration_delta.output_proj.weight
/// └── bos_latent: shape=(32,), dtype=torch.float32
///     ├── loaded: model.bos_latent
///     └── checkpoint: flow_lm.bos_emb
/// ```
pub const Gradium_Phonon_TTS_Model = struct {
    embed_tokens: Embedding,
    input_projection: Input_Projection,
    layers: [num_layers]Gradium_Phonon_TTS_Decoder_Layer,
    norm: zml.nn.LayerNorm,

    // FIX: This port currently targets the six-layer phonon-31m-en checkpoint (here and everywhere else).
    const num_layers = 6;

    pub fn init(store: zml.io.TensorStore.View) Gradium_Phonon_TTS_Model {
        const here = @typeName(@This()) ++ ".init";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        var self: Gradium_Phonon_TTS_Model = .{
            .embed_tokens = .init(store.withPrefix("flow_lm.condition_provider.conditioners.transcript_in_segment.embed")),
            .input_projection = .init(store),
            .layers = undefined,
            .norm = .{
                .weight = store.createReplicatedTensor("flow_lm.out_norm.weight", .{.d}),
                .bias = store.createReplicatedTensor("flow_lm.out_norm.bias", .{.d}),
                .eps = 1e-5,
            },
        };
        for (&self.layers, 0..) |*layer, i| {
            layer.* = .init(store.withPrefix("flow_lm.transformer.layers").withLayer(i));
        }
        return self;
    }

    /// Runs input preparation, all six decoder layers, and final normalization.
    ///
    /// `token_ids`: integer `[B, S]`, normally i64, the complete unpadded sequence
    /// from the tokenizer and generation loop. Contains text and control IDs;
    /// each completed audio frame adds one ID 6. S must be nonzero.
    /// `voice`: floating-point `[B, P, 512]`, ordered speaker features loaded from the voice
    /// asset and transposed by the caller from `[B, 512, P]`. Each row has P ID-5 markers.
    /// `audio_latents`: floating-point `[B, A, 32]`, the generation loop's complete normalized
    /// frame history, ordered like the A ID-6 markers. Initially A=0.
    /// All inputs share batch size B; positions start at zero on every call.
    /// Computation uses `voice.dtype()`; embeddings, parameters, and audio history
    /// are converted to that dtype.
    ///
    /// Returns `[B, S, 512]` in `voice.dtype()`, final normalized contextual features
    /// for every position. The prediction wrapper selects the last position for both heads.
    ///
    /// Dimension tags:
    /// - `.b`: batch; independent sequences, shared by token IDs and voice frames.
    /// - `.s`: sequence; token positions, preserved in the output.
    /// - `.p`: prompt frames; ordered voice frames inserted at audio-prompt markers.
    /// - `.ad`: audio-prompt features; 512 input features per voice frame.
    /// - `.a`: completed generated frames, in marker order; zero initially.
    /// - `.ld`: 32 normalized latent features per completed frame.
    /// - `.d`: model dimension; 512 output features per token, normalized by the final norm.
    /// Internal projection/attention tags are documented in their forward methods.
    ///
    /// Inputs: integer IDs `[.b, .s]`, floating-point voice `[.b, .p, .ad=512]`,
    /// and floating-point complete generated history `[.b, .a, .ld=32]`.
    /// Output: `[.b, .s, .d=512]` in `voice.dtype()`.
    pub fn forward_uncached(
        self: Gradium_Phonon_TTS_Model,
        token_ids: zml.Tensor,
        voice: zml.Tensor,
        audio_latents: zml.Tensor,
    ) zml.Tensor {
        const here = @typeName(@This()) ++ ".forward_uncached";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        const dtype = voice.dtype();
        std.debug.assert(dtype.isFloat() and audio_latents.dtype().isFloat());

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), "Gradium_Phonon_TTS_Model", .{});
        defer compiler.popLocation();

        var hidden = self.prepare_input(token_ids, voice, audio_latents);
        for (self.layers, 0..) |layer, index| {
            compiler.pushLocationFmt(@src(), "layer[{d}]", .{index});
            defer compiler.popLocation();
            hidden = layer.forward(hidden);
        }

        return self.normalize_output(hidden);
    }

    /// Processes a new chunk against the stored prefix, then applies final LayerNorm.
    ///
    /// `token_ids`: integer [B,chunk_length], IDs for only this call's positions.
    /// Prefill supplies the whole prompt; decode supplies one audio marker (ID 6).
    /// `voice`: floating [B,P,512], ordered speaker frames for the new ID-5 markers;
    /// P=0 during decode. Voice determines compute/output dtype even when empty.
    /// `audio_latents`: floating [B,A,32], normalized frames for the new ID-6 markers;
    /// A=0 during prefill and A=1 during decode. All rows have matching marker counts.
    /// `position`: scalar i32 first absolute slot of this chunk, zero at prefill.
    /// Caller ensures position >= 0 and position+chunk_length <= capacity.
    /// `cache`: K/V history, each floating [6,B,capacity,8,64], with all preceding
    /// slots populated and unused slots finite. Keys include RoPE; values do not.
    /// Each cache retains its storage dtype; parameters/inputs convert to voice dtype.
    ///
    /// Returns normalized [B,chunk_length,512] features and updated history for all
    /// six layers. Each input row contains only real positions, without padding.
    /// Tags: .b batch, .s new positions, .d=512 features, .p voice frames,
    /// .ad=512 voice features, .a generated frames, .ld=32 latent coordinates;
    /// cache .layer=6, .k=capacity, .h=8 heads, .hd=64 coordinates per head.
    pub fn forward_cached(
        self: Gradium_Phonon_TTS_Model,
        token_ids: zml.Tensor,
        voice: zml.Tensor,
        audio_latents: zml.Tensor,
        position: zml.Tensor,
        cache: KV_Cache,
    ) CachedHidden {
        const here = @typeName(@This()) ++ ".forward_cached";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        const dtype = voice.dtype();
        std.debug.assert(dtype.isFloat() and audio_latents.dtype().isFloat());

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), "Gradium_Phonon_TTS_Model", .{});
        defer compiler.popLocation();

        var hidden = self.prepare_input(token_ids, voice, audio_latents);
        var updated = cache;
        for (self.layers, 0..) |layer, index| {
            compiler.pushLocationFmt(@src(), "layer[{d}]", .{index});
            defer compiler.popLocation();

            const result = layer.forward_cached(hidden, position, updated, index);
            hidden = result.hidden;
            updated = result.cache;
        }
        const output = self.normalize_output(hidden);
        return .{ .hidden = output, .cache = updated };
    }

    /// Token IDs [B,S], voice [B,P,512], and latents [B,A,32] -> [B,S,512].
    /// Voice determines compute dtype, including when P=0 during cached decode.
    fn prepare_input(self: Gradium_Phonon_TTS_Model, token_ids: zml.Tensor, voice: zml.Tensor, audio_latents: zml.Tensor) zml.Tensor {
        const embeddings = self.embed_tokens.forward(token_ids).convert(voice.dtype());
        return self.input_projection.forward(embeddings, token_ids, voice, audio_latents);
    }

    /// Apply the learned final LayerNorm to [..., .d=512], preserving dtype.
    fn normalize_output(self: Gradium_Phonon_TTS_Model, hidden: zml.Tensor) zml.Tensor {
        const norm: zml.nn.LayerNorm = .{
            .weight = self.norm.weight.convert(hidden.dtype()),
            .bias = self.norm.bias.?.convert(hidden.dtype()),
            .eps = self.norm.eps,
        };
        return norm.forward(hidden);
    }

    pub fn load(
        self: *const Gradium_Phonon_TTS_Model,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
    ) !zml.Bufferized(Gradium_Phonon_TTS_Model) {
        const here = @typeName(@This()) ++ ".load";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        // bufferize does not traverse arrays, so load each layer separately.
        var buffers: zml.Bufferized(Gradium_Phonon_TTS_Model) = undefined;
        buffers.embed_tokens = try self.embed_tokens.load(allocator, io, platform, store);
        errdefer Embedding.unloadBuffers(&buffers.embed_tokens);
        buffers.input_projection = try self.input_projection.load(allocator, io, platform, store);
        errdefer Input_Projection.unloadBuffers(&buffers.input_projection);

        buffers.norm = try zml.mem.bufferize(allocator, zml.nn.LayerNorm, &self.norm);
        errdefer zml.Buffer.deinitAll(zml.nn.LayerNorm, &buffers.norm);
        // FIX: having this many loaders all over the place is clearly a mistake.
        // -> similarly, are default settings reallt good enough?
        var loader: zml.io.Loader = try .init(allocator, platform, .default);
        defer loader.deinit();
        try loader.load(io, zml.nn.LayerNorm, &self.norm, &buffers.norm, store, .{});
        try loader.await(io);

        var loaded: usize = 0;
        errdefer for (buffers.layers[0..loaded]) |*layer| {
            Gradium_Phonon_TTS_Decoder_Layer.unloadBuffers(layer);
        };

        for (&self.layers, &buffers.layers) |*layer, *layer_buffers| {
            layer_buffers.* = try layer.load(allocator, io, platform, store);
            loaded += 1;
        }

        return buffers;
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Gradium_Phonon_TTS_Model)) void {
        const here = @typeName(@This()) ++ ".unloadBuffers";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        Embedding.unloadBuffers(&self.embed_tokens);
        Input_Projection.unloadBuffers(&self.input_projection);
        zml.Buffer.deinitAll(zml.nn.LayerNorm, &self.norm);
        for (&self.layers) |*layer| {
            Gradium_Phonon_TTS_Decoder_Layer.unloadBuffers(layer);
        }
    }
};

test Gradium_Phonon_TTS_Model {
    const test_data = @import("transformer_test_data.zig");

    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    var rng: std.Random.DefaultPrng = .init(42);
    const random = rng.random();

    var model: Gradium_Phonon_TTS_Model = .{
        .embed_tokens = .{ .weight = .fromShape(.init(.{ .voc = 4001, .d = 512 }, .bf16)) },
        .input_projection = .{
            .audio_prompt_proj = .init(.fromShape(.init(.{ .d = 512, .ad = 512 }, .bf16)), null, .ad),
            .audio_latent_proj = .init(.fromShape(.init(.{ .d = 512, .ld = 32 }, .bf16)), null, .ld),
            .bos_latent = .fromShape(.init(.{ .ld = 32 }, .bf16)),
            .num_speakers_embed = .fromShape(.init(.{ .voc = 32, .spk = 16 }, .bf16)),
            .num_speakers_proj = .init(.fromShape(.init(.{ .d = 512, .spk = 16 }, .bf16)), null, .spk),
            .duration_delta_proj = .init(.fromShape(.init(.{ .d = 512, .dur = 128 }, .bf16)), null, .dur),
            .num_speakers = random.intRangeLessThan(i64, 0, 32),
            .duration_delta = random.float(f32),
        },
        .layers = undefined,
        .norm = .{
            .weight = .fromShape(.init(.{ .d = 512 }, .bf16)),
            .bias = .fromShape(.init(.{ .d = 512 }, .bf16)),
            .eps = 1e-5,
        },
    };
    for (&model.layers) |*layer| {
        layer.* = .{
            .input_layernorm = .{
                .weight = .fromShape(.init(.{ .d = 512 }, .bf16)),
                .bias = .fromShape(.init(.{ .d = 512 }, .bf16)),
                .eps = 1e-5,
            },
            .self_attn = .{
                .qkv_proj = .init(.fromShape(.init(.{ .qkv = 1536, .d = 512 }, .bf16)), null, .d),
                .out_proj = .init(.fromShape(.init(.{ .dout = 512, .d = 512 }, .bf16)), null, .d),
            },
            .post_attention_layernorm = .{
                .weight = .fromShape(.init(.{ .d = 512 }, .bf16)),
                .bias = .fromShape(.init(.{ .d = 512 }, .bf16)),
                .eps = 1e-5,
            },
            .mlp = .{
                .fc1 = .init(.fromShape(.init(.{ .dout = 2048, .d = 512 }, .bf16)), null, .d),
                .fc2 = .init(.fromShape(.init(.{ .d = 512, .dout = 2048 }, .bf16)), null, .dout),
            },
        };
    }

    // Each batch row has two voice markers, two text tokens, BOS, then two audio markers.
    const ids_shape: zml.Shape = .init(.{ .b = 2, .s = 7 }, .i64);
    var ids: [2][7]i64 = .{ .{ 5, 7, 5, 7, 4, 6, 6 }, .{ 7, 5, 7, 5, 4, 6, 6 } };
    for (&ids) |*batch| {
        for (batch) |*id| {
            if (id.* == 7) id.* = random.intRangeLessThan(i64, 7, 4001);
        }
    }
    var ids_buffer = try zml.Buffer.fromBytes(io, platform, ids_shape, std.mem.asBytes(&ids));
    defer ids_buffer.deinit();

    const voice_shape: zml.Shape = .init(.{ .b = 2, .p = 2, .ad = 512 }, .f32);
    var voice: [2][2][512]f32 = undefined;
    for (&voice) |*batch| {
        for (batch) |*frame| {
            for (frame) |*value| value.* = random.float(f32) * 2 - 1;
        }
    }
    var voice_buffer = try zml.Buffer.fromBytes(io, platform, voice_shape, std.mem.asBytes(&voice));
    defer voice_buffer.deinit();

    const audio_shape: zml.Shape = .init(.{ .b = 2, .a = 2, .ld = 32 }, .f32);
    var audio: [2][2][32]f32 = undefined;
    for (&audio) |*batch| {
        for (batch) |*frame| {
            for (frame) |*value| value.* = random.float(f32) * 2 - 1;
        }
    }
    var audio_buffer = try zml.Buffer.fromBytes(io, platform, audio_shape, std.mem.asBytes(&audio));
    defer audio_buffer.deinit();

    // Bufferize array elements explicitly, as in the production loader.
    var weight_buffers: zml.Bufferized(Gradium_Phonon_TTS_Model) = .{
        .embed_tokens = try zml.mem.bufferize(allocator, Embedding, &model.embed_tokens),
        .input_projection = try zml.mem.bufferize(allocator, Input_Projection, &model.input_projection),
        .norm = try zml.mem.bufferize(allocator, zml.nn.LayerNorm, &model.norm),
        .layers = undefined,
    };
    for (&model.layers, &weight_buffers.layers) |*layer, *buffers| {
        buffers.* = try zml.mem.bufferize(allocator, Gradium_Phonon_TTS_Decoder_Layer, layer);
    }
    const tensors = try zml.meta.collectValues(zml.Tensor, allocator, &model);
    defer allocator.free(tensors);
    const buffers = try zml.meta.collectMutPtrs(zml.Buffer, allocator, &weight_buffers);
    defer allocator.free(buffers);
    var loaded: usize = 0;
    defer for (buffers[0..loaded]) |buffer| buffer.deinit();
    for (tensors, buffers) |tensor, buffer| {
        const values = try zml.Slice.alloc(allocator, tensor.shape());
        defer values.free(allocator);
        for (values.items(zml.floats.BFloat16)) |*value| value.* = .fromF32(random.float(f32) * 2 - 1);
        buffer.* = try zml.Buffer.fromSlice(io, platform, values);
        loaded += 1;
    }

    const output_shape: zml.Shape = .init(.{ .b = 2, .s = 7, .d = 512 }, .f32);
    var exe = try platform.compile(allocator, io, model, .forward_uncached, .{
        zml.Tensor.fromShape(ids_shape),
        zml.Tensor.fromShape(voice_shape),
        zml.Tensor.fromShape(audio_shape),
    }, .{});
    defer exe.deinit();
    var output = try exe.eval(allocator, io, .{ weight_buffers, ids_buffer, voice_buffer, audio_buffer });
    defer output.deinit();

    const expected = test_data.transformer_model;
    const expected_slice = zml.Slice.init(output_shape, std.mem.asBytes(&expected));
    try zml.testing.expectClose(io, expected_slice, output, .{
        .absolute_tolerance = 1e-4,
        .relative_tolerance = 1e-4,
        .minimum_close_fraction = 1,
    });

    // Audio history may use a different float dtype from the voice features.
    for ([_]zml.DataType{ .f16, .bf16, .f32, .f64 }) |dtype| {
        const audio_dtype: zml.DataType = if (dtype == .f32) .f64 else .f32;
        var typed_exe = try platform.compile(allocator, io, model, .forward_uncached, .{
            zml.Tensor.fromShape(ids_shape),
            zml.Tensor.fromShape(voice_shape.withDtype(dtype)),
            zml.Tensor.fromShape(audio_shape.withDtype(audio_dtype)),
        }, .{});
        defer typed_exe.deinit();
        try std.testing.expect(typed_exe.output_shapes[0].eqlWithTags(output_shape.withDtype(dtype)));
    }
}

test "Gradium_Phonon_TTS_Model: cached" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    var rng: std.Random.DefaultPrng = .init(42);
    const random = rng.random();

    var model: Gradium_Phonon_TTS_Model = .{
        .embed_tokens = .{ .weight = .fromShape(.init(.{ .voc = 4001, .d = 512 }, .bf16)) },
        .input_projection = .{
            .audio_prompt_proj = .init(.fromShape(.init(.{ .d = 512, .ad = 512 }, .bf16)), null, .ad),
            .audio_latent_proj = .init(.fromShape(.init(.{ .d = 512, .ld = 32 }, .bf16)), null, .ld),
            .bos_latent = .fromShape(.init(.{ .ld = 32 }, .bf16)),
            .num_speakers_embed = .fromShape(.init(.{ .voc = 32, .spk = 16 }, .bf16)),
            .num_speakers_proj = .init(.fromShape(.init(.{ .d = 512, .spk = 16 }, .bf16)), null, .spk),
            .duration_delta_proj = .init(.fromShape(.init(.{ .d = 512, .dur = 128 }, .bf16)), null, .dur),
            .num_speakers = random.intRangeLessThan(i64, 0, 32),
            .duration_delta = random.float(f32),
        },
        .layers = undefined,
        .norm = .{
            .weight = .fromShape(.init(.{ .d = 512 }, .bf16)),
            .bias = .fromShape(.init(.{ .d = 512 }, .bf16)),
            .eps = 1e-5,
        },
    };
    for (&model.layers) |*layer| {
        layer.* = .{
            .input_layernorm = .{
                .weight = .fromShape(.init(.{ .d = 512 }, .bf16)),
                .bias = .fromShape(.init(.{ .d = 512 }, .bf16)),
                .eps = 1e-5,
            },
            .self_attn = .{
                .qkv_proj = .init(.fromShape(.init(.{ .qkv = 1536, .d = 512 }, .bf16)), null, .d),
                .out_proj = .init(.fromShape(.init(.{ .dout = 512, .d = 512 }, .bf16)), null, .d),
            },
            .post_attention_layernorm = .{
                .weight = .fromShape(.init(.{ .d = 512 }, .bf16)),
                .bias = .fromShape(.init(.{ .d = 512 }, .bf16)),
                .eps = 1e-5,
            },
            .mlp = .{
                .fc1 = .init(.fromShape(.init(.{ .dout = 2048, .d = 512 }, .bf16)), null, .d),
                .fc2 = .init(.fromShape(.init(.{ .d = 512, .dout = 2048 }, .bf16)), null, .dout),
            },
        };
    }

    // Each batch row has two voice markers, two text tokens, BOS, then two audio markers.
    const ids_shape: zml.Shape = .init(.{ .b = 2, .s = 7 }, .i64);
    var ids: [2][7]i64 = .{ .{ 5, 7, 5, 7, 4, 6, 6 }, .{ 7, 5, 7, 5, 4, 6, 6 } };
    for (&ids) |*batch| {
        for (batch) |*id| {
            if (id.* == 7) id.* = random.intRangeLessThan(i64, 7, 4001);
        }
    }
    var ids_buffer = try zml.Buffer.fromBytes(io, platform, ids_shape, std.mem.asBytes(&ids));
    defer ids_buffer.deinit();

    const voice_shape: zml.Shape = .init(.{ .b = 2, .p = 2, .ad = 512 }, .f32);
    var voice: [2][2][512]f32 = undefined;
    for (&voice) |*batch| {
        for (batch) |*frame| {
            for (frame) |*value| value.* = random.float(f32) * 2 - 1;
        }
    }
    var voice_buffer = try zml.Buffer.fromBytes(io, platform, voice_shape, std.mem.asBytes(&voice));
    defer voice_buffer.deinit();

    const audio_shape: zml.Shape = .init(.{ .b = 2, .a = 2, .ld = 32 }, .f32);
    var audio: [2][2][32]f32 = undefined;
    for (&audio) |*batch| {
        for (batch) |*frame| {
            for (frame) |*value| value.* = random.float(f32) * 2 - 1;
        }
    }
    var audio_buffer = try zml.Buffer.fromBytes(io, platform, audio_shape, std.mem.asBytes(&audio));
    defer audio_buffer.deinit();

    // Bufferize array elements explicitly, as in the production loader.
    var weight_buffers: zml.Bufferized(Gradium_Phonon_TTS_Model) = .{
        .embed_tokens = try zml.mem.bufferize(allocator, Embedding, &model.embed_tokens),
        .input_projection = try zml.mem.bufferize(allocator, Input_Projection, &model.input_projection),
        .norm = try zml.mem.bufferize(allocator, zml.nn.LayerNorm, &model.norm),
        .layers = undefined,
    };
    for (&model.layers, &weight_buffers.layers) |*layer, *buffers| {
        buffers.* = try zml.mem.bufferize(allocator, Gradium_Phonon_TTS_Decoder_Layer, layer);
    }
    const tensors = try zml.meta.collectValues(zml.Tensor, allocator, &model);
    defer allocator.free(tensors);
    const buffers = try zml.meta.collectMutPtrs(zml.Buffer, allocator, &weight_buffers);
    defer allocator.free(buffers);
    var loaded: usize = 0;
    defer for (buffers[0..loaded]) |buffer| buffer.deinit();
    for (tensors, buffers) |tensor, buffer| {
        const values = try zml.Slice.alloc(allocator, tensor.shape());
        defer values.free(allocator);
        for (values.items(zml.floats.BFloat16)) |*value| value.* = .fromF32(random.float(f32) * 2 - 1);
        buffer.* = try zml.Buffer.fromSlice(io, platform, values);
        loaded += 1;
    }

    const output_shape: zml.Shape = .init(.{ .b = 2, .s = 7, .d = 512 }, .f32);
    var uncached_exe = try platform.compile(allocator, io, model, .forward_uncached, .{
        zml.Tensor.fromShape(ids_shape),
        zml.Tensor.fromShape(voice_shape),
        zml.Tensor.fromShape(audio_shape),
    }, .{});
    defer uncached_exe.deinit();
    var uncached_output = try uncached_exe.eval(allocator, io, .{ weight_buffers, ids_buffer, voice_buffer, audio_buffer });
    defer uncached_output.deinit();

    // Prefill the prompt, then feed each generated frame through the returned cache.
    // Extra finite slots exercise masking beyond the positions written so far.
    const initial_cache = KV_Cache.init(2, 9);
    var cache_buffers = try initial_cache.initBuffer(allocator, io, platform);
    defer KV_Cache.deinitBuffer(&cache_buffers);
    var cached_output: [2][7][512]f32 = undefined;
    inline for (.{ .{ .start = 0, .end = 5 }, .{ .start = 5, .end = 6 }, .{ .start = 6, .end = 7 } }) |step| {
        const count = step.end - step.start;
        const chunk_ids_shape: zml.Shape = .init(.{ .b = 2, .s = count }, .i64);
        var chunk_ids: [2][count]i64 = undefined;
        for (&chunk_ids, ids) |*batch, source| {
            @memcpy(batch, source[step.start..step.end]);
        }
        var chunk_ids_buffer = try zml.Buffer.fromBytes(io, platform, chunk_ids_shape, std.mem.asBytes(&chunk_ids));
        defer chunk_ids_buffer.deinit();

        const prompt_frames = if (step.start == 0) 2 else 0;
        const chunk_voice_shape: zml.Shape = .init(.{ .b = 2, .p = prompt_frames, .ad = 512 }, .f32);
        var chunk_voice: [2][prompt_frames][512]f32 = undefined;
        for (&chunk_voice, voice) |*batch, source| {
            @memcpy(batch, source[0..prompt_frames]);
        }
        var chunk_voice_buffer = try zml.Buffer.fromBytes(io, platform, chunk_voice_shape, std.mem.asBytes(&chunk_voice));
        defer chunk_voice_buffer.deinit();

        const audio_frames = if (step.start == 0) 0 else 1;
        const chunk_audio_shape: zml.Shape = .init(.{ .b = 2, .a = audio_frames, .ld = 32 }, .f32);
        var chunk_audio: [2][audio_frames][32]f32 = undefined;
        if (audio_frames > 0) {
            for (&chunk_audio, audio) |*batch, source| {
                batch[0] = source[step.start - 5];
            }
        }
        var chunk_audio_buffer = try zml.Buffer.fromBytes(io, platform, chunk_audio_shape, std.mem.asBytes(&chunk_audio));
        defer chunk_audio_buffer.deinit();

        const position: i32 = step.start;
        var position_buffer = try zml.Buffer.fromBytes(io, platform, .scalar(.i32), std.mem.asBytes(&position));
        defer position_buffer.deinit();
        var cached_exe = try platform.compile(allocator, io, model, .forward_cached, .{
            zml.Tensor.fromShape(chunk_ids_shape),
            zml.Tensor.fromShape(chunk_voice_shape),
            zml.Tensor.fromShape(chunk_audio_shape),
            zml.Tensor.fromShape(.scalar(.i32)),
            initial_cache,
        }, .{});
        defer cached_exe.deinit();
        var runner = try cached_exe.runner(allocator);
        defer runner.deinit(allocator);
        var result: zml.Bufferized(CachedHidden) = undefined;
        runner.run(io, .{ weight_buffers, chunk_ids_buffer, chunk_voice_buffer, chunk_audio_buffer, position_buffer, cache_buffers }, .{&result}, .{ .wait = true });
        defer result.hidden.deinit();
        KV_Cache.deinitBuffer(&cache_buffers);
        cache_buffers = result.cache;

        const values = try result.hidden.getValue([2][count][512]f32, io);
        for (&cached_output, values) |*batch, source| {
            @memcpy(batch[step.start..step.end], &source);
        }
    }

    const cached_slice = zml.Slice.init(output_shape, std.mem.asBytes(&cached_output));
    // Cached and full-sequence execution use different matmul/reduction shapes.
    try zml.testing.expectClose(io, uncached_output, cached_slice, .{
        .absolute_tolerance = 1e-3,
        .relative_tolerance = 1e-4,
        .minimum_close_fraction = 1,
    });

    for ([_]zml.DataType{ .f16, .bf16, .f32, .f64 }) |dtype| {
        inline for (.{ true, false }) |prefill| {
            const count = if (prefill) 5 else 1;
            const cache_shape: zml.Shape = .init(.{ .layer = 6, .b = 2, .k = 9, .h = 8, .hd = 64 }, dtype);
            const cache: KV_Cache = .{
                .k = .fromShape(cache_shape.withDtype(if (prefill) dtype else .f16)),
                .v = .fromShape(cache_shape.withDtype(if (prefill) dtype else .bf16)),
            };
            var typed_exe = try platform.compile(allocator, io, model, .forward_cached, .{
                zml.Tensor.fromShape(.init(.{ .b = 2, .s = count }, .i64)),
                zml.Tensor.fromShape(.init(.{ .b = 2, .p = if (prefill) 2 else 0, .ad = 512 }, dtype)),
                zml.Tensor.fromShape(.init(.{ .b = 2, .a = if (prefill) 0 else 1, .ld = 32 }, .f32)),
                zml.Tensor.fromShape(.scalar(.i32)),
                cache,
            }, .{});
            defer typed_exe.deinit();
            try std.testing.expect(typed_exe.output_shapes[0].eqlWithTags(.init(.{ .b = 2, .s = count, .d = 512 }, dtype)));
            try std.testing.expect(typed_exe.output_shapes[1].eqlWithTags(cache.k.shape()));
            try std.testing.expect(typed_exe.output_shapes[2].eqlWithTags(cache.v.shape()));
        }
    }
}

// --- Embedding ---

/// Looks up the initial feature vector for each text or control token.
///
/// Input: integer IDs `[B, S]` in [0, 4001), normally i64 from the tokenizer
/// and generation loop. Output: `[B, S, 512]` in the table's floating-point dtype,
/// one learned table row per ID. These features have not yet mixed sequence
/// context. The input projection
/// subsequently replaces voice, audio-start, and generated-audio positions.
///
/// The bf16 vocabulary table `[4001, 512]` comes from
/// `flow_lm.condition_provider.conditioners.transcript_in_segment.embed.weight`.
/// Lookup gathers stored rows and preserves the table's dtype.
///
/// Reference:
/// ```
/// embed_tokens: Embedding
/// └── weight: shape=(4001, 512), dtype=torch.float32
///     ├── loaded: model.embed_tokens.weight
///     └── checkpoint: flow_lm.condition_provider.conditioners.transcript_in_segment.embed.weight
/// ```
//
// TODO: one could technically imagine caching all the embedding+voice/audio/BOS replacement stuff
const Embedding = struct {
    weight: zml.Tensor,

    pub fn init(store: zml.io.TensorStore.View) Embedding {
        const here = @typeName(@This()) ++ ".init";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        return .{
            .weight = store.createReplicatedTensor("weight", .{ .voc, .d }),
        };
    }

    // -- Description --

    /// Looks up one learned feature vector per text or control token.
    ///
    /// `input`: integer `[B, S]`, normally i64, IDs from the tokenizer and
    /// generation loop, passed by the model. Every value must be in [0, 4001).
    /// This method assigns tags `.b` and `.s`.
    ///
    /// Returns `[B, S, 512]` in `weight.dtype()`, raw embeddings passed to
    /// `Input_Projection` for voice/audio replacements before transformer context mixing.
    ///
    /// Dimension tags:
    /// - `.b`: batch; independent sequences of token IDs.
    /// - `.s`: sequence; token positions, preserved in the output.
    /// - `.voc`: vocabulary; the 4001 table rows indexed by the input IDs.
    ///   Gathering replaces this axis with the input's `.b` and `.s` axes.
    /// - `.d`: model dimension; the 512 features stored in each selected row.
    ///
    /// Shape: integer IDs `[.b, .s]` -> floating-point embeddings `[.b, .s, .d=512]`.
    pub fn forward(self: Embedding, input: zml.Tensor) zml.Tensor {
        const here = @typeName(@This()) ++ ".forward";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        std.debug.assert(input.dtype().isInteger());
        std.debug.assert(self.weight.dtype().isFloat());
        std.debug.assert(input.rank() == 2);

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), "Embedding", .{});
        defer compiler.popLocation();

        const ids = input.withTags(.{ .b, .s }); // [B,S] integer -> [b=B,s=S] integer
        return self.weight.gather(.{ .voc = ids }, .{}); // [voc=4001,d=512] + [B,S] integer indices -> [B,S,d=512], preserving weight dtype
    }

    // -- Runtime --

    pub fn load(
        self: *const Embedding,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
    ) !zml.Bufferized(Embedding) {
        const here = @typeName(@This()) ++ ".load";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        var buffers = try zml.mem.bufferize(allocator, Embedding, self);
        errdefer unloadBuffers(&buffers);

        var loader: zml.io.Loader = try .init(allocator, platform, .default);
        defer loader.deinit();

        try loader.load(io, Embedding, self, &buffers, store, .{});
        try loader.await(io);

        return buffers;
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Embedding)) void {
        const here = @typeName(@This()) ++ ".unloadBuffers";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        zml.Buffer.deinitAll(Embedding, self);
    }
};

test Embedding {
    const test_data = @import("transformer_test_data.zig");

    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    var rng: std.Random.DefaultPrng = .init(42);
    const random = rng.random();

    const layer: Embedding = .{
        .weight = .fromShape(.init(.{ .voc = 4001, .d = 512 }, .bf16)),
    };
    const input_shape: zml.Shape = .init(.{ .b = 2, .s = 2 }, .i64);
    var input: [2][2]i64 = undefined;
    for (&input) |*batch| {
        for (batch) |*id| id.* = random.intRangeLessThan(i64, 0, 4001);
    }
    const weights = try zml.Slice.alloc(allocator, layer.weight.shape());
    defer weights.free(allocator);
    for (weights.items(zml.floats.BFloat16)) |*value| value.* = .fromF32(random.float(f32) * 2 - 1);

    var weight_buffer = try zml.Buffer.fromSlice(io, platform, weights);
    defer weight_buffer.deinit();
    const weight_buffers: zml.Bufferized(Embedding) = .{ .weight = weight_buffer };
    var input_buffer = try zml.Buffer.fromBytes(io, platform, input_shape, std.mem.asBytes(&input));
    defer input_buffer.deinit();

    var exe = try platform.compile(allocator, io, layer, .forward, .{zml.Tensor.fromShape(input_shape)}, .{});
    defer exe.deinit();
    var output = try exe.eval(allocator, io, .{ weight_buffers, input_buffer });
    defer output.deinit();

    const expected = test_data.embedding;
    const output_shape: zml.Shape = .init(.{ .b = 2, .s = 2, .d = 512 }, .bf16);
    const expected_slice = try zml.Slice.alloc(allocator, output_shape);
    defer expected_slice.free(allocator);
    for (expected_slice.items(zml.floats.BFloat16), std.mem.bytesAsSlice(f32, std.mem.asBytes(&expected))) |*value, recorded| {
        value.* = .fromF32(recorded);
    }
    try zml.testing.expectClose(io, expected_slice, output, .{
        .absolute_tolerance = 1e-4,
        .relative_tolerance = 1e-4,
        .minimum_close_fraction = 1,
    });

    for ([_]zml.DataType{ .f16, .bf16, .f32, .f64 }) |dtype| {
        const typed_layer: Embedding = .{
            .weight = .fromShape(layer.weight.shape().withDtype(dtype)),
        };
        var typed_exe = try platform.compile(allocator, io, typed_layer, .forward, .{
            zml.Tensor.fromShape(input_shape.withDtype(.i32)),
        }, .{});
        defer typed_exe.deinit();
        try std.testing.expect(typed_exe.output_shapes[0].eqlWithTags(output_shape.withDtype(dtype)));
    }
}

// --- Input_Projection ---

/// Builds transformer inputs by inserting voice and audio into token embeddings.
///
/// Text positions retain their embedding. ID 5 receives the corresponding projected
/// voice frame; ID 4 receives the projected learned audio-start latent; ID 6
/// receives the corresponding projected generated frame. Audio-start and generated
/// positions also receive speaker-count and duration conditioning. Frame order
/// must match marker order separately in each unpadded sequence.
///
/// Inputs: floating-point embeddings `[B, S, 512]`, integer IDs `[B, S]`, floating-point voice
/// `[B, P, 512]`, and floating-point normalized audio history `[B, A, 32]`.
/// Output: `[B, S, 512]` in the embeddings' dtype, ready for the first decoder layer.
/// P and A may be zero. Cached decode supplies only one ID-6 position and frame.
///
/// Parameters come from the root checkpoint view: speaker feature projection
/// `flow_lm.condition_provider.conditioners.speaker_wavs.output_proj`, audio
/// projection `flow_lm.input_linear`, learned latent `flow_lm.bos_emb`, and the
/// `num_speakers` and `duration_delta` conditioners. All projections are bias-free;
/// stored bf16 tensors, voice, and audio history are converted to the embeddings' dtype.
/// Host fields default to one speaker and
/// zero duration delta; the latter generates 128 sinusoidal features in host f32,
/// then converts them to the embeddings' dtype.
const Input_Projection = struct {
    audio_prompt_proj: zml.nn.Linear,
    audio_latent_proj: zml.nn.Linear,
    bos_latent: zml.Tensor,
    num_speakers_embed: zml.Tensor,
    num_speakers_proj: zml.nn.Linear,
    duration_delta_proj: zml.nn.Linear,

    num_speakers: i64 = 1,
    duration_delta: f32 = 0,

    pub fn init(store: zml.io.TensorStore.View) Input_Projection {
        const here = @typeName(@This()) ++ ".init";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        return .{
            .audio_prompt_proj = .init(store.createReplicatedTensor(
                "flow_lm.condition_provider.conditioners.speaker_wavs.output_proj.weight",
                .{ .d, .ad },
            ), null, .ad),
            .audio_latent_proj = .init(store.createReplicatedTensor("flow_lm.input_linear.weight", .{ .d, .ld }), null, .ld),
            .bos_latent = store.createReplicatedTensor("flow_lm.bos_emb", .{.ld}),
            .num_speakers_embed = store.createReplicatedTensor(
                "flow_lm.condition_provider.conditioners.num_speakers.embed.weight",
                .{ .voc, .spk },
            ),
            .num_speakers_proj = .init(store.createReplicatedTensor(
                "flow_lm.condition_provider.conditioners.num_speakers.output_proj.weight",
                .{ .d, .spk },
            ), null, .spk),
            .duration_delta_proj = .init(store.createReplicatedTensor(
                "flow_lm.condition_provider.conditioners.duration_delta.output_proj.weight",
                .{ .d, .dur },
            ), null, .dur),
        };
    }

    // -- Description --

    /// Replaces control-token embeddings with the corresponding voice/audio features.
    ///
    /// `input`: floating-point `[B, S, 512]`, raw token embeddings from `Embedding.forward`.
    /// `token_ids`: integer `[B, S]`, normally i64, the same tokenizer/generation
    /// IDs used for that lookup. IDs 5, 4, and 6 select voice, audio-start, and
    /// generated-frame replacements; other positions retain `input`.
    /// `voice`: floating-point `[B, P, 512]`, ordered speaker features from the loaded voice
    /// asset, passed by the model. Each row must have exactly P ID-5 markers.
    /// `audio_latents`: floating-point `[B, A, 32]`, normalized completed frames from the
    /// generation loop, in ID-6 marker order with exactly A markers per row.
    /// The full-prefix path passes the whole history; prefill passes A=0, and
    /// cached decode passes one frame with one ID-6 position and P=0.
    /// All inputs share B and are unpadded; P and A may be zero.
    ///
    /// Returns `[B, S, 512]` in `input.dtype()`, prepared input features for the first
    /// decoder layer. Voice, audio history, and parameters are converted to this dtype.
    /// Projected BOS and generated frames include speaker/duration context.
    ///
    /// Dimension tags:
    /// - `.b`: batch; independent sequences, shared by IDs, embeddings and voices.
    /// - `.s`: sequence; token positions. IDs select which positions to replace.
    /// - `.d`: model dimension; 512 features at each output sequence position.
    /// - `.p`: prompt frames; voice-frame order, gathered into ID 5 positions.
    /// - `.ad`: input audio-prompt features; 512 features contracted by voice projection.
    /// - `.a`: generated audio frames; ordered history, gathered into ID 6 positions.
    /// - `.ld`: latent dimension; 32 features in BOS and each generated frame.
    /// - `.voc`: speaker-count vocabulary; the row selected by num_speakers.
    /// - `.spk`: speaker-count embedding dimension; 16 features projected to `.d`.
    /// - `.dur`: duration embedding dimension; 128 sinusoidal features projected to `.d`.
    ///
    /// Inputs: floating-point embeddings `[.b, .s, .d]`, integer IDs `[.b, .s]`,
    /// floating-point voice frames `[.b, .p, .ad]` and latents `[.b, .a, .ld=32]`.
    /// Output: `[.b, .s, .d]` in `input.dtype()`.
    pub fn forward(
        self: Input_Projection,
        input: zml.Tensor,
        token_ids: zml.Tensor,
        voice: zml.Tensor,
        audio_latents: zml.Tensor,
    ) zml.Tensor {
        const here = @typeName(@This()) ++ ".forward";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        const dtype = input.dtype();
        std.debug.assert(dtype.isFloat() and input.rank() == 3);
        std.debug.assert(token_ids.dtype().isInteger() and token_ids.rank() == 2);
        std.debug.assert(voice.dtype().isFloat() and voice.rank() == 3);
        std.debug.assert(audio_latents.dtype().isFloat() and audio_latents.rank() == 3);

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), "Input_Projection", .{});
        defer compiler.popLocation();

        const embeddings = input.withTags(.{ .b, .s, .d });
        const ids = token_ids.withTags(.{ .b, .s });
        const voice_frames = voice.convert(dtype).withTags(.{ .b, .p, .ad });
        const audio_frames = audio_latents.convert(dtype).withTags(.{ .b, .a, .ld });
        std.debug.assert(embeddings.dim(.b) == ids.dim(.b) and embeddings.dim(.s) == ids.dim(.s));
        std.debug.assert(embeddings.dim(.b) == voice_frames.dim(.b));
        std.debug.assert(embeddings.dim(.b) == audio_frames.dim(.b) and audio_frames.dim(.ld) == 32);

        const condition = self.speaker_condition(dtype).add(self.duration_condition(dtype));
        const bos = self.audio_latent_proj.forward(self.bos_latent.convert(dtype), dtype).add(condition);
        const bos_mask = ids.cmp(.EQ, zml.Tensor.scalar(4, ids.dtype())).broad(embeddings.shape());
        var output = bos_mask.select(bos.broad(embeddings.shape()), embeddings);

        // Empty histories have nothing to gather; prefill has no generated audio,
        // and cached decode has no new voice frames.
        if (voice_frames.dim(.p) > 0) {
            const projected = self.audio_prompt_proj.forward(voice_frames, dtype);
            output = replace_frame_markers(output, ids, 5, projected, .p);
        }
        if (audio_frames.dim(.a) > 0) {
            const projected = self.audio_latent_proj.forward(audio_frames, dtype);
            const conditioned = projected.add(condition.broad(projected.shape()));
            output = replace_frame_markers(output, ids, 6, conditioned, .a);
        }
        return output;
    }

    /// Host speaker count -> learned [.d=512] conditioning vector.
    fn speaker_condition(self: Input_Projection, dtype: zml.DataType) zml.Tensor {
        const count = zml.Tensor.constant(.{ .i64 = self.num_speakers });
        const speaker = self.num_speakers_embed.gather(.{ .voc = count }, .{}).convert(dtype);
        return self.num_speakers_proj.forward(speaker, dtype);
    }

    /// Host duration delta -> sinusoidal [.dur=128] features -> [.d=512].
    /// The scalar is fixed while tracing; this does not run a host loop per token.
    fn duration_condition(self: Input_Projection, dtype: zml.DataType) zml.Tensor {
        var features: [128]f32 = undefined;
        for (0..64) |i| {
            const exponent = @as(f32, @floatFromInt(i)) / 63.0;
            const phase = 1000.0 * self.duration_delta / std.math.pow(f32, 10000.0, exponent);
            features[i] = @cos(phase);
            features[i + 64] = @sin(phase);
        }
        const duration = zml.Tensor.constantTensor(.init(.{ .dur = 128 }, .f32), std.mem.asBytes(&features)).convert(dtype);
        return self.duration_delta_proj.forward(duration, dtype);
    }

    /// Insert projected frames [B,frame_axis,512] at the matching IDs in [B,S].
    /// The first marker takes frame 0, the second frame 1, independently per row.
    /// Non-marker positions retain their embeddings. The caller skips empty frames.
    fn replace_frame_markers(embeddings: zml.Tensor, ids: zml.Tensor, marker_id: i64, frames: zml.Tensor, comptime frame_axis: @EnumLiteral()) zml.Tensor {
        const mask = ids.cmp(.EQ, zml.Tensor.scalar(marker_id, ids.dtype()));
        const indices = mask.convert(.i32).cumulativeSum(.s).subConstant(1).maximum(zml.Tensor.scalar(0, .i32));
        // Both .b axes are shared, so gather selects frames within each batch row.
        const indexed_frames = frames.renameTag(frame_axis, .frame);
        const replacements = indexed_frames.gather(.{ .frame = indices }, .{});
        return mask.broad(embeddings.shape()).select(replacements, embeddings);
    }

    // -- Runtime --

    pub fn load(
        self: *const Input_Projection,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
    ) !zml.Bufferized(Input_Projection) {
        const here = @typeName(@This()) ++ ".load";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        var buffers = try zml.mem.bufferize(allocator, Input_Projection, self);
        errdefer unloadBuffers(&buffers);

        var loader: zml.io.Loader = try .init(allocator, platform, .default);
        defer loader.deinit();

        try loader.load(io, Input_Projection, self, &buffers, store, .{});
        try loader.await(io);

        return buffers;
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Input_Projection)) void {
        const here = @typeName(@This()) ++ ".unloadBuffers";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        zml.Buffer.deinitAll(Input_Projection, self);
    }
};

test Input_Projection {
    const test_data = @import("transformer_test_data.zig");

    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    var rng: std.Random.DefaultPrng = .init(42);
    const random = rng.random();

    const layer: Input_Projection = .{
        .audio_prompt_proj = .init(.fromShape(.init(.{ .d = 512, .ad = 512 }, .bf16)), null, .ad),
        .audio_latent_proj = .init(.fromShape(.init(.{ .d = 512, .ld = 32 }, .bf16)), null, .ld),
        .bos_latent = .fromShape(.init(.{ .ld = 32 }, .bf16)),
        .num_speakers_embed = .fromShape(.init(.{ .voc = 32, .spk = 16 }, .bf16)),
        .num_speakers_proj = .init(.fromShape(.init(.{ .d = 512, .spk = 16 }, .bf16)), null, .spk),
        .duration_delta_proj = .init(.fromShape(.init(.{ .d = 512, .dur = 128 }, .bf16)), null, .dur),
        .num_speakers = random.intRangeLessThan(i64, 0, 32),
        .duration_delta = random.float(f32),
    };
    var weight_buffers = try zml.mem.bufferize(allocator, Input_Projection, &layer);
    const tensors = try zml.meta.collectValues(zml.Tensor, allocator, &layer);
    defer allocator.free(tensors);
    const buffers = try zml.meta.collectMutPtrs(zml.Buffer, allocator, &weight_buffers);
    defer allocator.free(buffers);
    var loaded: usize = 0;
    defer for (buffers[0..loaded]) |buffer| buffer.deinit();
    for (tensors, buffers) |tensor, buffer| {
        const values = try zml.Slice.alloc(allocator, tensor.shape());
        defer values.free(allocator);
        for (values.items(zml.floats.BFloat16)) |*value| value.* = .fromF32(random.float(f32) * 2 - 1);
        buffer.* = try zml.Buffer.fromSlice(io, platform, values);
        loaded += 1;
    }

    const input_shape: zml.Shape = .init(.{ .b = 2, .s = 7, .d = 512 }, .f32);
    var input: [2][7][512]f32 = undefined;
    for (&input) |*batch| {
        for (batch) |*position| {
            for (position) |*value| value.* = random.float(f32) * 2 - 1;
        }
    }
    var input_buffer = try zml.Buffer.fromBytes(io, platform, input_shape, std.mem.asBytes(&input));
    defer input_buffer.deinit();

    // Different marker orders in each batch row exercise per-sequence frame indexing.
    // Also cover empty voice/audio histories, where both gathers must be skipped.
    for ([_]usize{ 2, 0 }) |frame_count| {
        const ids_shape: zml.Shape = .init(.{ .b = 2, .s = 7 }, .i64);
        var ids: [2][7]i64 = .{ .{ 7, 5, 4, 6, 5, 6, 7 }, .{ 6, 5, 7, 5, 4, 7, 6 } };
        for (&ids) |*batch| {
            for (batch) |*id| {
                if (id.* == 7 or (frame_count == 0 and (id.* == 5 or id.* == 6))) {
                    id.* = random.intRangeLessThan(i64, 7, 4001);
                }
            }
        }
        var ids_buffer = try zml.Buffer.fromBytes(io, platform, ids_shape, std.mem.asBytes(&ids));
        defer ids_buffer.deinit();

        const voice_shape: zml.Shape = .init(.{ .b = 2, .p = frame_count, .ad = 512 }, .f32);
        const voice = try zml.Slice.alloc(allocator, voice_shape);
        defer voice.free(allocator);
        for (voice.items(f32)) |*value| value.* = random.float(f32) * 2 - 1;
        var voice_buffer = try zml.Buffer.fromSlice(io, platform, voice);
        defer voice_buffer.deinit();

        const audio_shape: zml.Shape = .init(.{ .b = 2, .a = frame_count, .ld = 32 }, .f32);
        const audio = try zml.Slice.alloc(allocator, audio_shape);
        defer audio.free(allocator);
        for (audio.items(f32)) |*value| value.* = random.float(f32) * 2 - 1;
        var audio_buffer = try zml.Buffer.fromSlice(io, platform, audio);
        defer audio_buffer.deinit();

        var exe = try platform.compile(allocator, io, layer, .forward, .{
            zml.Tensor.fromShape(input_shape),
            zml.Tensor.fromShape(ids_shape),
            zml.Tensor.fromShape(voice_shape),
            zml.Tensor.fromShape(audio_shape),
        }, .{});
        defer exe.deinit();
        var output = try exe.eval(allocator, io, .{ weight_buffers, input_buffer, ids_buffer, voice_buffer, audio_buffer });
        defer output.deinit();

        const expected = if (frame_count > 0) &test_data.input_projection else &test_data.input_projection_empty;
        const expected_slice = zml.Slice.init(input_shape, std.mem.asBytes(expected));
        try zml.testing.expectClose(io, expected_slice, output, .{
            .absolute_tolerance = 1e-4,
            .relative_tolerance = 1e-4,
            .minimum_close_fraction = 1,
        });

        // Leave auxiliary inputs in f32 to check conversion to the primary input dtype.
        for ([_]zml.DataType{ .f16, .bf16, .f64 }) |dtype| {
            var typed_exe = try platform.compile(allocator, io, layer, .forward, .{
                zml.Tensor.fromShape(input_shape.withDtype(dtype)),
                zml.Tensor.fromShape(ids_shape),
                zml.Tensor.fromShape(voice_shape),
                zml.Tensor.fromShape(audio_shape),
            }, .{});
            defer typed_exe.deinit();
            try std.testing.expect(typed_exe.output_shapes[0].eqlWithTags(input_shape.withDtype(dtype)));
        }
    }
}

// --- Gradium_Phonon_TTS_Decoder_Layer ---
// The transformer block, which runs in multiple passes.
//
// Broadly:
// ```
// a = input_layernorm(x)
// a = self_attn(a)
// y = x + a
//
// m = post_attention_layernorm(y)
// m = mlp(m)
// output = y + m
// ```

/// Mixes sequence context, then refines each position's features.
///
/// The attention branch normalizes its input, performs causal self-attention,
/// and adds the result to the original features. The MLP branch normalizes that
/// sum, applies a 512 -> 2048 -> 512 network with GELU, and adds another residual.
/// Input and output are floating-point `[B, S, 512]`, preserving the input dtype;
/// batch rows remain independent.
///
/// `forward` recomputes a complete sequence starting at position zero.
/// `forward_cached` processes Q new positions using stored K/V history and returns
/// the updated cache along with features `[B, Q, 512]` in the input dtype.
/// Each cache tensor retains its own storage dtype.
///
/// Each layer has separate bf16 parameters under
/// `flow_lm.transformer.layers.{index}`: `norm1`, `self_attn.in_proj`,
/// `self_attn.out_proj`, `norm2`, `linear1`, and `linear2`.
/// Parameters are converted to the input dtype; norm epsilon is 1e-5 and GELU
/// uses erf. ZML evaluates normalization statistics, RoPE trigonometry, attention
/// softmax, and erf internally in f32, then converts back to the input dtype.
///
/// Reference:
/// ```
/// GradiumPhononTTSDecoderLayer
/// ├── self_attn: GradiumPhononTTSAttention
/// │   ├── q_proj: Linear
/// │   │   └── weight: shape=(512, 512), dtype=torch.float32
/// │   │       ├── loaded: model.layers.0.self_attn.q_proj.weight
/// │   │       └── checkpoint: flow_lm.transformer.layers.0.self_attn.in_proj.weight
/// │   ├── k_proj: Linear
/// │   │   └── weight: shape=(512, 512), dtype=torch.float32
/// │   │       ├── loaded: model.layers.0.self_attn.k_proj.weight
/// │   │       └── checkpoint: flow_lm.transformer.layers.0.self_attn.in_proj.weight
/// │   ├── v_proj: Linear
/// │   │   └── weight: shape=(512, 512), dtype=torch.float32
/// │   │       ├── loaded: model.layers.0.self_attn.v_proj.weight
/// │   │       └── checkpoint: flow_lm.transformer.layers.0.self_attn.in_proj.weight
/// │   └── o_proj: Linear
/// │       └── weight: shape=(512, 512), dtype=torch.float32
/// │           ├── loaded: model.layers.0.self_attn.o_proj.weight
/// │           └── checkpoint: flow_lm.transformer.layers.0.self_attn.out_proj.weight
/// ├── mlp: GradiumPhononTTSMLP
/// │   ├── activation_fn: GELUActivation
/// │   ├── fc1: Linear
/// │   │   └── weight: shape=(2048, 512), dtype=torch.float32
/// │   │       ├── loaded: model.layers.0.mlp.fc1.weight
/// │   │       └── checkpoint: flow_lm.transformer.layers.0.linear1.weight
/// │   └── fc2: Linear
/// │       └── weight: shape=(512, 2048), dtype=torch.float32
/// │           ├── loaded: model.layers.0.mlp.fc2.weight
/// │           └── checkpoint: flow_lm.transformer.layers.0.linear2.weight
/// ├── input_layernorm: LayerNorm
/// │   ├── weight: shape=(512,), dtype=torch.float32
/// │   │   ├── loaded: model.layers.0.input_layernorm.weight
/// │   │   └── checkpoint: flow_lm.transformer.layers.0.norm1.weight
/// │   └── bias: shape=(512,), dtype=torch.float32
/// │       ├── loaded: model.layers.0.input_layernorm.bias
/// │       └── checkpoint: flow_lm.transformer.layers.0.norm1.bias
/// └── post_attention_layernorm: LayerNorm
///     ├── weight: shape=(512,), dtype=torch.float32
///     │   ├── loaded: model.layers.0.post_attention_layernorm.weight
///     │   └── checkpoint: flow_lm.transformer.layers.0.norm2.weight
///     └── bias: shape=(512,), dtype=torch.float32
///         ├── loaded: model.layers.0.post_attention_layernorm.bias
///         └── checkpoint: flow_lm.transformer.layers.0.norm2.bias
/// ```
const Gradium_Phonon_TTS_Decoder_Layer = struct {
    input_layernorm: Input_Layer_Norm,
    self_attn: Gradium_Phonon_TTS_Attention,
    post_attention_layernorm: Post_Attention_Layer_Norm,
    mlp: Gradium_Phonon_TTS_MLP,

    pub fn init(store: zml.io.TensorStore.View) Gradium_Phonon_TTS_Decoder_Layer {
        const here = @typeName(@This()) ++ ".init";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        return .{
            .input_layernorm = .init(store),
            .self_attn = .init(store),
            .post_attention_layernorm = .init(store),
            .mlp = .init(store),
        };
    }

    /// Runs a complete sequence through attention and MLP residual branches.
    ///
    /// `input`: floating-point `[B, S, 512]`, prepared model inputs for the first decoder
    /// layer or residual features from the preceding layer. This method assigns
    /// tags `.b`, `.s`, and `.d`; sequences must be unpadded.
    ///
    /// Returns `[B, S, 512]` in `input.dtype()`, after both residual additions:
    /// `x = input + attention(norm1(input)); output = x + mlp(norm2(x))`.
    /// The output feeds the next decoder layer or the model's final normalization.
    ///
    /// Dimension tags:
    /// - `.d`: model dimension, the 512 features describing each token. Layer norm
    ///   normalizes across these features; the MLP transforms them independently per token.
    /// - `.b`: batch; independent sequences, never mixed by the layer.
    /// - `.s`: sequence; token positions. Causal attention mixes information from
    ///   earlier positions and the current position; norm and MLP act per token.
    /// All branches retain these tags so residual additions align corresponding
    /// tokens and features. Attention's internal tags are documented in its forward.
    ///
    /// Shape: `[.b, .s, .d=512] -> [.b, .s, .d=512]`, preserving the input dtype.
    pub fn forward(self: Gradium_Phonon_TTS_Decoder_Layer, input: zml.Tensor) zml.Tensor {
        const here = @typeName(@This()) ++ ".forward";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        std.debug.assert(input.dtype().isFloat() and input.rank() == 3);
        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), "Gradium_Phonon_TTS_Decoder_Layer", .{});
        defer compiler.popLocation();

        const hidden = input.withTags(.{ .b, .s, .d });
        const normalized = self.input_layernorm.forward(hidden);
        const attention = self.self_attn.forward_uncached(normalized);
        const residual = hidden.add(attention);
        return self.add_mlp_residual(residual);
    }

    /// Runs attention and MLP residual branches for a new chunk.
    ///
    /// `input`: floating [B,chunk_length,512], prepared model features for layer 0
    /// or the previous layer's output. Each row contains only real positions.
    /// `position`: scalar i32 first absolute slot, zero at prefill. Caller ensures
    /// position >= 0 and position+chunk_length <= capacity.
    /// `cache`: K/V history, each floating [6,B,capacity,8,64], with the preceding
    /// prefix populated and unused slots finite. Keys include RoPE; values do not.
    /// Attention converts history to input.dtype() and new entries to storage dtype.
    /// `layer_index`: host usize 0..5, selecting this layer's history.
    ///
    /// Returns residual features [B,chunk_length,512] in input.dtype() and history
    /// updated for this layer. Other layers and each cache's storage dtype persist.
    /// Tags: .b independent sequences, .s new positions, .d=512 features;
    /// cache .layer=6, .k=capacity, .h=8 heads, .hd=64 coordinates per head.
    pub fn forward_cached(
        self: Gradium_Phonon_TTS_Decoder_Layer,
        input: zml.Tensor,
        position: zml.Tensor,
        cache: KV_Cache,
        layer_index: usize,
    ) CachedHidden {
        const here = @typeName(@This()) ++ ".forward_cached";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        std.debug.assert(input.dtype().isFloat() and input.rank() == 3);
        const hidden = input.withTags(.{ .b, .s, .d });
        const normalized = self.input_layernorm.forward(hidden);
        const attention = self.self_attn.forward_cached(normalized, position, cache, layer_index);
        const residual = hidden.add(attention.hidden);
        return .{ .hidden = self.add_mlp_residual(residual), .cache = attention.cache };
    }

    /// Post-attention residual [B,S,512] -> normalize, MLP, residual addition.
    fn add_mlp_residual(self: Gradium_Phonon_TTS_Decoder_Layer, residual: zml.Tensor) zml.Tensor {
        const normalized = self.post_attention_layernorm.forward(residual);
        const update = self.mlp.forward(normalized);
        return residual.add(update);
    }

    pub fn load(
        self: *const Gradium_Phonon_TTS_Decoder_Layer,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
    ) !zml.Bufferized(Gradium_Phonon_TTS_Decoder_Layer) {
        const here = @typeName(@This()) ++ ".load";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        var buffers = try zml.mem.bufferize(allocator, Gradium_Phonon_TTS_Decoder_Layer, self);
        errdefer unloadBuffers(&buffers);

        var loader: zml.io.Loader = try .init(allocator, platform, .default);
        defer loader.deinit();

        try loader.load(io, Gradium_Phonon_TTS_Decoder_Layer, self, &buffers, store, .{});
        try loader.await(io);

        return buffers;
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Gradium_Phonon_TTS_Decoder_Layer)) void {
        const here = @typeName(@This()) ++ ".unloadBuffers";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        zml.Buffer.deinitAll(Gradium_Phonon_TTS_Decoder_Layer, self);
    }
};

test Gradium_Phonon_TTS_Decoder_Layer {
    const test_data = @import("transformer_test_data.zig");

    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    var rng: std.Random.DefaultPrng = .init(42);
    const random = rng.random();

    const layer: Gradium_Phonon_TTS_Decoder_Layer = .{
        .input_layernorm = .{
            .weight = .fromShape(.init(.{ .d = 512 }, .bf16)),
            .bias = .fromShape(.init(.{ .d = 512 }, .bf16)),
            .eps = 1e-5,
        },
        .self_attn = .{
            .qkv_proj = .init(.fromShape(.init(.{ .qkv = 1536, .d = 512 }, .bf16)), null, .d),
            .out_proj = .init(.fromShape(.init(.{ .dout = 512, .d = 512 }, .bf16)), null, .d),
        },
        .post_attention_layernorm = .{
            .weight = .fromShape(.init(.{ .d = 512 }, .bf16)),
            .bias = .fromShape(.init(.{ .d = 512 }, .bf16)),
            .eps = 1e-5,
        },
        .mlp = .{
            .fc1 = .init(.fromShape(.init(.{ .dout = 2048, .d = 512 }, .bf16)), null, .d),
            .fc2 = .init(.fromShape(.init(.{ .d = 512, .dout = 2048 }, .bf16)), null, .dout),
        },
    };
    const input_shape: zml.Shape = .init(.{ .b = 2, .s = 3, .d = 512 }, .f32);
    var input: [2][3][512]f32 = undefined;
    for (&input) |*batch| {
        for (batch) |*position| {
            for (position) |*value| value.* = random.float(f32) * 2 - 1;
        }
    }
    var input_buffer = try zml.Buffer.fromBytes(io, platform, input_shape, std.mem.asBytes(&input));
    defer input_buffer.deinit();

    var weight_buffers = try zml.mem.bufferize(allocator, Gradium_Phonon_TTS_Decoder_Layer, &layer);
    const tensors = try zml.meta.collectValues(zml.Tensor, allocator, &layer);
    defer allocator.free(tensors);
    const buffers = try zml.meta.collectMutPtrs(zml.Buffer, allocator, &weight_buffers);
    defer allocator.free(buffers);
    var loaded: usize = 0;
    defer for (buffers[0..loaded]) |buffer| buffer.deinit();
    for (tensors, buffers) |tensor, buffer| {
        const values = try zml.Slice.alloc(allocator, tensor.shape());
        defer values.free(allocator);
        for (values.items(zml.floats.BFloat16)) |*value| value.* = .fromF32(random.float(f32) * 2 - 1);
        buffer.* = try zml.Buffer.fromSlice(io, platform, values);
        loaded += 1;
    }

    var exe = try platform.compile(allocator, io, layer, .forward, .{zml.Tensor.fromShape(input_shape)}, .{});
    defer exe.deinit();
    var output = try exe.eval(allocator, io, .{ weight_buffers, input_buffer });
    defer output.deinit();

    const expected = test_data.decoder_layer;
    const expected_slice = zml.Slice.init(input_shape, std.mem.asBytes(&expected));
    // Large random projections can cancel near zero; allow f32 reduction-order error.
    try zml.testing.expectClose(io, expected_slice, output, .{
        .absolute_tolerance = 1e-3,
        .relative_tolerance = 1e-4,
        .minimum_close_fraction = 1,
    });

    for ([_]zml.DataType{ .f16, .bf16, .f64 }) |dtype| {
        var typed_exe = try platform.compile(allocator, io, layer, .forward, .{
            zml.Tensor.fromShape(input_shape.withDtype(dtype)),
        }, .{});
        defer typed_exe.deinit();
        try std.testing.expect(typed_exe.output_shapes[0].eqlWithTags(input_shape.withDtype(dtype)));
    }
}

test "Gradium_Phonon_TTS_Decoder_Layer: cached" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    var rng: std.Random.DefaultPrng = .init(42);
    const random = rng.random();

    const layer: Gradium_Phonon_TTS_Decoder_Layer = .{
        .input_layernorm = .{
            .weight = .fromShape(.init(.{ .d = 512 }, .bf16)),
            .bias = .fromShape(.init(.{ .d = 512 }, .bf16)),
            .eps = 1e-5,
        },
        .self_attn = .{
            .qkv_proj = .init(.fromShape(.init(.{ .qkv = 1536, .d = 512 }, .bf16)), null, .d),
            .out_proj = .init(.fromShape(.init(.{ .dout = 512, .d = 512 }, .bf16)), null, .d),
        },
        .post_attention_layernorm = .{
            .weight = .fromShape(.init(.{ .d = 512 }, .bf16)),
            .bias = .fromShape(.init(.{ .d = 512 }, .bf16)),
            .eps = 1e-5,
        },
        .mlp = .{
            .fc1 = .init(.fromShape(.init(.{ .dout = 2048, .d = 512 }, .bf16)), null, .d),
            .fc2 = .init(.fromShape(.init(.{ .d = 512, .dout = 2048 }, .bf16)), null, .dout),
        },
    };
    const input_shape: zml.Shape = .init(.{ .b = 2, .s = 3, .d = 512 }, .f32);
    var input: [2][3][512]f32 = undefined;
    for (&input) |*batch| {
        for (batch) |*position| {
            for (position) |*value| value.* = random.float(f32) * 2 - 1;
        }
    }
    var input_buffer = try zml.Buffer.fromBytes(io, platform, input_shape, std.mem.asBytes(&input));
    defer input_buffer.deinit();

    var weight_buffers = try zml.mem.bufferize(allocator, Gradium_Phonon_TTS_Decoder_Layer, &layer);
    const tensors = try zml.meta.collectValues(zml.Tensor, allocator, &layer);
    defer allocator.free(tensors);
    const buffers = try zml.meta.collectMutPtrs(zml.Buffer, allocator, &weight_buffers);
    defer allocator.free(buffers);
    var loaded: usize = 0;
    defer for (buffers[0..loaded]) |buffer| buffer.deinit();
    for (tensors, buffers) |tensor, buffer| {
        const values = try zml.Slice.alloc(allocator, tensor.shape());
        defer values.free(allocator);
        for (values.items(zml.floats.BFloat16)) |*value| value.* = .fromF32(random.float(f32) * 2 - 1);
        buffer.* = try zml.Buffer.fromSlice(io, platform, values);
        loaded += 1;
    }

    var uncached_exe = try platform.compile(allocator, io, layer, .forward, .{zml.Tensor.fromShape(input_shape)}, .{});
    defer uncached_exe.deinit();
    var uncached_output = try uncached_exe.eval(allocator, io, .{ weight_buffers, input_buffer });
    defer uncached_output.deinit();

    // Leave unused capacity after the sequence to exercise masking of future slots.
    const initial_cache = KV_Cache.init(2, 5);
    var cache_buffers = try initial_cache.initBuffer(allocator, io, platform);
    defer KV_Cache.deinitBuffer(&cache_buffers);
    var cached_output: [2][3][512]f32 = undefined;

    // Prefill two positions, then decode the third using the returned history.
    inline for (.{ .{ .start = 0, .end = 2 }, .{ .start = 2, .end = 3 } }) |step| {
        const count = step.end - step.start;
        const chunk_shape: zml.Shape = .init(.{ .b = 2, .s = count, .d = 512 }, .f32);
        var chunk: [2][count][512]f32 = undefined;
        for (&chunk, input) |*batch, source| {
            @memcpy(batch, source[step.start..step.end]);
        }
        var chunk_buffer = try zml.Buffer.fromBytes(io, platform, chunk_shape, std.mem.asBytes(&chunk));
        defer chunk_buffer.deinit();
        const position: i32 = step.start;
        var position_buffer = try zml.Buffer.fromBytes(io, platform, .scalar(.i32), std.mem.asBytes(&position));
        defer position_buffer.deinit();

        var cached_exe = try platform.compile(allocator, io, layer, .forward_cached, .{
            zml.Tensor.fromShape(chunk_shape),
            zml.Tensor.fromShape(.scalar(.i32)),
            initial_cache,
            @as(usize, 2),
        }, .{});
        defer cached_exe.deinit();
        var runner = try cached_exe.runner(allocator);
        defer runner.deinit(allocator);
        var result: zml.Bufferized(CachedHidden) = undefined;
        runner.run(io, .{ weight_buffers, chunk_buffer, position_buffer, cache_buffers, @as(usize, 2) }, .{&result}, .{ .wait = true });
        defer result.hidden.deinit();
        KV_Cache.deinitBuffer(&cache_buffers);
        cache_buffers = result.cache;

        const values = try result.hidden.getValue([2][count][512]f32, io);
        for (&cached_output, values) |*batch, source| {
            @memcpy(batch[step.start..step.end], &source);
        }
    }

    const cached_slice = zml.Slice.init(input_shape, std.mem.asBytes(&cached_output));
    // Cached and full-sequence attention use different matmul/reduction shapes.
    try zml.testing.expectClose(io, uncached_output, cached_slice, .{
        .absolute_tolerance = 1e-3,
        .relative_tolerance = 1e-4,
        .minimum_close_fraction = 1,
    });

    for ([_]zml.DataType{ .f16, .bf16, .f32, .f64 }) |dtype| {
        const cache_shape: zml.Shape = .init(.{ .layer = 6, .b = 2, .k = 5, .h = 8, .hd = 64 }, dtype);
        // Check matching dtypes as well as separate K/V storage dtypes.
        for ([_]KV_Cache{
            .{ .k = .fromShape(cache_shape), .v = .fromShape(cache_shape) },
            .{ .k = .fromShape(cache_shape.withDtype(.f16)), .v = .fromShape(cache_shape.withDtype(.bf16)) },
        }) |cache| {
            var cached_exe = try platform.compile(allocator, io, layer, .forward_cached, .{
                zml.Tensor.fromShape(input_shape.withDtype(dtype)),
                zml.Tensor.fromShape(.init(.{}, .i32)),
                cache,
                @as(usize, 2),
            }, .{});
            defer cached_exe.deinit();
            try std.testing.expect(cached_exe.output_shapes[0].eqlWithTags(input_shape.withDtype(dtype)));
            try std.testing.expect(cached_exe.output_shapes[1].eqlWithTags(cache.k.shape()));
            try std.testing.expect(cached_exe.output_shapes[2].eqlWithTags(cache.v.shape()));
        }
    }
}

// --- Gradium_Phonon_TTS_Attention ---

/// Lets each sequence position gather information from its causal context.
///
/// A packed projection creates queries, keys, and values for eight heads of
/// 64 features. RoPE rotates queries and keys using absolute sequence positions.
/// Scaled dot-product attention weights visible values, then an output projection
/// combines the heads into 512 features. The decoder layer adds the residual.
///
/// Input and output: floating-point `[B, new_positions, 512]`, preserving dtype.
/// `forward_uncached` processes the entire supplied sequence starting at zero.
/// `forward_cached` processes only the new chunk: the whole prompt at prefill,
/// or one position at decode. It writes that chunk's K/V into the selected layer's
/// history and attends to the stored prefix plus visible new positions.
/// Both paths require unpadded inputs.
///
/// Parameters come from `flow_lm.transformer.layers.{index}`:
/// `self_attn.in_proj.weight` [1536, 512] and `self_attn.out_proj.weight` [512, 512].
/// Both are bias-free and converted from bf16 to the input dtype. RoPE uses
/// interleaved pairs and theta 10000; attention uses scale 1/sqrt(64) and softmax
/// over key positions. ZML computes RoPE trigonometry and attention softmax in
/// f32 internally, then converts back to the input dtype.
///
/// Reference:
/// ```
/// self_attn: GradiumPhononTTSAttention
/// ├── q_proj: Linear
/// │   └── weight: shape=(512, 512), dtype=torch.float32
/// │       ├── loaded: model.layers.1.self_attn.q_proj.weight
/// │       └── checkpoint: flow_lm.transformer.layers.1.self_attn.in_proj.weight
/// ├── k_proj: Linear
/// │   └── weight: shape=(512, 512), dtype=torch.float32
/// │       ├── loaded: model.layers.1.self_attn.k_proj.weight
/// │       └── checkpoint: flow_lm.transformer.layers.1.self_attn.in_proj.weight
/// ├── v_proj: Linear
/// │   └── weight: shape=(512, 512), dtype=torch.float32
/// │       ├── loaded: model.layers.1.self_attn.v_proj.weight
/// │       └── checkpoint: flow_lm.transformer.layers.1.self_attn.in_proj.weight
/// └── o_proj: Linear
///     └── weight: shape=(512, 512), dtype=torch.float32
///         ├── loaded: model.layers.1.self_attn.o_proj.weight
///         └── checkpoint: flow_lm.transformer.layers.1.self_attn.out_proj.weight
/// ```
const Gradium_Phonon_TTS_Attention = struct {
    qkv_proj: zml.nn.Linear,
    out_proj: zml.nn.Linear,

    const num_heads = 8;
    const head_dim = 64;

    pub fn init(store: zml.io.TensorStore.View) Gradium_Phonon_TTS_Attention {
        const here = @typeName(@This()) ++ ".init";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        return .{
            .qkv_proj = .init(store.createReplicatedTensor("self_attn.in_proj.weight", .{ .qkv, .d }), null, .d),
            .out_proj = .init(store.createReplicatedTensor("self_attn.out_proj.weight", .{ .dout, .d }), null, .d),
        };
    }

    /// Computes causal attention over a complete unpadded sequence.
    ///
    /// `input`: floating [B, S, 512], pre-attention features from Input_Layer_Norm.
    /// All S positions are processed together, starting at absolute position zero.
    /// Returns projected features [B, S, 512] in input.dtype(), before the caller's
    /// residual addition. Parameters are converted to the input dtype.
    ///
    /// Tags: .b independent sequences, .s input/output positions, .d model features.
    /// Attention uses .q query positions and .k key/value positions, both length S;
    /// .h=8 heads and .hd=64 coordinates within each head. Future keys are masked.
    pub fn forward_uncached(self: Gradium_Phonon_TTS_Attention, input: zml.Tensor) zml.Tensor {
        const here = @typeName(@This()) ++ ".forward_uncached";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), "Gradium_Phonon_TTS_Attention", .{});
        defer compiler.popLocation();

        const hidden = input.withTags(.{ .b, .s, .d });
        const projected = self.project_qkv(hidden);
        const rotated = apply_rope(projected, null);
        const mask = zml.nn.causalAttnMask(.{ .q = hidden.dim(.s), .k = hidden.dim(.s) }, hidden.dtype(), null);
        return self.attend_and_project(rotated.queries, rotated.keys, rotated.values, mask);
    }

    /// Processes new positions using their cached prefix and causal context.
    ///
    /// `input`: floating [B, chunk_length, 512], pre-attention features from
    /// Input_Layer_Norm. chunk_length counts only positions processed THIS call:
    /// the whole prompt during prefill, or one previous audio frame during decode.
    /// `position`: scalar i32, the first new absolute sequence position, supplied
    /// by the generation loop. New positions occupy [position, position+chunk_length).
    /// Prefill starts at zero. Caller must ensure position >= 0 and
    /// position+chunk_length <= capacity.
    /// `cache`: K/V history, each floating [6, B, capacity, 8, 64]. Earlier slots
    /// must be populated; unused slots must be finite (normally zero). Keys already
    /// include RoPE. New entries convert to each cache tensor's storage dtype;
    /// history converts back to input.dtype() for attention.
    /// `layer_index`: host usize 0..5, selecting this attention layer's history.
    ///
    /// Returns features [B, chunk_length, 512] in input.dtype(), before residual
    /// addition, and updated caches preserving their respective shapes and dtypes.
    /// Each query sees itself and all earlier positions; future/unused slots are masked.
    ///
    /// Tags: .b batch, .s new input/output positions, .d=512 features;
    /// .layer=6 layers, .k=capacity cache slots, .h=8 heads, .hd=64 coordinates.
    /// Internally .s becomes .q for queries, whose length is chunk_length;
    /// .k for the complete stored history has length capacity.
    pub fn forward_cached(
        self: Gradium_Phonon_TTS_Attention,
        input: zml.Tensor,
        position: zml.Tensor,
        cache: KV_Cache,
        layer_index: usize,
    ) CachedHidden {
        const here = @typeName(@This()) ++ ".forward_cached";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        const hidden = input.withTags(.{ .b, .s, .d });
        const chunk_length = hidden.dim(.s);
        const capacity = cache.k.dim(.k);
        std.debug.assert(position.rank() == 0 and position.dtype() == .i32);
        std.debug.assert(cache.k.dtype().isFloat() and cache.v.dtype().isFloat());
        std.debug.assert(hidden.dim(.b) == cache.k.dim(.b) and chunk_length <= capacity);
        std.debug.assert(layer_index < 6);

        const positions = absolute_positions(position, chunk_length);
        const projected = self.project_qkv(hidden);
        const rotated = apply_rope(projected, positions);
        const updated_cache = update_cache(cache, layer_index, position, rotated.keys, rotated.values);
        const history = read_cached_layer(updated_cache, layer_index, hidden.dtype());
        const mask = causal_mask(positions, capacity, hidden.dtype());
        const output = self.attend_and_project(rotated.queries, history.keys, history.values, mask);
        return .{ .hidden = output, .cache = updated_cache };
    }

    /// Before RoPE all fields are [B, new_positions, 8, 64], tagged .b/.s/.h/.hd.
    /// After RoPE, queries use .q instead of .s and keys/values use .k instead.
    const QKV = struct {
        queries: zml.Tensor,
        keys: zml.Tensor,
        values: zml.Tensor,
    };

    /// [B, new_positions, 512] -> three head-split projections. The packed
    /// projection axis .qkv holds 1536 features: queries, then keys, then values.
    fn project_qkv(self: Gradium_Phonon_TTS_Attention, hidden: zml.Tensor) QKV {
        std.debug.assert(hidden.dtype().isFloat() and hidden.rank() == 3);
        std.debug.assert(hidden.dim(.d) == num_heads * head_dim);
        const packed_features = self.qkv_proj.forward(hidden, hidden.dtype());
        const projections = packed_features.chunkExact(.qkv, 3);
        return .{
            .queries = projections[0].splitAxis(.qkv, .{ .h = num_heads, .hd = head_dim }),
            .keys = projections[1].splitAxis(.qkv, .{ .h = num_heads, .hd = head_dim }),
            .values = projections[2].splitAxis(.qkv, .{ .h = num_heads, .hd = head_dim }),
        };
    }

    /// Rotate Q/K at absolute positions [.s], or positions 0..S-1 when null.
    /// The checkpoint packs adjacent real/imaginary coordinates, so RoPE must
    /// use interleaved pairs. Values are not rotated. Assign attention axis tags.
    fn apply_rope(projected: QKV, positions: ?zml.Tensor) QKV {
        const options: zml.nn.RopeOpts = .{
            .layout = .interleaved,
            .scaling = .{ .default = .{ .rope_theta = 10000 } },
        };
        return .{
            .queries = zml.nn.rope(projected.queries, positions, options).rename(.{ .s = .q }),
            .keys = zml.nn.rope(projected.keys, positions, options).rename(.{ .s = .k }),
            .values = projected.values.rename(.{ .s = .k }),
        };
    }

    /// Scalar i32 first position -> [.s=chunk_length] absolute i32 positions.
    fn absolute_positions(position: zml.Tensor, chunk_length: i64) zml.Tensor {
        const offsets = zml.Tensor.arange(.{ .end = chunk_length }, .i32).withTags(.{.s});
        return offsets.add(position.broad(offsets.shape()));
    }

    /// Write new [B, chunk_length, 8, 64] K/V into one layer's consecutive slots.
    /// Keep all layers, capacity, and storage dtypes intact. The prediction wrapper
    /// decides buffer donation; this helper only describes the tensor update.
    fn update_cache(cache: KV_Cache, layer_index: usize, position: zml.Tensor, keys: zml.Tensor, values: zml.Tensor) KV_Cache {
        const layer = zml.Tensor.scalar(@as(i32, @intCast(layer_index)), .i32);
        return .{
            .k = cache.k.dynamicUpdateSlice(.{ .layer = layer, .k = position }, keys.convert(cache.k.dtype())),
            .v = cache.v.dynamicUpdateSlice(.{ .layer = layer, .k = position }, values.convert(cache.v.dtype())),
        };
    }

    const Cached_Layer = struct { keys: zml.Tensor, values: zml.Tensor };

    /// Select one layer's full [B, capacity, 8, 64] history and convert it to the
    /// compute dtype. The causal mask, rather than this read, excludes future slots.
    fn read_cached_layer(cache: KV_Cache, layer_index: usize, dtype: zml.DataType) Cached_Layer {
        const start: i64 = @intCast(layer_index);
        return .{
            .keys = cache.k.slice(.layer, .{ .start = start, .end = start + 1 }).squeeze(.layer).convert(dtype),
            .values = cache.v.slice(.layer, .{ .start = start, .end = start + 1 }).squeeze(.layer).convert(dtype),
        };
    }

    /// [.s=new_positions] absolute query indices -> [.q=new_positions, .k=capacity]
    /// additive mask. Zero allows a key; negative infinity blocks keys after the query.
    fn causal_mask(positions: zml.Tensor, capacity: i64, dtype: zml.DataType) zml.Tensor {
        const shape: zml.Shape = .init(.{ .q = positions.dim(.s), .k = capacity }, .i32);
        const queries = positions.rename(.{ .s = .q }).broad(shape);
        const keys = zml.Tensor.iota(shape, .k);
        const visible = keys.cmp(.LE, queries);
        return visible.select(zml.Tensor.scalar(0, dtype), zml.Tensor.scalar(-std.math.inf(f32), dtype));
    }

    /// Attend from [B, new_positions, 8, 64] queries to keys/values, then merge
    /// heads and project to [B, new_positions, 512], tagged .b/.s/.d.
    /// .dout distinguishes the output projection's result from its contracted .d axis.
    fn attend_and_project(self: Gradium_Phonon_TTS_Attention, queries: zml.Tensor, keys: zml.Tensor, values: zml.Tensor, mask: zml.Tensor) zml.Tensor {
        const attended = zml.nn.sdpa(queries, keys, values, .{ .attn_mask = mask });
        const features = attended.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s });
        return self.out_proj.forward(features, queries.dtype()).rename(.{ .dout = .d });
    }

    // -- Runtime --

    pub fn load(
        self: *const Gradium_Phonon_TTS_Attention,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
    ) !zml.Bufferized(Gradium_Phonon_TTS_Attention) {
        const here = @typeName(@This()) ++ ".load";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        var buffers = try zml.mem.bufferize(allocator, Gradium_Phonon_TTS_Attention, self);
        errdefer unloadBuffers(&buffers);

        var loader: zml.io.Loader = try .init(allocator, platform, .default);
        defer loader.deinit();

        try loader.load(io, Gradium_Phonon_TTS_Attention, self, &buffers, store, .{});
        try loader.await(io);

        return buffers;
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Gradium_Phonon_TTS_Attention)) void {
        const here = @typeName(@This()) ++ ".unloadBuffers";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        zml.Buffer.deinitAll(Gradium_Phonon_TTS_Attention, self);
    }
};

test Gradium_Phonon_TTS_Attention {
    const test_data = @import("transformer_test_data.zig");

    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    var rng: std.Random.DefaultPrng = .init(42);
    const random = rng.random();

    const layer: Gradium_Phonon_TTS_Attention = .{
        .qkv_proj = .init(.fromShape(.init(.{ .qkv = 1536, .d = 512 }, .bf16)), null, .d),
        .out_proj = .init(.fromShape(.init(.{ .dout = 512, .d = 512 }, .bf16)), null, .d),
    };
    const input_shape: zml.Shape = .init(.{ .b = 2, .s = 3, .d = 512 }, .f32);
    var input: [2][3][512]f32 = undefined;
    for (&input) |*batch| {
        for (batch) |*position| {
            for (position) |*value| value.* = random.float(f32) * 2 - 1;
        }
    }
    var qkv_weights: [1536][512]zml.floats.BFloat16 = undefined;
    for (&qkv_weights) |*row| {
        for (row) |*value| value.* = .fromF32(random.float(f32) * 2 - 1);
    }
    var out_weights: [512][512]zml.floats.BFloat16 = undefined;
    for (&out_weights) |*row| {
        for (row) |*value| value.* = .fromF32(random.float(f32) * 2 - 1);
    }

    var qkv_weight_buffer = try zml.Buffer.fromBytes(io, platform, layer.qkv_proj.weight.shape(), std.mem.asBytes(&qkv_weights));
    defer qkv_weight_buffer.deinit();
    var out_weight_buffer = try zml.Buffer.fromBytes(io, platform, layer.out_proj.weight.shape(), std.mem.asBytes(&out_weights));
    defer out_weight_buffer.deinit();
    const weight_buffers: zml.Bufferized(Gradium_Phonon_TTS_Attention) = .{
        .qkv_proj = .{ .weight = qkv_weight_buffer, .bias = null, .quantization = null },
        .out_proj = .{ .weight = out_weight_buffer, .bias = null, .quantization = null },
    };
    var input_buffer = try zml.Buffer.fromBytes(io, platform, input_shape, std.mem.asBytes(&input));
    defer input_buffer.deinit();

    var exe = try platform.compile(allocator, io, layer, .forward_uncached, .{zml.Tensor.fromShape(input_shape)}, .{});
    defer exe.deinit();
    var output = try exe.eval(allocator, io, .{ weight_buffers, input_buffer });
    defer output.deinit();

    const expected = test_data.attention_uncached;
    const expected_slice = zml.Slice.init(input_shape, std.mem.asBytes(&expected));
    // Large random projections can cancel near zero; allow f32 reduction-order error.
    try zml.testing.expectClose(io, expected_slice, output, .{
        .absolute_tolerance = 1e-3,
        .relative_tolerance = 1e-4,
        .minimum_close_fraction = 1,
    });

    for ([_]zml.DataType{ .f16, .bf16, .f64 }) |dtype| {
        var typed_exe = try platform.compile(allocator, io, layer, .forward_uncached, .{
            zml.Tensor.fromShape(input_shape.withDtype(dtype)),
        }, .{});
        defer typed_exe.deinit();
        try std.testing.expect(typed_exe.output_shapes[0].eqlWithTags(input_shape.withDtype(dtype)));
    }
}

test "Gradium_Phonon_TTS_Attention: cached" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();

    var rng: std.Random.DefaultPrng = .init(42);
    const random = rng.random();

    const layer: Gradium_Phonon_TTS_Attention = .{
        .qkv_proj = .init(.fromShape(.init(.{ .qkv = 1536, .d = 512 }, .bf16)), null, .d),
        .out_proj = .init(.fromShape(.init(.{ .dout = 512, .d = 512 }, .bf16)), null, .d),
    };
    const input_shape: zml.Shape = .init(.{ .b = 2, .s = 3, .d = 512 }, .f32);
    var input: [2][3][512]f32 = undefined;
    for (&input) |*batch| {
        for (batch) |*position| {
            for (position) |*value| value.* = random.float(f32) * 2 - 1;
        }
    }
    var qkv_weights: [1536][512]zml.floats.BFloat16 = undefined;
    for (&qkv_weights) |*row| {
        for (row) |*value| value.* = .fromF32(random.float(f32) * 2 - 1);
    }
    var out_weights: [512][512]zml.floats.BFloat16 = undefined;
    for (&out_weights) |*row| {
        for (row) |*value| value.* = .fromF32(random.float(f32) * 2 - 1);
    }

    var qkv_weight_buffer = try zml.Buffer.fromBytes(io, platform, layer.qkv_proj.weight.shape(), std.mem.asBytes(&qkv_weights));
    defer qkv_weight_buffer.deinit();
    var out_weight_buffer = try zml.Buffer.fromBytes(io, platform, layer.out_proj.weight.shape(), std.mem.asBytes(&out_weights));
    defer out_weight_buffer.deinit();
    const weight_buffers: zml.Bufferized(Gradium_Phonon_TTS_Attention) = .{
        .qkv_proj = .{ .weight = qkv_weight_buffer, .bias = null, .quantization = null },
        .out_proj = .{ .weight = out_weight_buffer, .bias = null, .quantization = null },
    };
    var input_buffer = try zml.Buffer.fromBytes(io, platform, input_shape, std.mem.asBytes(&input));
    defer input_buffer.deinit();

    var uncached_exe = try platform.compile(allocator, io, layer, .forward_uncached, .{zml.Tensor.fromShape(input_shape)}, .{});
    defer uncached_exe.deinit();
    var uncached_output = try uncached_exe.eval(allocator, io, .{ weight_buffers, input_buffer });
    defer uncached_output.deinit();

    // Leave unused capacity after the sequence to exercise masking of future slots.
    const initial_cache = KV_Cache.init(2, 5);
    var cache_buffers = try initial_cache.initBuffer(allocator, io, platform);
    defer KV_Cache.deinitBuffer(&cache_buffers);
    var cached_output: [2][3][512]f32 = undefined;

    // Prefill two positions, then decode the third using the returned history.
    inline for (.{ .{ .start = 0, .end = 2 }, .{ .start = 2, .end = 3 } }) |step| {
        const count = step.end - step.start;
        const chunk_shape: zml.Shape = .init(.{ .b = 2, .s = count, .d = 512 }, .f32);
        var chunk: [2][count][512]f32 = undefined;
        for (&chunk, input) |*batch, source| {
            @memcpy(batch, source[step.start..step.end]);
        }
        var chunk_buffer = try zml.Buffer.fromBytes(io, platform, chunk_shape, std.mem.asBytes(&chunk));
        defer chunk_buffer.deinit();
        const position: i32 = step.start;
        var position_buffer = try zml.Buffer.fromBytes(io, platform, .scalar(.i32), std.mem.asBytes(&position));
        defer position_buffer.deinit();

        var cached_exe = try platform.compile(allocator, io, layer, .forward_cached, .{
            zml.Tensor.fromShape(chunk_shape),
            zml.Tensor.fromShape(.scalar(.i32)),
            initial_cache,
            @as(usize, 2),
        }, .{});
        defer cached_exe.deinit();
        var runner = try cached_exe.runner(allocator);
        defer runner.deinit(allocator);
        var result: zml.Bufferized(CachedHidden) = undefined;
        runner.run(io, .{ weight_buffers, chunk_buffer, position_buffer, cache_buffers, @as(usize, 2) }, .{&result}, .{ .wait = true });
        defer result.hidden.deinit();
        KV_Cache.deinitBuffer(&cache_buffers);
        cache_buffers = result.cache;

        const values = try result.hidden.getValue([2][count][512]f32, io);
        for (&cached_output, values) |*batch, source| {
            @memcpy(batch[step.start..step.end], &source);
        }
    }

    const cached_slice = zml.Slice.init(input_shape, std.mem.asBytes(&cached_output));
    // Cached and full-sequence attention use different matmul/reduction shapes.
    try zml.testing.expectClose(io, uncached_output, cached_slice, .{
        .absolute_tolerance = 1e-3,
        .relative_tolerance = 1e-4,
        .minimum_close_fraction = 1,
    });

    for ([_]zml.DataType{ .f16, .bf16, .f32, .f64 }) |dtype| {
        const cache_shape: zml.Shape = .init(.{ .layer = 6, .b = 2, .k = 5, .h = 8, .hd = 64 }, dtype);
        // Check matching dtypes as well as separate K/V storage dtypes.
        for ([_]KV_Cache{
            .{ .k = .fromShape(cache_shape), .v = .fromShape(cache_shape) },
            .{ .k = .fromShape(cache_shape.withDtype(.f16)), .v = .fromShape(cache_shape.withDtype(.bf16)) },
        }) |cache| {
            var cached_exe = try platform.compile(allocator, io, layer, .forward_cached, .{
                zml.Tensor.fromShape(input_shape.withDtype(dtype)),
                zml.Tensor.fromShape(.init(.{}, .i32)),
                cache,
                @as(usize, 2),
            }, .{});
            defer cached_exe.deinit();
            try std.testing.expect(cached_exe.output_shapes[0].eqlWithTags(input_shape.withDtype(dtype)));
            try std.testing.expect(cached_exe.output_shapes[1].eqlWithTags(cache.k.shape()));
            try std.testing.expect(cached_exe.output_shapes[2].eqlWithTags(cache.v.shape()));
        }
    }
}

// --- Gradium_Phonon_TTS_MLP ---

/// Refines each position's features with a two-layer feed-forward network.
///
/// A bias-free linear layer expands 512 features to 2048, exact erf GELU acts
/// elementwise, and another bias-free layer returns to 512 features. Sequence
/// positions and batch rows are processed independently. The decoder layer
/// supplies normalized features and adds this module's output as a residual.
///
/// Input and output: floating-point `[..., .d=512]`, preserving all leading axes.
/// Parameters come from `flow_lm.transformer.layers.{index}`:
/// `linear1.weight` [2048, 512] and `linear2.weight` [512, 2048], stored in bf16
/// and converted to the input dtype.
///
/// Reference:
/// ```
/// mlp: GradiumPhononTTSMLP
/// ├── activation_fn: GELUActivation
/// ├── fc1: Linear
/// │   └── weight: shape=(2048, 512), dtype=torch.float32
/// │       ├── loaded: model.layers.1.mlp.fc1.weight
/// │       └── checkpoint: flow_lm.transformer.layers.1.linear1.weight
/// └── fc2: Linear
///     └── weight: shape=(512, 2048), dtype=torch.float32
///         ├── loaded: model.layers.1.mlp.fc2.weight
///         └── checkpoint: flow_lm.transformer.layers.1.linear2.weight
/// ```
const Gradium_Phonon_TTS_MLP = struct {
    fc1: zml.nn.Linear,
    fc2: zml.nn.Linear,

    pub fn init(store: zml.io.TensorStore.View) Gradium_Phonon_TTS_MLP {
        const here = @typeName(@This()) ++ ".init";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        return .{
            .fc1 = .init(store.createReplicatedTensor("linear1.weight", .{ .dout, .d }), null, .d),
            .fc2 = .init(store.createReplicatedTensor("linear2.weight", .{ .d, .dout }), null, .dout),
        };
    }

    // -- Description --

    /// Applies fc1 -> exact GELU -> fc2 independently at each position.
    ///
    /// `input`: floating-point `[..., 512]`, normalized features from the decoder layer's
    /// `Post_Attention_Layer_Norm`, after the attention residual addition.
    /// Usually `[B, S, 512]`, or `[B, 1, 512]` during cached decode. This method
    /// tags only the final axis `.d`, preserving any leading axes.
    ///
    /// Returns floating-point `[..., .d=512]`, the MLP residual contribution. The decoder
    /// layer adds it to the features from before pre-MLP normalization.
    ///
    /// Parameters and output use the input dtype. ZML's erf approximation is
    /// evaluated in f32 and converted back before the remaining GELU operations.
    ///
    /// Dimension tags:
    /// - `.d`: model dimension; 512 features per token. Contracted by fc1 and
    ///   restored by fc2 so the output can feed the next transformer operation.
    /// - `.dout`: fc1's output dimension; the expanded 2048 intermediate features
    ///   per token. GELU acts elementwise; fc2 contracts this axis back to `.d`.
    /// - Leading axes are preserved without mixing their entries. Conventionally
    ///   `.b` is batch (independent sequences) and `.s` is sequence (token positions).
    ///   Only the final axis is tagged here; untagged leading axes remain untagged.
    ///
    /// Shape: `[..., .d=512] -> [..., .dout=2048] -> [..., .d=512]`, preserving the input dtype.
    pub fn forward(self: Gradium_Phonon_TTS_MLP, input: zml.Tensor) zml.Tensor {
        const here = @typeName(@This()) ++ ".forward";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        std.debug.assert(input.dtype().isFloat());

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), "Gradium_Phonon_TTS_MLP", .{});
        defer compiler.popLocation();

        const x = input.withPartialTags(.{.d});

        compiler.pushLocationFmt(@src(), "layer[fc1]", .{});
        const hidden = self.fc1.forward(x, x.dtype());
        compiler.popLocation();

        compiler.pushLocationFmt(@src(), "layer[gelu]", .{});
        const activated = gelu_erf(hidden);
        compiler.popLocation();

        compiler.pushLocationFmt(@src(), "layer[fc2]", .{});
        const output = self.fc2.forward(activated, activated.dtype());
        compiler.popLocation();
        return output;
    }

    fn gelu_erf(x: zml.Tensor) zml.Tensor {
        // Tensor.erf returns f32; convert back before combining it with x.
        const erf_term = x.scale(1.0 / std.math.sqrt(2.0)).erf().convert(x.dtype());
        return x.scale(0.5).mul(erf_term.addConstant(1.0));
    }

    // -- Runtime --

    pub fn load(
        self: *const Gradium_Phonon_TTS_MLP,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
    ) !zml.Bufferized(Gradium_Phonon_TTS_MLP) {
        const here = @typeName(@This()) ++ ".load";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        var buffers = try zml.mem.bufferize(allocator, Gradium_Phonon_TTS_MLP, self);
        errdefer unloadBuffers(&buffers);

        var loader: zml.io.Loader = try .init(allocator, platform, .default);
        defer loader.deinit();

        try loader.load(io, Gradium_Phonon_TTS_MLP, self, &buffers, store, .{});
        try loader.await(io);

        return buffers;
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Gradium_Phonon_TTS_MLP)) void {
        const here = @typeName(@This()) ++ ".unloadBuffers";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        zml.nn.Linear.unloadBuffers(&self.fc1);
        zml.nn.Linear.unloadBuffers(&self.fc2);
    }
};

test Gradium_Phonon_TTS_MLP {
    const test_data = @import("transformer_test_data.zig");

    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    var rng: std.Random.DefaultPrng = .init(42);
    const random = rng.random();

    const layer: Gradium_Phonon_TTS_MLP = .{
        .fc1 = .init(.fromShape(.init(.{ .dout = 2048, .d = 512 }, .bf16)), null, .d),
        .fc2 = .init(.fromShape(.init(.{ .d = 512, .dout = 2048 }, .bf16)), null, .dout),
    };
    const input_shape: zml.Shape = .init(.{ .b = 2, .s = 2, .d = 512 }, .f32);
    var input: [2][2][512]f32 = undefined;
    for (&input) |*batch| {
        for (batch) |*position| {
            for (position) |*value| value.* = random.float(f32) * 2 - 1;
        }
    }
    var fc1_weights: [2048][512]zml.floats.BFloat16 = undefined;
    for (&fc1_weights) |*row| {
        for (row) |*value| value.* = .fromF32(random.float(f32) * 2 - 1);
    }
    var fc2_weights: [512][2048]zml.floats.BFloat16 = undefined;
    for (&fc2_weights) |*row| {
        for (row) |*value| value.* = .fromF32(random.float(f32) * 2 - 1);
    }

    var fc1_weight_buffer = try zml.Buffer.fromBytes(io, platform, layer.fc1.weight.shape(), std.mem.asBytes(&fc1_weights));
    defer fc1_weight_buffer.deinit();
    var fc2_weight_buffer = try zml.Buffer.fromBytes(io, platform, layer.fc2.weight.shape(), std.mem.asBytes(&fc2_weights));
    defer fc2_weight_buffer.deinit();
    const weight_buffers: zml.Bufferized(Gradium_Phonon_TTS_MLP) = .{
        .fc1 = .{ .weight = fc1_weight_buffer, .bias = null, .quantization = null },
        .fc2 = .{ .weight = fc2_weight_buffer, .bias = null, .quantization = null },
    };
    var input_buffer = try zml.Buffer.fromBytes(io, platform, input_shape, std.mem.asBytes(&input));
    defer input_buffer.deinit();

    var exe = try platform.compile(allocator, io, layer, .forward, .{zml.Tensor.fromShape(input_shape)}, .{});
    defer exe.deinit();
    var output = try exe.eval(allocator, io, .{ weight_buffers, input_buffer });
    defer output.deinit();

    const expected = test_data.mlp;
    const expected_slice = zml.Slice.init(.init(.{ .b = 2, .s = 2, .d = 512 }, .f32), std.mem.asBytes(&expected));
    // Large random projections can cancel near zero; allow f32 reduction-order error.
    try zml.testing.expectClose(io, expected_slice, output, .{
        .absolute_tolerance = 1e-3,
        .relative_tolerance = 1e-4,
        .minimum_close_fraction = 1,
    });

    for ([_]zml.DataType{ .f16, .bf16, .f64 }) |dtype| {
        var typed_exe = try platform.compile(allocator, io, layer, .forward, .{
            zml.Tensor.fromShape(input_shape.withDtype(dtype)),
        }, .{});
        defer typed_exe.deinit();
        try std.testing.expect(typed_exe.output_shapes[0].eqlWithTags(.init(.{ .b = 2, .s = 2, .d = 512 }, dtype)));
    }
}

// --- Input_Layer_Norm ---

/// Normalizes features entering a decoder layer, before its attention branch.
///
/// Computes mean and population variance across each position's 512 features,
/// normalizes with epsilon 1e-5, then applies a learned per-feature scale and bias.
/// Batch rows and sequence positions retain separate statistics.
///
/// Input and output: floating-point `[..., .d=512]`, preserving all leading axes.
/// Parameters come from `flow_lm.transformer.layers.{index}.norm1`:
/// bf16 `weight` [512] and `bias` [512], converted to the input dtype. Statistics are computed
/// from the current input on every call.
///
/// Reference:
/// ```
/// input_layernorm: LayerNorm
/// ├── weight: shape=(512,), dtype=torch.float32
/// │   ├── loaded: model.layers.1.input_layernorm.weight
/// │   └── checkpoint: flow_lm.transformer.layers.1.norm1.weight
/// └── bias: shape=(512,), dtype=torch.float32
///     ├── loaded: model.layers.1.input_layernorm.bias
///     └── checkpoint: flow_lm.transformer.layers.1.norm1.bias
/// ```
const Input_Layer_Norm = struct {
    weight: zml.Tensor,
    bias: zml.Tensor,
    eps: f32,

    pub fn init(store: zml.io.TensorStore.View) Input_Layer_Norm {
        const here = @typeName(@This()) ++ ".init";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        return .{
            .weight = store.createReplicatedTensor("norm1.weight", .{.d}),
            .bias = store.createReplicatedTensor("norm1.bias", .{.d}),
            .eps = 1e-5, // TODO: shouldnt this be coming from json then?
        };
    }

    // -- Description --

    /// Normalizes each position before the decoder layer's attention branch.
    ///
    /// `input`: floating-point `[..., 512]`, the decoder layer's incoming residual features:
    /// prepared input projections for layer 0, or the preceding layer's output.
    /// Usually `[B, S, 512]`, or `[B, 1, 512]` during cached decode. This method
    /// tags only the final axis `.d`, preserving any leading axes.
    ///
    /// Returns floating-point `[..., .d=512]`, normalized and learned scale/shift features
    /// passed to self-attention. The parent retains `input` for the residual addition.
    ///
    /// Parameters and output use the input dtype. ZML's LayerNorm computes
    /// normalization statistics internally in f32 before converting back.
    ///
    /// Dimension tags:
    /// - `.d`: model dimension; the final axis of 512 features per token. Mean and
    ///   variance are computed across this axis, then learned weight and bias
    ///   (each shaped `[.d]`) scale and shift each normalized feature.
    /// - Leading axes are preserved. Conventionally `.b` is batch (independent
    ///   sequences) and `.s` is sequence (token positions). Statistics are separate
    ///   for every token in every sequence; neither axis is reduced or mixed.
    ///   Only the final axis is tagged here; untagged leading axes remain untagged.
    ///
    /// Shape: `[..., .d=512] -> [..., .d=512]`, preserving the floating-point dtype.
    pub fn forward(self: Input_Layer_Norm, input: zml.Tensor) zml.Tensor {
        const here = @typeName(@This()) ++ ".forward";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        std.debug.assert(input.dtype().isFloat());

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), "Input_Layer_Norm", .{});
        defer compiler.popLocation();

        const x = input.withPartialTags(.{.d});
        const weight = self.weight.convert(x.dtype());
        const bias = self.bias.convert(x.dtype());
        const norm: zml.nn.LayerNorm = .{ .weight = weight, .bias = bias, .eps = self.eps };

        compiler.pushLocationFmt(@src(), "layer[norm]", .{});
        const output = norm.forward(x);
        compiler.popLocation();
        return output;
    }

    // -- Runtime --

    pub fn load(
        self: *const Input_Layer_Norm,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
    ) !zml.Bufferized(Input_Layer_Norm) {
        const here = @typeName(@This()) ++ ".load";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        var buffers = try zml.mem.bufferize(allocator, Input_Layer_Norm, self);
        errdefer unloadBuffers(&buffers);

        var loader: zml.io.Loader = try .init(allocator, platform, .default);
        defer loader.deinit();

        try loader.load(io, Input_Layer_Norm, self, &buffers, store, .{});
        try loader.await(io);

        return buffers;
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Input_Layer_Norm)) void {
        const here = @typeName(@This()) ++ ".unloadBuffers";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        zml.Buffer.deinitAll(Input_Layer_Norm, self);
    }
};

test Input_Layer_Norm {
    const test_data = @import("transformer_test_data.zig");

    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    var rng: std.Random.DefaultPrng = .init(42);
    const random = rng.random();

    const layer: Input_Layer_Norm = .{
        .weight = .fromShape(.init(.{ .d = 512 }, .bf16)),
        .bias = .fromShape(.init(.{ .d = 512 }, .bf16)),
        .eps = 1e-5,
    };
    const input_shape: zml.Shape = .init(.{ .b = 2, .s = 2, .d = 512 }, .f32);
    var input: [2][2][512]f32 = undefined;
    for (&input) |*batch| {
        for (batch) |*position| {
            for (position) |*value| value.* = random.float(f32) * 2 - 1;
        }
    }
    var weight_values: [512]zml.floats.BFloat16 = undefined;
    for (&weight_values) |*value| value.* = .fromF32(random.float(f32) * 2 - 1);
    var bias_values: [512]zml.floats.BFloat16 = undefined;
    for (&bias_values) |*value| value.* = .fromF32(random.float(f32) * 2 - 1);

    var weight = try zml.Buffer.fromBytes(io, platform, layer.weight.shape(), std.mem.asBytes(&weight_values));
    defer weight.deinit();
    var bias = try zml.Buffer.fromBytes(io, platform, layer.bias.shape(), std.mem.asBytes(&bias_values));
    defer bias.deinit();
    const weight_buffers: zml.Bufferized(Input_Layer_Norm) = .{ .weight = weight, .bias = bias };
    var input_buffer = try zml.Buffer.fromBytes(io, platform, input_shape, std.mem.asBytes(&input));
    defer input_buffer.deinit();

    var exe = try platform.compile(allocator, io, layer, .forward, .{zml.Tensor.fromShape(input_shape)}, .{});
    defer exe.deinit();
    var output = try exe.eval(allocator, io, .{ weight_buffers, input_buffer });
    defer output.deinit();

    const expected = test_data.input_layer_norm;
    const expected_slice = zml.Slice.init(.init(.{ .b = 2, .s = 2, .d = 512 }, .f32), std.mem.asBytes(&expected));
    try zml.testing.expectClose(io, expected_slice, output, .{
        .absolute_tolerance = 1e-4,
        .relative_tolerance = 1e-4,
        .minimum_close_fraction = 1,
    });

    for ([_]zml.DataType{ .f16, .bf16, .f64 }) |dtype| {
        var typed_exe = try platform.compile(allocator, io, layer, .forward, .{
            zml.Tensor.fromShape(input_shape.withDtype(dtype)),
        }, .{});
        defer typed_exe.deinit();
        try std.testing.expect(typed_exe.output_shapes[0].eqlWithTags(.init(.{ .b = 2, .s = 2, .d = 512 }, dtype)));
    }
}

// --- Post_Attention_Layer_Norm ---

/// Normalizes features after the attention residual addition, before the MLP branch.
///
/// Computes mean and population variance across each position's 512 features,
/// normalizes with epsilon 1e-5, then applies a learned per-feature scale and bias.
/// Batch rows and sequence positions retain separate statistics.
///
/// Input and output: floating-point `[..., .d=512]`, preserving all leading axes.
/// Parameters come from `flow_lm.transformer.layers.{index}.norm2`:
/// bf16 `weight` [512] and `bias` [512], converted to the input dtype. Statistics are computed
/// from the current input on every call.
///
/// Reference:
/// ```
/// post_attention_layernorm: LayerNorm
/// ├── weight: shape=(512,), dtype=torch.float32
/// │   ├── loaded: model.layers.1.post_attention_layernorm.weight
/// │   └── checkpoint: flow_lm.transformer.layers.1.norm2.weight
/// └── bias: shape=(512,), dtype=torch.float32
///     ├── loaded: model.layers.1.post_attention_layernorm.bias
///     └── checkpoint: flow_lm.transformer.layers.1.norm2.bias
/// ```
const Post_Attention_Layer_Norm = struct {
    weight: zml.Tensor,
    bias: zml.Tensor,
    eps: f32,

    pub fn init(store: zml.io.TensorStore.View) Post_Attention_Layer_Norm {
        const here = @typeName(@This()) ++ ".init";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        return .{
            .weight = store.createReplicatedTensor("norm2.weight", .{.d}),
            .bias = store.createReplicatedTensor("norm2.bias", .{.d}),
            .eps = 1e-5, // TODO: shouldnt this be coming from json then?
        };
    }

    // -- Description --

    /// Normalizes each position before the decoder layer's MLP branch.
    ///
    /// `input`: floating-point `[..., 512]`, the sum of the decoder layer's incoming features
    /// and its attention output. Usually `[B, S, 512]`, or `[B, 1, 512]` during
    /// cached decode. This method tags only the final axis `.d`, preserving any
    /// leading axes.
    ///
    /// Returns floating-point `[..., .d=512]`, normalized and learned scale/shift features
    /// passed to the MLP. The parent retains `input` for the next residual addition.
    ///
    /// Parameters and output use the input dtype. ZML's LayerNorm computes
    /// normalization statistics internally in f32 before converting back.
    ///
    /// Dimension tags:
    /// - `.d`: model dimension; the final axis of 512 features per token. Mean and
    ///   variance are computed across this axis, then learned weight and bias
    ///   (each shaped `[.d]`) scale and shift each normalized feature.
    /// - Leading axes are preserved. Conventionally `.b` is batch (independent
    ///   sequences) and `.s` is sequence (token positions). Statistics are separate
    ///   for every token in every sequence; neither axis is reduced or mixed.
    ///   Only the final axis is tagged here; untagged leading axes remain untagged.
    ///
    /// Shape: `[..., .d=512] -> [..., .d=512]`, preserving the floating-point dtype.
    pub fn forward(self: Post_Attention_Layer_Norm, input: zml.Tensor) zml.Tensor {
        const here = @typeName(@This()) ++ ".forward";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        std.debug.assert(input.dtype().isFloat());

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), "Post_Attention_Layer_Norm", .{});
        defer compiler.popLocation();

        const x = input.withPartialTags(.{.d});
        const weight = self.weight.convert(x.dtype());
        const bias = self.bias.convert(x.dtype());
        const norm: zml.nn.LayerNorm = .{ .weight = weight, .bias = bias, .eps = self.eps };

        compiler.pushLocationFmt(@src(), "layer[norm]", .{});
        const output = norm.forward(x);
        compiler.popLocation();
        return output;
    }

    // -- Runtime --

    pub fn load(
        self: *const Post_Attention_Layer_Norm,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
    ) !zml.Bufferized(Post_Attention_Layer_Norm) {
        const here = @typeName(@This()) ++ ".load";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        var buffers = try zml.mem.bufferize(allocator, Post_Attention_Layer_Norm, self);
        errdefer unloadBuffers(&buffers);

        var loader: zml.io.Loader = try .init(allocator, platform, .default);
        defer loader.deinit();

        try loader.load(io, Post_Attention_Layer_Norm, self, &buffers, store, .{});
        try loader.await(io);

        return buffers;
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Post_Attention_Layer_Norm)) void {
        const here = @typeName(@This()) ++ ".unloadBuffers";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        zml.Buffer.deinitAll(Post_Attention_Layer_Norm, self);
    }
};

test Post_Attention_Layer_Norm {
    const test_data = @import("transformer_test_data.zig");

    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    var rng: std.Random.DefaultPrng = .init(42);
    const random = rng.random();

    const layer: Post_Attention_Layer_Norm = .{
        .weight = .fromShape(.init(.{ .d = 512 }, .bf16)),
        .bias = .fromShape(.init(.{ .d = 512 }, .bf16)),
        .eps = 1e-5,
    };
    const input_shape: zml.Shape = .init(.{ .b = 2, .s = 2, .d = 512 }, .f32);
    var input: [2][2][512]f32 = undefined;
    for (&input) |*batch| {
        for (batch) |*position| {
            for (position) |*value| value.* = random.float(f32) * 2 - 1;
        }
    }
    var weight_values: [512]zml.floats.BFloat16 = undefined;
    for (&weight_values) |*value| value.* = .fromF32(random.float(f32) * 2 - 1);
    var bias_values: [512]zml.floats.BFloat16 = undefined;
    for (&bias_values) |*value| value.* = .fromF32(random.float(f32) * 2 - 1);

    var weight = try zml.Buffer.fromBytes(io, platform, layer.weight.shape(), std.mem.asBytes(&weight_values));
    defer weight.deinit();
    var bias = try zml.Buffer.fromBytes(io, platform, layer.bias.shape(), std.mem.asBytes(&bias_values));
    defer bias.deinit();
    const weight_buffers: zml.Bufferized(Post_Attention_Layer_Norm) = .{ .weight = weight, .bias = bias };
    var input_buffer = try zml.Buffer.fromBytes(io, platform, input_shape, std.mem.asBytes(&input));
    defer input_buffer.deinit();

    var exe = try platform.compile(allocator, io, layer, .forward, .{zml.Tensor.fromShape(input_shape)}, .{});
    defer exe.deinit();
    var output = try exe.eval(allocator, io, .{ weight_buffers, input_buffer });
    defer output.deinit();

    const expected = test_data.post_attention_layer_norm;
    const expected_slice = zml.Slice.init(.init(.{ .b = 2, .s = 2, .d = 512 }, .f32), std.mem.asBytes(&expected));
    try zml.testing.expectClose(io, expected_slice, output, .{
        .absolute_tolerance = 1e-4,
        .relative_tolerance = 1e-4,
        .minimum_close_fraction = 1,
    });

    for ([_]zml.DataType{ .f16, .bf16, .f64 }) |dtype| {
        var typed_exe = try platform.compile(allocator, io, layer, .forward, .{
            zml.Tensor.fromShape(input_shape.withDtype(dtype)),
        }, .{});
        defer typed_exe.deinit();
        try std.testing.expect(typed_exe.output_shapes[0].eqlWithTags(.init(.{ .b = 2, .s = 2, .d = 512 }, dtype)));
    }
}
