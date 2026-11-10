const std = @import("std");
const zml = @import("zml");

const convolution = @import("convolution.zig");
const transformer = @import("transformer.zig");
const seanet = @import("seanet.zig");

const test_data = @import("model_test_data.zig");

test {
    std.testing.refAllDecls(@This());
}

// --- PocketTTSAudioTokenizerModel ---

/// Decodes denormalized 32-coordinate Phonon latents into 24 kHz mono audio.
/// The uncached path processes a whole sequence; the cached path carries both
/// convolution overlap and transformer history between contiguous chunks.
///
/// Reference:
/// ```
/// PocketTTSAudioTokenizerModel
/// ├── decoder: PocketTTSAudioTokenizerDecoderModel
/// │   ├── upsample: PocketTTSAudioTokenizerConvTranspose1d
/// │   │   └── conv: ConvTranspose1d
/// │   │       └── weight: shape=(512, 1, 32), dtype=torch.float32
/// │   │           └── loaded: decoder.upsample.conv.weight
/// │   ├── decoder_transformer: PocketTTSAudioTokenizerTransformerModel
/// │   │   ├── layers: ModuleList
/// │   │   │   └── 0..1: PocketTTSAudioTokenizerTransformerLayer
/// │   │   │       ├── self_attn: PocketTTSAudioTokenizerAttention
/// │   │   │       │   ├── q_proj: Linear
/// │   │   │       │   │   └── weight: shape=(512, 512), dtype=torch.float32
/// │   │   │       │   │       └── loaded: decoder.decoder_transformer.layers.1.self_attn.q_proj.weight
/// │   │   │       │   ├── k_proj: Linear
/// │   │   │       │   │   └── weight: shape=(512, 512), dtype=torch.float32
/// │   │   │       │   │       └── loaded: decoder.decoder_transformer.layers.1.self_attn.k_proj.weight
/// │   │   │       │   ├── v_proj: Linear
/// │   │   │       │   │   └── weight: shape=(512, 512), dtype=torch.float32
/// │   │   │       │   │       └── loaded: decoder.decoder_transformer.layers.1.self_attn.v_proj.weight
/// │   │   │       │   └── o_proj: Linear
/// │   │   │       │       └── weight: shape=(512, 512), dtype=torch.float32
/// │   │   │       │           └── loaded: decoder.decoder_transformer.layers.1.self_attn.o_proj.weight
/// │   │   │       ├── mlp: PocketTTSAudioTokenizerMLP
/// │   │   │       │   ├── activation_fn: GELUActivation
/// │   │   │       │   ├── fc1: Linear
/// │   │   │       │   │   └── weight: shape=(2048, 512), dtype=torch.float32
/// │   │   │       │   │       └── loaded: decoder.decoder_transformer.layers.1.mlp.fc1.weight
/// │   │   │       │   └── fc2: Linear
/// │   │   │       │       └── weight: shape=(512, 2048), dtype=torch.float32
/// │   │   │       │           └── loaded: decoder.decoder_transformer.layers.1.mlp.fc2.weight
/// │   │   │       ├── input_layernorm: LayerNorm
/// │   │   │       │   ├── weight: shape=(512,), dtype=torch.float32
/// │   │   │       │   │   └── loaded: decoder.decoder_transformer.layers.1.input_layernorm.weight
/// │   │   │       │   └── bias: shape=(512,), dtype=torch.float32
/// │   │   │       │       └── loaded: decoder.decoder_transformer.layers.1.input_layernorm.bias
/// │   │   │       ├── post_attention_layernorm: LayerNorm
/// │   │   │       │   ├── weight: shape=(512,), dtype=torch.float32
/// │   │   │       │   │   └── loaded: decoder.decoder_transformer.layers.1.post_attention_layernorm.weight
/// │   │   │       │   └── bias: shape=(512,), dtype=torch.float32
/// │   │   │       │       └── loaded: decoder.decoder_transformer.layers.1.post_attention_layernorm.bias
/// │   │   │       ├── self_attn_layer_scale: PocketTTSAudioTokenizerLayerScale
/// │   │   │       │   └── scale: shape=(512,), dtype=torch.float32
/// │   │   │       │       └── loaded: decoder.decoder_transformer.layers.1.self_attn_layer_scale.scale
/// │   │   │       └── mlp_layer_scale: PocketTTSAudioTokenizerLayerScale
/// │   │   │           └── scale: shape=(512,), dtype=torch.float32
/// │   │   │               └── loaded: decoder.decoder_transformer.layers.1.mlp_layer_scale.scale
/// │   │   └── rotary_emb: PocketTTSAudioTokenizerRotaryEmbedding
/// │   └── decoder_seanet: PocketTTSAudioTokenizerDecoder
/// │       └── layers: ModuleList
/// │           ├── 0: PocketTTSAudioTokenizerConv1d
/// │           │   └── conv: Conv1d
/// │           │       ├── weight: shape=(512, 512, 7), dtype=torch.float32
/// │           │       │   └── loaded: decoder.decoder_seanet.layers.0.conv.weight
/// │           │       └── bias: shape=(512,), dtype=torch.float32
/// │           │           └── loaded: decoder.decoder_seanet.layers.0.conv.bias
/// │           ├── 1: ELU
/// │           ├── 2: PocketTTSAudioTokenizerConvTranspose1d
/// │           │   └── conv: ConvTranspose1d
/// │           │       ├── weight: shape=(512, 256, 12), dtype=torch.float32
/// │           │       │   └── loaded: decoder.decoder_seanet.layers.2.conv.weight
/// │           │       └── bias: shape=(256,), dtype=torch.float32
/// │           │           └── loaded: decoder.decoder_seanet.layers.2.conv.bias
/// │           ├── 3: PocketTTSAudioTokenizerResnetBlock
/// │           │   ├── block: ModuleList
/// │           │   │   ├── 0: ELU
/// │           │   │   ├── 1: PocketTTSAudioTokenizerConv1d
/// │           │   │   │   └── conv: Conv1d
/// │           │   │   │       ├── weight: shape=(128, 256, 3), dtype=torch.float32
/// │           │   │   │       │   └── loaded: decoder.decoder_seanet.layers.3.block.1.conv.weight
/// │           │   │   │       └── bias: shape=(128,), dtype=torch.float32
/// │           │   │   │           └── loaded: decoder.decoder_seanet.layers.3.block.1.conv.bias
/// │           │   │   ├── 2: ELU
/// │           │   │   └── 3: PocketTTSAudioTokenizerConv1d
/// │           │   │       └── conv: Conv1d
/// │           │   │           ├── weight: shape=(256, 128, 1), dtype=torch.float32
/// │           │   │           │   └── loaded: decoder.decoder_seanet.layers.3.block.3.conv.weight
/// │           │   │           └── bias: shape=(256,), dtype=torch.float32
/// │           │   │               └── loaded: decoder.decoder_seanet.layers.3.block.3.conv.bias
/// │           │   └── shortcut: Identity
/// │           ├── 4: ELU
/// │           ├── 5: PocketTTSAudioTokenizerConvTranspose1d
/// │           │   └── conv: ConvTranspose1d
/// │           │       ├── weight: shape=(256, 128, 10), dtype=torch.float32
/// │           │       │   └── loaded: decoder.decoder_seanet.layers.5.conv.weight
/// │           │       └── bias: shape=(128,), dtype=torch.float32
/// │           │           └── loaded: decoder.decoder_seanet.layers.5.conv.bias
/// │           ├── 6: PocketTTSAudioTokenizerResnetBlock
/// │           │   ├── block: ModuleList
/// │           │   │   ├── 0: ELU
/// │           │   │   ├── 1: PocketTTSAudioTokenizerConv1d
/// │           │   │   │   └── conv: Conv1d
/// │           │   │   │       ├── weight: shape=(64, 128, 3), dtype=torch.float32
/// │           │   │   │       │   └── loaded: decoder.decoder_seanet.layers.6.block.1.conv.weight
/// │           │   │   │       └── bias: shape=(64,), dtype=torch.float32
/// │           │   │   │           └── loaded: decoder.decoder_seanet.layers.6.block.1.conv.bias
/// │           │   │   ├── 2: ELU
/// │           │   │   └── 3: PocketTTSAudioTokenizerConv1d
/// │           │   │       └── conv: Conv1d
/// │           │   │           ├── weight: shape=(128, 64, 1), dtype=torch.float32
/// │           │   │           │   └── loaded: decoder.decoder_seanet.layers.6.block.3.conv.weight
/// │           │   │           └── bias: shape=(128,), dtype=torch.float32
/// │           │   │               └── loaded: decoder.decoder_seanet.layers.6.block.3.conv.bias
/// │           │   └── shortcut: Identity
/// │           ├── 7: ELU
/// │           ├── 8: PocketTTSAudioTokenizerConvTranspose1d
/// │           │   └── conv: ConvTranspose1d
/// │           │       ├── weight: shape=(128, 64, 8), dtype=torch.float32
/// │           │       │   └── loaded: decoder.decoder_seanet.layers.8.conv.weight
/// │           │       └── bias: shape=(64,), dtype=torch.float32
/// │           │           └── loaded: decoder.decoder_seanet.layers.8.conv.bias
/// │           ├── 9: PocketTTSAudioTokenizerResnetBlock
/// │           │   ├── block: ModuleList
/// │           │   │   ├── 0: ELU
/// │           │   │   ├── 1: PocketTTSAudioTokenizerConv1d
/// │           │   │   │   └── conv: Conv1d
/// │           │   │   │       ├── weight: shape=(32, 64, 3), dtype=torch.float32
/// │           │   │   │       │   └── loaded: decoder.decoder_seanet.layers.9.block.1.conv.weight
/// │           │   │   │       └── bias: shape=(32,), dtype=torch.float32
/// │           │   │   │           └── loaded: decoder.decoder_seanet.layers.9.block.1.conv.bias
/// │           │   │   ├── 2: ELU
/// │           │   │   └── 3: PocketTTSAudioTokenizerConv1d
/// │           │   │       └── conv: Conv1d
/// │           │   │           ├── weight: shape=(64, 32, 1), dtype=torch.float32
/// │           │   │           │   └── loaded: decoder.decoder_seanet.layers.9.block.3.conv.weight
/// │           │   │           └── bias: shape=(64,), dtype=torch.float32
/// │           │   │               └── loaded: decoder.decoder_seanet.layers.9.block.3.conv.bias
/// │           │   └── shortcut: Identity
/// │           ├── 10: ELU
/// │           └── 11: PocketTTSAudioTokenizerConv1d
/// │               └── conv: Conv1d
/// │                   ├── weight: shape=(1, 64, 3), dtype=torch.float32
/// │                   │   └── loaded: decoder.decoder_seanet.layers.11.conv.weight
/// │                   └── bias: shape=(1,), dtype=torch.float32
/// │                       └── loaded: decoder.decoder_seanet.layers.11.conv.bias
/// └── latent_proj: Conv1d
///     └── weight: shape=(512, 32, 1), dtype=torch.float32
///         └── loaded: latent_proj.weight
/// ```
pub const Pocket_TTS_Audio_Tokenizer_Model = struct {
    latent_proj: Latent_Projection,
    decoder: Decoder,

    /// Expects the root gradium/phonon-mimi checkpoint view.
    pub fn init(store: zml.io.TensorStore.View) Pocket_TTS_Audio_Tokenizer_Model {
        const here = @typeName(@This()) ++ ".init";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        return .{ .latent_proj = .init(store.withPrefix("latent_proj")), .decoder = .init(store.withPrefix("decoder")) };
    }

    /// `input`: floating [B,S,32], denormalized Phonon latents in codec space,
    /// i.e. normalized*flow_lm.emb_std+flow_lm.emb_mean. Returns floating
    /// [B,1,S*1920] mono samples at 24 kHz, preserving input.dtype().
    /// Tags .b independent sequences, .s latent/sample time, .ld latent coordinates,
    /// .d output channels. Each input frame yields exactly 80 ms of audio.
    pub fn forward_uncached(self: Pocket_TTS_Audio_Tokenizer_Model, input: zml.Tensor) zml.Tensor {
        const here = @typeName(@This()) ++ ".forward_uncached";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        return self.decoder.forward_uncached(self.latent_proj.forward(input));
    }

    /// `input`: floating [B,new_frames,32], only the new denormalized latents;
    /// `position`: scalar i32 number of upsampled positions already decoded
    /// (16 per previous latent). `cache`: zero-initialized/previously returned
    /// convolution and K/V history. Requires position+16*new_frames <= capacity.
    /// Returns [B,1,new_frames*1920] finalized audio and updated history.
    /// Each tensor retains its input dtype. Tags as above;
    /// cache semantics and tags are documented on Decoder_Cache.
    pub fn forward_cached(self: Pocket_TTS_Audio_Tokenizer_Model, input: zml.Tensor, position: zml.Tensor, cache: Decoder_Cache) Cached_Audio {
        const here = @typeName(@This()) ++ ".forward_cached";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        return self.decoder.forward_cached(self.latent_proj.forward(input), position, cache);
    }

    pub fn bufferize(self: Pocket_TTS_Audio_Tokenizer_Model, allocator: std.mem.Allocator) !zml.Bufferized(Pocket_TTS_Audio_Tokenizer_Model) {
        const here = @typeName(@This()) ++ ".bufferize";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        return .{ .latent_proj = try zml.mem.bufferize(allocator, Latent_Projection, &self.latent_proj), .decoder = try self.decoder.bufferize(allocator) };
    }

    pub fn load(
        self: *const Pocket_TTS_Audio_Tokenizer_Model,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
    ) !zml.Bufferized(Pocket_TTS_Audio_Tokenizer_Model) {
        const here = @typeName(@This()) ++ ".load";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        var buffers = try self.bufferize(allocator);
        errdefer unloadBuffers(&buffers);
        var loader: zml.io.Loader = try .init(allocator, platform, .default);
        defer loader.deinit();
        try loader.load(io, Pocket_TTS_Audio_Tokenizer_Model, self, &buffers, store, .{});
        try loader.await(io);
        return buffers;
    }

    pub fn unloadBuffers(buffers: *zml.Bufferized(Pocket_TTS_Audio_Tokenizer_Model)) void {
        const here = @typeName(@This()) ++ ".unloadBuffers";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        zml.Buffer.deinitAll(Pocket_TTS_Audio_Tokenizer_Model, buffers);
    }
};

/// Complete streaming decoder history: depthwise upsample's unfinished tail
/// [.b,.d=512,.s=16], transformer K/V [.layer=2,.b,.k=C,.h=8,.hd=64], and SEANet
/// input/overlap histories. All start at zero and belong to one sequence batch.
pub const Decoder_Cache = struct {
    upsample: zml.Tensor,
    transformer: transformer.KV_Cache,
    seanet: seanet.Cache,

    pub fn initBuffers(self: Decoder_Cache, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform) !zml.Bufferized(Decoder_Cache) {
        var buffers: zml.Bufferized(Decoder_Cache) = undefined;
        const tensors = try zml.meta.collectValues(zml.Tensor, allocator, &self);
        defer allocator.free(tensors);
        const targets = try zml.meta.collectMutPtrs(zml.Buffer, allocator, &buffers);
        defer allocator.free(targets);
        var loaded: usize = 0;
        errdefer for (targets[0..loaded]) |buffer| buffer.deinit();
        for (tensors, targets) |tensor, buffer| {
            const zeros = try zml.Slice.alloc(allocator, tensor.shape());
            defer zeros.free(allocator);
            @memset(zeros.data(), 0);
            buffer.* = try .fromSlice(io, platform, zeros);
            loaded += 1;
        }
        return buffers;
    }
};

pub const Cached_Audio = struct { audio: zml.Tensor, cache: Decoder_Cache };

/// Temporal upsampling, two-layer transformer, and SEANet waveform synthesis.
/// This is the reference's PocketTTSAudioTokenizerDecoderModel; latent projection
/// belongs to its parent. Targets the decoder-only gradium/phonon-mimi config.
pub const Decoder = struct {
    upsample: convolution.ConvTranspose1d,
    decoder_transformer: transformer.Transformer,
    decoder_seanet: seanet.SEANet,

    pub fn init(store: zml.io.TensorStore.View) Decoder {
        const here = @typeName(@This()) ++ ".init";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        return .{
            .upsample = .init(store.withPrefix("upsample"), 16, 512),
            .decoder_transformer = .init(store.withPrefix("decoder_transformer")),
            .decoder_seanet = .init(store.withPrefix("decoder_seanet")),
        };
    }

    pub fn bufferize(self: Decoder, allocator: std.mem.Allocator) !zml.Bufferized(Decoder) {
        const here = @typeName(@This()) ++ ".bufferize";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        return .{
            .upsample = try zml.mem.bufferize(allocator, convolution.ConvTranspose1d, &self.upsample),
            .decoder_transformer = try self.decoder_transformer.bufferize(allocator),
            .decoder_seanet = try self.decoder_seanet.bufferize(allocator),
        };
    }

    /// `input`: floating [B,512,S], Latent_Projection's channels-first output.
    /// Returns [B,1,S*1920] mono audio at 24 kHz in input.dtype(). Tags .b batch,
    /// .d channels, .s time; transformer temporarily uses [.b,.s,.d] at 200 Hz.
    pub fn forward_uncached(self: Decoder, input: zml.Tensor) zml.Tensor {
        const here = @typeName(@This()) ++ ".forward_uncached";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        const upsampled = self.upsample.forward_uncached(input);
        const transformer_input = upsampled.transpose(.{ .b, .s, .d });
        const contextualized = self.decoder_transformer.forward_uncached(transformer_input);
        const waveform_features = contextualized.transpose(.{ .b, .d, .s });
        return self.decoder_seanet.forward_uncached(waveform_features);
    }

    /// `input`: floating [B,512,new_frames], projected latents for this call.
    /// `position`: scalar i32 first new transformer index, measured at 200 Hz.
    /// `cache`: convolution/attention history from init_cache, zero-filled once
    /// then carried between calls. Returns new_frames*1920 samples and updated
    /// histories preserving their storage dtypes. Tags are listed on Decoder_Cache.
    pub fn forward_cached(self: Decoder, input: zml.Tensor, position: zml.Tensor, cache: Decoder_Cache) Cached_Audio {
        const here = @typeName(@This()) ++ ".forward_cached";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        const upsampled = self.upsample.forward_cached(input, cache.upsample);
        const transformer_input = upsampled.hidden.transpose(.{ .b, .s, .d });
        const contextualized = self.decoder_transformer.forward_cached(transformer_input, position, cache.transformer);
        const waveform_features = contextualized.hidden.transpose(.{ .b, .d, .s });
        const decoded = self.decoder_seanet.forward_cached(waveform_features, cache.seanet);
        return .{
            .audio = decoded.hidden,
            .cache = .{
                .upsample = upsampled.cache,
                .transformer = contextualized.cache,
                .seanet = decoded.cache,
            },
        };
    }

    pub fn init_cache(self: Decoder, batch: i64, capacity: i64, dtype: zml.DataType) Decoder_Cache {
        const here = @typeName(@This()) ++ ".init_cache";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        return .{
            .upsample = .fromShape(self.upsample.cache_shape(batch, dtype)),
            .transformer = .init(batch, capacity, dtype),
            .seanet = self.decoder_seanet.init_cache(batch, dtype),
        };
    }
};

// --- Latent_Projection ---

/// Projects each denormalized audio latent from 32 to 512 features.
///
/// This is the decoder's bias-free Conv1d(32, 512, kernel_size=1). A single
/// kernel tap mixes latent coordinates independently at each frame, so it is
/// implemented as a matrix multiplication followed by an axis transpose.
/// The f32 checkpoint weight [512, 32, 1] comes from `latent_proj.weight`.
/// Input and output preserve the input's floating-point dtype.
///
/// Reference:
/// ```
/// latent_proj: Conv1d
/// └── weight: shape=(512, 32, 1), dtype=torch.float32
///     └── loaded: latent_proj.weight
/// ```
pub const Latent_Projection = struct {
    weight: zml.Tensor,

    /// Expects the checkpoint view at `latent_proj`.
    pub fn init(store: zml.io.TensorStore.View) Latent_Projection {
        const here = @typeName(@This()) ++ ".init";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        return .{ .weight = store.createReplicatedTensor("weight", .{ .d, .ld, .k }) };
    }

    /// Projects continuous decoder-space latents, assigning tags `.b`, `.s`, `.ld`.
    ///
    /// `input`: floating-point [B, S, 32], denormalized Phonon output. The caller
    /// first applies `normalized * flow_lm.emb_std + flow_lm.emb_mean` using the
    /// TTS checkpoint's statistics. S is the number of generated audio frames.
    ///
    /// Returns [B, 512, S] in `input.dtype()`, in the channels-first layout
    /// expected by the decoder's temporal upsampling convolution.
    ///
    /// Dimension tags:
    /// - `.b`: independent sequences; batch rows never mix.
    /// - `.s`: audio frames; projected independently and preserved in order.
    /// - `.ld`: 32 latent coordinates, contracted with the projection weight.
    /// - `.d`: 512 projected decoder features.
    /// - `.k`: the weight's single kernel tap, removed before the matrix product.
    ///
    /// Shape: [.b, .s, .ld=32] -> [.b, .d=512, .s].
    pub fn forward(self: Latent_Projection, input: zml.Tensor) zml.Tensor {
        const here = @typeName(@This()) ++ ".forward";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        std.debug.assert(input.dtype().isFloat() and input.rank() == 3);
        std.debug.assert(self.weight.dtype().isFloat() and self.weight.rank() == 3);
        std.debug.assert(self.weight.dim(.k) == 1);
        std.debug.assert(self.weight.dim(.ld) == 32 and self.weight.dim(.d) == 512);

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), "Latent_Projection", .{});
        defer compiler.popLocation();

        const x = input.withTags(.{ .b, .s, .ld });
        std.debug.assert(x.dim(.ld) == 32);
        const weight = self.weight.squeeze(.k).convert(x.dtype());
        const projected = x.dot(weight, .ld);
        return projected.transpose(.{ .b, .d, .s });
    }

    pub fn load(
        self: *const Latent_Projection,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
    ) !zml.Bufferized(Latent_Projection) {
        const here = @typeName(@This()) ++ ".load";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        var buffers = try zml.mem.bufferize(allocator, Latent_Projection, self);
        errdefer unloadBuffers(&buffers);
        var loader: zml.io.Loader = try .init(allocator, platform, .default);
        defer loader.deinit();
        try loader.load(io, Latent_Projection, self, &buffers, store, .{});
        try loader.await(io);
        return buffers;
    }

    pub fn unloadBuffers(buffers: *zml.Bufferized(Latent_Projection)) void {
        const here = @typeName(@This()) ++ ".unloadBuffers";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        zml.Buffer.deinitAll(Latent_Projection, buffers);
    }
};

test Latent_Projection {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    var rng: std.Random.DefaultPrng = .init(42);
    const random = rng.random();

    const layer: Latent_Projection = .{
        .weight = .fromShape(.init(.{ .d = 512, .ld = 32, .k = 1 }, .f32)),
    };
    const input_shape: zml.Shape = .init(.{ .b = 2, .s = 3, .ld = 32 }, .f32);
    var input: [2][3][32]f32 = undefined;
    for (&input) |*batch| {
        for (batch) |*frame| {
            for (frame) |*value| value.* = random.float(f32) * 2 - 1;
        }
    }
    var weights: [512][32][1]f32 = undefined;
    for (&weights) |*row| {
        for (row) |*coordinate| coordinate[0] = random.float(f32) * 2 - 1;
    }
    var weight_buffer = try zml.Buffer.fromBytes(io, platform, layer.weight.shape(), std.mem.asBytes(&weights));
    defer weight_buffer.deinit();
    const weight_buffers: zml.Bufferized(Latent_Projection) = .{ .weight = weight_buffer };
    var input_buffer = try zml.Buffer.fromBytes(io, platform, input_shape, std.mem.asBytes(&input));
    defer input_buffer.deinit();
    var exe = try platform.compile(allocator, io, layer, .forward, .{zml.Tensor.fromShape(input_shape)}, .{});
    defer exe.deinit();
    var output = try exe.eval(allocator, io, .{ weight_buffers, input_buffer });
    defer output.deinit();

    const expected = test_data.latent_projection;
    const output_shape: zml.Shape = .init(.{ .b = 2, .d = 512, .s = 3 }, .f32);
    try std.testing.expect(output.shape().eqlWithTags(output_shape));
    try zml.testing.expectClose(io, zml.Slice.init(output_shape, std.mem.asBytes(&expected)), output, .{
        .absolute_tolerance = 1e-4,
        .relative_tolerance = 1e-4,
        .minimum_close_fraction = 1,
    });

    for ([_]zml.DataType{ .f16, .bf16, .f64 }) |dtype| {
        var typed_exe = try platform.compile(allocator, io, layer, .forward, .{
            zml.Tensor.fromShape(input_shape.withDtype(dtype)),
        }, .{});
        defer typed_exe.deinit();
        try std.testing.expect(typed_exe.output_shapes[0].eqlWithTags(output_shape.withDtype(dtype)));
    }
}

test "Decoder" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    var rng: std.Random.DefaultPrng = .init(42);
    const random = rng.random();
    const layer: Decoder = .{
        .upsample = .{
            .weight = .fromShape(.init(.{ .in = 512, .out = 1, .k = 32 }, .f32)),
            .bias = null,
            .stride = 16,
            .groups = 512,
        },
        .decoder_transformer = .{
            .layers = .{
                .{
                    .self_attn = .{
                        .q_proj = .init(.fromShape(.init(.{ .out = 512, .d = 512 }, .f32)), null, .d),
                        .k_proj = .init(.fromShape(.init(.{ .out = 512, .d = 512 }, .f32)), null, .d),
                        .v_proj = .init(.fromShape(.init(.{ .out = 512, .d = 512 }, .f32)), null, .d),
                        .o_proj = .init(.fromShape(.init(.{ .out = 512, .d = 512 }, .f32)), null, .d),
                    },
                    .mlp = .{
                        .fc1 = .init(.fromShape(.init(.{ .ff = 2048, .d = 512 }, .f32)), null, .d),
                        .fc2 = .init(.fromShape(.init(.{ .d = 512, .ff = 2048 }, .f32)), null, .ff),
                    },
                    .input_layernorm = .{
                        .weight = .fromShape(.init(.{ .d = 512 }, .f32)),
                        .bias = .fromShape(.init(.{ .d = 512 }, .f32)),
                    },
                    .post_attention_layernorm = .{
                        .weight = .fromShape(.init(.{ .d = 512 }, .f32)),
                        .bias = .fromShape(.init(.{ .d = 512 }, .f32)),
                    },
                    .self_attn_layer_scale = .{
                        .scale = .fromShape(.init(.{ .d = 512 }, .f32)),
                    },
                    .mlp_layer_scale = .{
                        .scale = .fromShape(.init(.{ .d = 512 }, .f32)),
                    },
                },
                .{
                    .self_attn = .{
                        .q_proj = .init(.fromShape(.init(.{ .out = 512, .d = 512 }, .f32)), null, .d),
                        .k_proj = .init(.fromShape(.init(.{ .out = 512, .d = 512 }, .f32)), null, .d),
                        .v_proj = .init(.fromShape(.init(.{ .out = 512, .d = 512 }, .f32)), null, .d),
                        .o_proj = .init(.fromShape(.init(.{ .out = 512, .d = 512 }, .f32)), null, .d),
                    },
                    .mlp = .{
                        .fc1 = .init(.fromShape(.init(.{ .ff = 2048, .d = 512 }, .f32)), null, .d),
                        .fc2 = .init(.fromShape(.init(.{ .d = 512, .ff = 2048 }, .f32)), null, .ff),
                    },
                    .input_layernorm = .{
                        .weight = .fromShape(.init(.{ .d = 512 }, .f32)),
                        .bias = .fromShape(.init(.{ .d = 512 }, .f32)),
                    },
                    .post_attention_layernorm = .{
                        .weight = .fromShape(.init(.{ .d = 512 }, .f32)),
                        .bias = .fromShape(.init(.{ .d = 512 }, .f32)),
                    },
                    .self_attn_layer_scale = .{
                        .scale = .fromShape(.init(.{ .d = 512 }, .f32)),
                    },
                    .mlp_layer_scale = .{
                        .scale = .fromShape(.init(.{ .d = 512 }, .f32)),
                    },
                },
            },
        },
        .decoder_seanet = .{
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
    const output_shape: zml.Shape = .init(.{ .b = 2, .d = 1, .s = 5760 }, .f32);
    try std.testing.expect(output.shape().eqlWithTags(output_shape));
    const expected = test_data.decoder;
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

test "Pocket_TTS_Audio_Tokenizer_Model" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    var rng: std.Random.DefaultPrng = .init(42);
    const random = rng.random();
    const layer: Pocket_TTS_Audio_Tokenizer_Model = .{
        .latent_proj = .{
            .weight = .fromShape(.init(.{ .d = 512, .ld = 32, .k = 1 }, .f32)),
        },
        .decoder = .{
            .upsample = .{
                .weight = .fromShape(.init(.{ .in = 512, .out = 1, .k = 32 }, .f32)),
                .bias = null,
                .stride = 16,
                .groups = 512,
            },
            .decoder_transformer = .{
                .layers = .{
                    .{
                        .self_attn = .{
                            .q_proj = .init(.fromShape(.init(.{ .out = 512, .d = 512 }, .f32)), null, .d),
                            .k_proj = .init(.fromShape(.init(.{ .out = 512, .d = 512 }, .f32)), null, .d),
                            .v_proj = .init(.fromShape(.init(.{ .out = 512, .d = 512 }, .f32)), null, .d),
                            .o_proj = .init(.fromShape(.init(.{ .out = 512, .d = 512 }, .f32)), null, .d),
                        },
                        .mlp = .{
                            .fc1 = .init(.fromShape(.init(.{ .ff = 2048, .d = 512 }, .f32)), null, .d),
                            .fc2 = .init(.fromShape(.init(.{ .d = 512, .ff = 2048 }, .f32)), null, .ff),
                        },
                        .input_layernorm = .{
                            .weight = .fromShape(.init(.{ .d = 512 }, .f32)),
                            .bias = .fromShape(.init(.{ .d = 512 }, .f32)),
                        },
                        .post_attention_layernorm = .{
                            .weight = .fromShape(.init(.{ .d = 512 }, .f32)),
                            .bias = .fromShape(.init(.{ .d = 512 }, .f32)),
                        },
                        .self_attn_layer_scale = .{
                            .scale = .fromShape(.init(.{ .d = 512 }, .f32)),
                        },
                        .mlp_layer_scale = .{
                            .scale = .fromShape(.init(.{ .d = 512 }, .f32)),
                        },
                    },
                    .{
                        .self_attn = .{
                            .q_proj = .init(.fromShape(.init(.{ .out = 512, .d = 512 }, .f32)), null, .d),
                            .k_proj = .init(.fromShape(.init(.{ .out = 512, .d = 512 }, .f32)), null, .d),
                            .v_proj = .init(.fromShape(.init(.{ .out = 512, .d = 512 }, .f32)), null, .d),
                            .o_proj = .init(.fromShape(.init(.{ .out = 512, .d = 512 }, .f32)), null, .d),
                        },
                        .mlp = .{
                            .fc1 = .init(.fromShape(.init(.{ .ff = 2048, .d = 512 }, .f32)), null, .d),
                            .fc2 = .init(.fromShape(.init(.{ .d = 512, .ff = 2048 }, .f32)), null, .ff),
                        },
                        .input_layernorm = .{
                            .weight = .fromShape(.init(.{ .d = 512 }, .f32)),
                            .bias = .fromShape(.init(.{ .d = 512 }, .f32)),
                        },
                        .post_attention_layernorm = .{
                            .weight = .fromShape(.init(.{ .d = 512 }, .f32)),
                            .bias = .fromShape(.init(.{ .d = 512 }, .f32)),
                        },
                        .self_attn_layer_scale = .{
                            .scale = .fromShape(.init(.{ .d = 512 }, .f32)),
                        },
                        .mlp_layer_scale = .{
                            .scale = .fromShape(.init(.{ .d = 512 }, .f32)),
                        },
                    },
                },
            },
            .decoder_seanet = .{
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
            },
        },
    };
    const input_shape: zml.Shape = .init(.{ .b = 2, .s = 3, .ld = 32 }, .f32);
    var input: [2][3][32]f32 = undefined;
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
    const output_shape: zml.Shape = .init(.{ .b = 2, .d = 1, .s = 5760 }, .f32);
    try std.testing.expect(output.shape().eqlWithTags(output_shape));
    const expected = test_data.audio_model;
    try zml.testing.expectClose(io, zml.Slice.init(output_shape, std.mem.asBytes(&expected)), output, .{
        .absolute_tolerance = 1e-5,
        .relative_tolerance = 1e-4,
        .minimum_close_fraction = 1,
    });
    for ([_]zml.DataType{ .f16, .bf16, .f64 }) |dtype| {
        var typed_exe = try platform.compile(allocator, io, layer, .forward_uncached, .{zml.Tensor.fromShape(input_shape.withDtype(dtype))}, .{});
        defer typed_exe.deinit();
        try std.testing.expect(typed_exe.output_shapes[0].eqlWithTags(output_shape.withDtype(dtype)));
        const cache = layer.decoder.init_cache(2, 64, .f32);
        var cached_exe = try platform.compile(allocator, io, layer, .forward_cached, .{
            zml.Tensor.fromShape(input_shape.withDtype(dtype)), zml.Tensor.fromShape(.scalar(.i32)), cache,
        }, .{});
        defer cached_exe.deinit();
        try std.testing.expect(cached_exe.output_shapes[0].eqlWithTags(output_shape.withDtype(dtype)));
        const histories = try zml.meta.collectValues(zml.Tensor, allocator, &cache);
        defer allocator.free(histories);
        for (histories, cached_exe.output_shapes[1..]) |history, shape| try std.testing.expect(shape.eqlWithTags(history.shape()));
    }
}

test "Decoder: cached" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    var rng: std.Random.DefaultPrng = .init(42);
    const random = rng.random();
    const layer: Decoder = .{
        .upsample = .{
            .weight = .fromShape(.init(.{ .in = 512, .out = 1, .k = 32 }, .f32)),
            .bias = null,
            .stride = 16,
            .groups = 512,
        },
        .decoder_transformer = .{
            .layers = .{
                .{
                    .self_attn = .{
                        .q_proj = .init(.fromShape(.init(.{ .out = 512, .d = 512 }, .f32)), null, .d),
                        .k_proj = .init(.fromShape(.init(.{ .out = 512, .d = 512 }, .f32)), null, .d),
                        .v_proj = .init(.fromShape(.init(.{ .out = 512, .d = 512 }, .f32)), null, .d),
                        .o_proj = .init(.fromShape(.init(.{ .out = 512, .d = 512 }, .f32)), null, .d),
                    },
                    .mlp = .{
                        .fc1 = .init(.fromShape(.init(.{ .ff = 2048, .d = 512 }, .f32)), null, .d),
                        .fc2 = .init(.fromShape(.init(.{ .d = 512, .ff = 2048 }, .f32)), null, .ff),
                    },
                    .input_layernorm = .{
                        .weight = .fromShape(.init(.{ .d = 512 }, .f32)),
                        .bias = .fromShape(.init(.{ .d = 512 }, .f32)),
                    },
                    .post_attention_layernorm = .{
                        .weight = .fromShape(.init(.{ .d = 512 }, .f32)),
                        .bias = .fromShape(.init(.{ .d = 512 }, .f32)),
                    },
                    .self_attn_layer_scale = .{
                        .scale = .fromShape(.init(.{ .d = 512 }, .f32)),
                    },
                    .mlp_layer_scale = .{
                        .scale = .fromShape(.init(.{ .d = 512 }, .f32)),
                    },
                },
                .{
                    .self_attn = .{
                        .q_proj = .init(.fromShape(.init(.{ .out = 512, .d = 512 }, .f32)), null, .d),
                        .k_proj = .init(.fromShape(.init(.{ .out = 512, .d = 512 }, .f32)), null, .d),
                        .v_proj = .init(.fromShape(.init(.{ .out = 512, .d = 512 }, .f32)), null, .d),
                        .o_proj = .init(.fromShape(.init(.{ .out = 512, .d = 512 }, .f32)), null, .d),
                    },
                    .mlp = .{
                        .fc1 = .init(.fromShape(.init(.{ .ff = 2048, .d = 512 }, .f32)), null, .d),
                        .fc2 = .init(.fromShape(.init(.{ .d = 512, .ff = 2048 }, .f32)), null, .ff),
                    },
                    .input_layernorm = .{
                        .weight = .fromShape(.init(.{ .d = 512 }, .f32)),
                        .bias = .fromShape(.init(.{ .d = 512 }, .f32)),
                    },
                    .post_attention_layernorm = .{
                        .weight = .fromShape(.init(.{ .d = 512 }, .f32)),
                        .bias = .fromShape(.init(.{ .d = 512 }, .f32)),
                    },
                    .self_attn_layer_scale = .{
                        .scale = .fromShape(.init(.{ .d = 512 }, .f32)),
                    },
                    .mlp_layer_scale = .{
                        .scale = .fromShape(.init(.{ .d = 512 }, .f32)),
                    },
                },
            },
        },
        .decoder_seanet = .{
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
    const cache: Decoder_Cache = layer.init_cache(2, 64, .f32);
    var cache_buffers: zml.Bufferized(Decoder_Cache) = undefined;
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
    var cached_output: [2][1][5760]f32 = undefined;
    inline for (.{ .{ .start = 0, .end = 1 }, .{ .start = 1, .end = 3 } }) |step| {
        const count = step.end - step.start;
        const chunk_shape: zml.Shape = .init(.{ .b = 2, .d = 512, .s = count }, .f32);
        var chunk: [2][512][count]f32 = undefined;
        for (&chunk, input) |*batch, source| {
            for (batch, source) |*channel, values| @memcpy(channel, values[step.start..step.end]);
        }
        var chunk_buffer = try zml.Buffer.fromBytes(io, platform, chunk_shape, std.mem.asBytes(&chunk));
        defer chunk_buffer.deinit();
        const position: i32 = step.start * 16;
        var position_buffer = try zml.Buffer.fromBytes(io, platform, .scalar(.i32), std.mem.asBytes(&position));
        defer position_buffer.deinit();
        var exe = try platform.compile(allocator, io, layer, .forward_cached, .{ zml.Tensor.fromShape(chunk_shape), zml.Tensor.fromShape(.scalar(.i32)), cache }, .{});
        defer exe.deinit();
        var runner = try exe.runner(allocator);
        defer runner.deinit(allocator);
        var result: zml.Bufferized(Cached_Audio) = undefined;
        runner.run(io, .{ weight_buffers, chunk_buffer, position_buffer, cache_buffers }, .{&result}, .{ .wait = true });
        defer result.audio.deinit();
        zml.Buffer.deinitAll(Decoder_Cache, &cache_buffers);
        cache_buffers = result.cache;
        const values = try result.audio.getValue([2][1][count * 1920]f32, io);
        for (&cached_output, values) |*batch, source| {
            for (batch, source) |*channel, samples| @memcpy(channel[step.start * 1920 .. step.end * 1920], &samples);
        }
    }
    const output_shape: zml.Shape = .init(.{ .b = 2, .d = 1, .s = 5760 }, .f32);
    try std.testing.expect(uncached_output.shape().eqlWithTags(output_shape));
    try zml.testing.expectClose(io, uncached_output, zml.Slice.init(output_shape, std.mem.asBytes(&cached_output)), .{
        .absolute_tolerance = 1e-5,
        .relative_tolerance = 1e-4,
        .minimum_close_fraction = 1,
    });
}

test "Pocket_TTS_Audio_Tokenizer_Model: cached" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    var rng: std.Random.DefaultPrng = .init(42);
    const random = rng.random();
    const layer: Pocket_TTS_Audio_Tokenizer_Model = .{
        .latent_proj = .{
            .weight = .fromShape(.init(.{ .d = 512, .ld = 32, .k = 1 }, .f32)),
        },
        .decoder = .{
            .upsample = .{
                .weight = .fromShape(.init(.{ .in = 512, .out = 1, .k = 32 }, .f32)),
                .bias = null,
                .stride = 16,
                .groups = 512,
            },
            .decoder_transformer = .{
                .layers = .{
                    .{
                        .self_attn = .{
                            .q_proj = .init(.fromShape(.init(.{ .out = 512, .d = 512 }, .f32)), null, .d),
                            .k_proj = .init(.fromShape(.init(.{ .out = 512, .d = 512 }, .f32)), null, .d),
                            .v_proj = .init(.fromShape(.init(.{ .out = 512, .d = 512 }, .f32)), null, .d),
                            .o_proj = .init(.fromShape(.init(.{ .out = 512, .d = 512 }, .f32)), null, .d),
                        },
                        .mlp = .{
                            .fc1 = .init(.fromShape(.init(.{ .ff = 2048, .d = 512 }, .f32)), null, .d),
                            .fc2 = .init(.fromShape(.init(.{ .d = 512, .ff = 2048 }, .f32)), null, .ff),
                        },
                        .input_layernorm = .{
                            .weight = .fromShape(.init(.{ .d = 512 }, .f32)),
                            .bias = .fromShape(.init(.{ .d = 512 }, .f32)),
                        },
                        .post_attention_layernorm = .{
                            .weight = .fromShape(.init(.{ .d = 512 }, .f32)),
                            .bias = .fromShape(.init(.{ .d = 512 }, .f32)),
                        },
                        .self_attn_layer_scale = .{
                            .scale = .fromShape(.init(.{ .d = 512 }, .f32)),
                        },
                        .mlp_layer_scale = .{
                            .scale = .fromShape(.init(.{ .d = 512 }, .f32)),
                        },
                    },
                    .{
                        .self_attn = .{
                            .q_proj = .init(.fromShape(.init(.{ .out = 512, .d = 512 }, .f32)), null, .d),
                            .k_proj = .init(.fromShape(.init(.{ .out = 512, .d = 512 }, .f32)), null, .d),
                            .v_proj = .init(.fromShape(.init(.{ .out = 512, .d = 512 }, .f32)), null, .d),
                            .o_proj = .init(.fromShape(.init(.{ .out = 512, .d = 512 }, .f32)), null, .d),
                        },
                        .mlp = .{
                            .fc1 = .init(.fromShape(.init(.{ .ff = 2048, .d = 512 }, .f32)), null, .d),
                            .fc2 = .init(.fromShape(.init(.{ .d = 512, .ff = 2048 }, .f32)), null, .ff),
                        },
                        .input_layernorm = .{
                            .weight = .fromShape(.init(.{ .d = 512 }, .f32)),
                            .bias = .fromShape(.init(.{ .d = 512 }, .f32)),
                        },
                        .post_attention_layernorm = .{
                            .weight = .fromShape(.init(.{ .d = 512 }, .f32)),
                            .bias = .fromShape(.init(.{ .d = 512 }, .f32)),
                        },
                        .self_attn_layer_scale = .{
                            .scale = .fromShape(.init(.{ .d = 512 }, .f32)),
                        },
                        .mlp_layer_scale = .{
                            .scale = .fromShape(.init(.{ .d = 512 }, .f32)),
                        },
                    },
                },
            },
            .decoder_seanet = .{
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
            },
        },
    };
    const input_shape: zml.Shape = .init(.{ .b = 2, .s = 17, .ld = 32 }, .f32);
    var input: [2][17][32]f32 = undefined;
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
    const cache: Decoder_Cache = layer.decoder.init_cache(2, 288, .f32);
    var cache_buffers: zml.Bufferized(Decoder_Cache) = undefined;
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
    var cached_output: [2][1][32640]f32 = undefined;
    inline for (.{ .{ .start = 0, .end = 1 }, .{ .start = 1, .end = 3 }, .{ .start = 3, .end = 17 } }) |step| {
        const count = step.end - step.start;
        const chunk_shape: zml.Shape = .init(.{ .b = 2, .s = count, .ld = 32 }, .f32);
        var chunk: [2][count][32]f32 = undefined;
        for (&chunk, input) |*batch, source| @memcpy(batch, source[step.start..step.end]);
        var chunk_buffer = try zml.Buffer.fromBytes(io, platform, chunk_shape, std.mem.asBytes(&chunk));
        defer chunk_buffer.deinit();
        const position: i32 = step.start * 16;
        var position_buffer = try zml.Buffer.fromBytes(io, platform, .scalar(.i32), std.mem.asBytes(&position));
        defer position_buffer.deinit();
        var exe = try platform.compile(allocator, io, layer, .forward_cached, .{ zml.Tensor.fromShape(chunk_shape), zml.Tensor.fromShape(.scalar(.i32)), cache }, .{});
        defer exe.deinit();
        var runner = try exe.runner(allocator);
        defer runner.deinit(allocator);
        var result: zml.Bufferized(Cached_Audio) = undefined;
        runner.run(io, .{ weight_buffers, chunk_buffer, position_buffer, cache_buffers }, .{&result}, .{ .wait = true });
        defer result.audio.deinit();
        zml.Buffer.deinitAll(Decoder_Cache, &cache_buffers);
        cache_buffers = result.cache;
        const values = try result.audio.getValue([2][1][count * 1920]f32, io);
        for (&cached_output, values) |*batch, source| {
            for (batch, source) |*channel, samples| @memcpy(channel[step.start * 1920 .. step.end * 1920], &samples);
        }
    }
    const output_shape: zml.Shape = .init(.{ .b = 2, .d = 1, .s = 32640 }, .f32);
    try std.testing.expect(uncached_output.shape().eqlWithTags(output_shape));
    try zml.testing.expectClose(io, uncached_output, zml.Slice.init(output_shape, std.mem.asBytes(&cached_output)), .{
        .absolute_tolerance = 1e-5,
        .relative_tolerance = 1e-4,
        .minimum_close_fraction = 1,
    });
}
