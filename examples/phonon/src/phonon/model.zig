const std = @import("std");
const zml = @import("zml");

const transformer = @import("transformer.zig");
const eos_head = @import("eos_head.zig");
const flow_head = @import("flow_head.zig");
const test_data = @import("model_test_data.zig");

test {
    std.testing.refAllDecls(@This());
}

// A gentle reminder of the tags we're dealing with:
//
// | Tag     | Meaning                                                            | Size in this implementation                           |
// | ------- | ------------------------------------------------------------------ | ----------------------------------------------------: |
// | `.b`    | Batch: independent sequences                                       | 1 in the executable                                   |
// | `.s`    | Sequence positions                                                 | Prompt positions plus generated-history positions     |
// | `.d`    | Transformer or flow hidden feature dimension                       | 512                                                   |
// | `.dout` | A distinct output-feature axis during a projection                 | Depends on the layer                                  |
// | `.voc`  | Rows of an embedding vocabulary                                    | 4001 for text; 32 for speaker count                   |
// | `.p`    | Frames in the saved voice prompt                                   | 125 for the currently selected voice                  |
// | `.ad`   | Features per voice-prompt frame                                    | 512                                                   |
// | `.ld`   | Features per generated-audio latent / BOS latent                   | 32                                                    |
// | `.a`    | Completed generated-audio frames supplied as history               | 0 initially, then 1, 2, 3, …                          |
// | `.t`    | Scalar flow time per batch row                                     | 1; its value is progress within one frame's sampling  |
// | `.f`    | Timestep frequency/features                                        | 128 stored frequencies, then 256 cosine/sine features |
// | `.m`    | Condition-derived modulation features                              | 1536 for shift/scale/gate; 1024 for shift/scale       |
// | `.spk`  | Features in the speaker-count embedding                            | 16                                                    |
// | `.dur`  | Features in the duration encoding                                  | 128                                                   |
// | `.qkv`  | Packed projection output, or one chunk before splitting into heads | 1536, then 512                                        |
// | `.h`    | Attention heads                                                    | 8                                                     |
// | `.hd`   | Features per attention head                                        | 64                                                    |
// | `.q`    | Query positions in attention                                       | Same as `.s` during prefill                           |
// | `.k`    | Key/value positions in attention                                   | Same as `.s` during prefill                           |
// | `.hq`   | Internal SDPA query-head grouping axis                             | 1 here                                                |

// For reference, the entire pipeline:
//
// Text + selected voice
//   ↓
// Conversation template
//   ↓
// Expand voice placeholders
//   ↓
// Text normalization + tokenization
//   ↓
// Token IDs + voice latents + masks
//   ↓
// Embedding lookup + replace special positions with projected audio features
//   ↓
// Six transformer layers
//   ↓
// Final normalization
//   ↓
// Flow head + stop prediction
//   ↓
// Repeat to generate audio latent frames
//   ↓
// Convert latent representation for audio decoder
//   ↓
// Audio decoder neural network
//   ↓
// Waveform trimming / optional fading
//   ↓
// WAV file / playback

// --- Gradium_Phonon_TTS_For_Conditional_Generation ---

/// Combines the transformer, EOS head, and flow head for one audio-frame prediction.
///
/// The transformer builds contextual features from text, voice, and preceding
/// audio. Only its last position conditions the two heads: EOS produces a raw
/// stopping score, while the flow head predicts a velocity for a sampler-supplied
/// candidate audio latent and flow interval. The caller updates that candidate,
/// retains the completed frame, and applies the stopping policy.
///
/// `forward_uncached` recomputes the full prefix. `forward_cached_prefill` processes the initial
/// prompt and fills a K/V cache; `forward_cached_decode` processes one completed audio
/// frame using that cached context. Prefill already produces the prediction for
/// the first audio frame. All paths return EOS scores `[B, 1]` in the voice dtype
/// (uncached/prefill) or frame dtype (decode), and velocities `[B, 32]` in the
/// candidate dtype. Cached paths preserve the input K/V storage dtypes.
/// B counts independent sequences; inputs must be unpadded.
///
/// Parameters come from the root checkpoint view: `model` owns the transformer,
/// `flow_head` uses `flow_lm.flow_net`, and `eos_head` uses `flow_lm.out_eos`.
/// The children target phonon-31m-en and convert stored bf16 parameters to their
/// respective floating input dtypes.
/// Sampling, stopping, latent rescaling, and waveform decoding belong to callers.
///
/// Reference:
/// ```
/// GradiumPhononTTSForConditionalGeneration
/// ├── model: GradiumPhononTTSModel
/// │   ├── embed_tokens: Embedding
/// │   │   └── weight: shape=(4001, 512), dtype=torch.float32
/// │   │       ├── loaded: model.embed_tokens.weight
/// │   │       └── checkpoint: flow_lm.condition_provider.conditioners.transcript_in_segment.embed.weight
/// │   ├── layers: ModuleList
/// │   │   └── 0..5: GradiumPhononTTSDecoderLayer
/// │   │       ├── self_attn: GradiumPhononTTSAttention
/// │   │       │   ├── q_proj: Linear
/// │   │       │   │   └── weight: shape=(512, 512), dtype=torch.float32
/// │   │       │   │       ├── loaded: model.layers.5.self_attn.q_proj.weight
/// │   │       │   │       └── checkpoint: flow_lm.transformer.layers.5.self_attn.in_proj.weight
/// │   │       │   ├── k_proj: Linear
/// │   │       │   │   └── weight: shape=(512, 512), dtype=torch.float32
/// │   │       │   │       ├── loaded: model.layers.5.self_attn.k_proj.weight
/// │   │       │   │       └── checkpoint: flow_lm.transformer.layers.5.self_attn.in_proj.weight
/// │   │       │   ├── v_proj: Linear
/// │   │       │   │   └── weight: shape=(512, 512), dtype=torch.float32
/// │   │       │   │       ├── loaded: model.layers.5.self_attn.v_proj.weight
/// │   │       │   │       └── checkpoint: flow_lm.transformer.layers.5.self_attn.in_proj.weight
/// │   │       │   └── o_proj: Linear
/// │   │       │       └── weight: shape=(512, 512), dtype=torch.float32
/// │   │       │           ├── loaded: model.layers.5.self_attn.o_proj.weight
/// │   │       │           └── checkpoint: flow_lm.transformer.layers.5.self_attn.out_proj.weight
/// │   │       ├── mlp: GradiumPhononTTSMLP
/// │   │       │   ├── activation_fn: GELUActivation
/// │   │       │   ├── fc1: Linear
/// │   │       │   │   └── weight: shape=(2048, 512), dtype=torch.float32
/// │   │       │   │       ├── loaded: model.layers.5.mlp.fc1.weight
/// │   │       │   │       └── checkpoint: flow_lm.transformer.layers.5.linear1.weight
/// │   │       │   └── fc2: Linear
/// │   │       │       └── weight: shape=(512, 2048), dtype=torch.float32
/// │   │       │           ├── loaded: model.layers.5.mlp.fc2.weight
/// │   │       │           └── checkpoint: flow_lm.transformer.layers.5.linear2.weight
/// │   │       ├── input_layernorm: LayerNorm
/// │   │       │   ├── weight: shape=(512,), dtype=torch.float32
/// │   │       │   │   ├── loaded: model.layers.5.input_layernorm.weight
/// │   │       │   │   └── checkpoint: flow_lm.transformer.layers.5.norm1.weight
/// │   │       │   └── bias: shape=(512,), dtype=torch.float32
/// │   │       │       ├── loaded: model.layers.5.input_layernorm.bias
/// │   │       │       └── checkpoint: flow_lm.transformer.layers.5.norm1.bias
/// │   │       └── post_attention_layernorm: LayerNorm
/// │   │           ├── weight: shape=(512,), dtype=torch.float32
/// │   │           │   ├── loaded: model.layers.5.post_attention_layernorm.weight
/// │   │           │   └── checkpoint: flow_lm.transformer.layers.5.norm2.weight
/// │   │           └── bias: shape=(512,), dtype=torch.float32
/// │   │               ├── loaded: model.layers.5.post_attention_layernorm.bias
/// │   │               └── checkpoint: flow_lm.transformer.layers.5.norm2.bias
/// │   ├── norm: LayerNorm
/// │   │   ├── weight: shape=(512,), dtype=torch.float32
/// │   │   │   ├── loaded: model.norm.weight
/// │   │   │   └── checkpoint: flow_lm.out_norm.weight
/// │   │   └── bias: shape=(512,), dtype=torch.float32
/// │   │       ├── loaded: model.norm.bias
/// │   │       └── checkpoint: flow_lm.out_norm.bias
/// │   ├── rotary_emb: GradiumPhononTTSRotaryEmbedding
/// │   ├── audio_prompt_proj: Linear
/// │   │   └── weight: shape=(512, 512), dtype=torch.float32
/// │   │       ├── loaded: model.audio_prompt_proj.weight
/// │   │       └── checkpoint: flow_lm.condition_provider.conditioners.speaker_wavs.output_proj.weight
/// │   ├── audio_latent_proj: Linear
/// │   │   └── weight: shape=(512, 32), dtype=torch.float32
/// │   │       ├── loaded: model.audio_latent_proj.weight
/// │   │       └── checkpoint: flow_lm.input_linear.weight
/// │   ├── num_speakers_embed: Embedding
/// │   │   └── weight: shape=(32, 16), dtype=torch.float32
/// │   │       ├── loaded: model.num_speakers_embed.weight
/// │   │       └── checkpoint: flow_lm.condition_provider.conditioners.num_speakers.embed.weight
/// │   ├── num_speakers_proj: Linear
/// │   │   └── weight: shape=(512, 16), dtype=torch.float32
/// │   │       ├── loaded: model.num_speakers_proj.weight
/// │   │       └── checkpoint: flow_lm.condition_provider.conditioners.num_speakers.output_proj.weight
/// │   ├── duration_delta_proj: Linear
/// │   │   └── weight: shape=(512, 128), dtype=torch.float32
/// │   │       ├── loaded: model.duration_delta_proj.weight
/// │   │       └── checkpoint: flow_lm.condition_provider.conditioners.duration_delta.output_proj.weight
/// │   └── bos_latent: shape=(32,), dtype=torch.float32
/// │       ├── loaded: model.bos_latent
/// │       └── checkpoint: flow_lm.bos_emb
/// ├── flow_head: GradiumPhononTTSFlowHead
/// │   ├── input_proj: Linear
/// │   │   ├── weight: shape=(512, 32), dtype=torch.float32
/// │   │   │   ├── loaded: flow_head.input_proj.weight
/// │   │   │   └── checkpoint: flow_lm.flow_net.input_proj.weight
/// │   │   └── bias: shape=(512,), dtype=torch.float32
/// │   │       ├── loaded: flow_head.input_proj.bias
/// │   │       └── checkpoint: flow_lm.flow_net.input_proj.bias
/// │   ├── cond_proj: Linear
/// │   │   ├── weight: shape=(512, 512), dtype=torch.float32
/// │   │   │   ├── loaded: flow_head.cond_proj.weight
/// │   │   │   └── checkpoint: flow_lm.flow_net.cond_embed.weight
/// │   │   └── bias: shape=(512,), dtype=torch.float32
/// │   │       ├── loaded: flow_head.cond_proj.bias
/// │   │       └── checkpoint: flow_lm.flow_net.cond_embed.bias
/// │   ├── timestep_embeddings: ModuleList
/// │   │   └── 0..1: GradiumPhononTTSTimestepEmbedding
/// │   │       ├── fc1: Linear
/// │   │       │   ├── weight: shape=(512, 256), dtype=torch.float32
/// │   │       │   │   ├── loaded: flow_head.timestep_embeddings.1.fc1.weight
/// │   │       │   │   └── checkpoint: flow_lm.flow_net.time_embed.1.mlp.0.weight
/// │   │       │   └── bias: shape=(512,), dtype=torch.float32
/// │   │       │       ├── loaded: flow_head.timestep_embeddings.1.fc1.bias
/// │   │       │       └── checkpoint: flow_lm.flow_net.time_embed.1.mlp.0.bias
/// │   │       ├── fc2: Linear
/// │   │       │   ├── weight: shape=(512, 512), dtype=torch.float32
/// │   │       │   │   ├── loaded: flow_head.timestep_embeddings.1.fc2.weight
/// │   │       │   │   └── checkpoint: flow_lm.flow_net.time_embed.1.mlp.2.weight
/// │   │       │   └── bias: shape=(512,), dtype=torch.float32
/// │   │       │       ├── loaded: flow_head.timestep_embeddings.1.fc2.bias
/// │   │       │       └── checkpoint: flow_lm.flow_net.time_embed.1.mlp.2.bias
/// │   │       ├── act_fn: SiLUActivation
/// │   │       └── norm: GradiumPhononTTSTimestepNorm
/// │   │           └── weight: shape=(512,), dtype=torch.float32
/// │   │               ├── loaded: flow_head.timestep_embeddings.1.norm.weight
/// │   │               └── checkpoint: flow_lm.flow_net.time_embed.1.mlp.3.alpha
/// │   ├── layers: ModuleList
/// │   │   └── 0..5: GradiumPhononTTSFlowResBlock
/// │   │       ├── norm: GradiumPhononTTSFlowLayerNorm
/// │   │       │   ├── weight: shape=(512,), dtype=torch.float32
/// │   │       │   │   ├── loaded: flow_head.layers.5.norm.weight
/// │   │       │   │   └── checkpoint: flow_lm.flow_net.res_blocks.5.in_ln.weight
/// │   │       │   └── bias: shape=(512,), dtype=torch.float32
/// │   │       │       ├── loaded: flow_head.layers.5.norm.bias
/// │   │       │       └── checkpoint: flow_lm.flow_net.res_blocks.5.in_ln.bias
/// │   │       ├── fc1: Linear
/// │   │       │   ├── weight: shape=(512, 512), dtype=torch.float32
/// │   │       │   │   ├── loaded: flow_head.layers.5.fc1.weight
/// │   │       │   │   └── checkpoint: flow_lm.flow_net.res_blocks.5.mlp.0.weight
/// │   │       │   └── bias: shape=(512,), dtype=torch.float32
/// │   │       │       ├── loaded: flow_head.layers.5.fc1.bias
/// │   │       │       └── checkpoint: flow_lm.flow_net.res_blocks.5.mlp.0.bias
/// │   │       ├── fc2: Linear
/// │   │       │   ├── weight: shape=(512, 512), dtype=torch.float32
/// │   │       │   │   ├── loaded: flow_head.layers.5.fc2.weight
/// │   │       │   │   └── checkpoint: flow_lm.flow_net.res_blocks.5.mlp.2.weight
/// │   │       │   └── bias: shape=(512,), dtype=torch.float32
/// │   │       │       ├── loaded: flow_head.layers.5.fc2.bias
/// │   │       │       └── checkpoint: flow_lm.flow_net.res_blocks.5.mlp.2.bias
/// │   │       ├── act_fn: SiLUActivation
/// │   │       └── adaln_proj: Linear
/// │   │           ├── weight: shape=(1536, 512), dtype=torch.float32
/// │   │           │   ├── loaded: flow_head.layers.5.adaln_proj.weight
/// │   │           │   └── checkpoint: flow_lm.flow_net.res_blocks.5.adaLN_modulation.1.weight
/// │   │           └── bias: shape=(1536,), dtype=torch.float32
/// │   │               ├── loaded: flow_head.layers.5.adaln_proj.bias
/// │   │               └── checkpoint: flow_lm.flow_net.res_blocks.5.adaLN_modulation.1.bias
/// │   └── final_layer: GradiumPhononTTSFlowFinalLayer
/// │       ├── norm: GradiumPhononTTSFlowLayerNorm
/// │       ├── adaln_proj: Linear
/// │       │   ├── weight: shape=(1024, 512), dtype=torch.float32
/// │       │   │   ├── loaded: flow_head.final_layer.adaln_proj.weight
/// │       │   │   └── checkpoint: flow_lm.flow_net.final_layer.adaLN_modulation.1.weight
/// │       │   └── bias: shape=(1024,), dtype=torch.float32
/// │       │       ├── loaded: flow_head.final_layer.adaln_proj.bias
/// │       │       └── checkpoint: flow_lm.flow_net.final_layer.adaLN_modulation.1.bias
/// │       ├── act_fn: SiLUActivation
/// │       └── proj: Linear
/// │           ├── weight: shape=(32, 512), dtype=torch.float32
/// │           │   ├── loaded: flow_head.final_layer.proj.weight
/// │           │   └── checkpoint: flow_lm.flow_net.final_layer.linear.weight
/// │           └── bias: shape=(32,), dtype=torch.float32
/// │               ├── loaded: flow_head.final_layer.proj.bias
/// │               └── checkpoint: flow_lm.flow_net.final_layer.linear.bias
/// └── eos_head: Linear
///     ├── weight: shape=(1, 512), dtype=torch.float32
///     │   ├── loaded: eos_head.weight
///     │   └── checkpoint: flow_lm.out_eos.weight
///     └── bias: shape=(1,), dtype=torch.float32
///         ├── loaded: eos_head.bias
///         └── checkpoint: flow_lm.out_eos.bias
/// ```
pub const Gradium_Phonon_TTS_For_Conditional_Generation = struct {
    model: transformer.Gradium_Phonon_TTS_Model,
    flow_head: flow_head.Gradium_Phonon_TTS_Flow_Head,
    eos_head: eos_head.EOS_Head,

    /// Returns the two predictions used to complete one normalized audio frame.
    ///
    /// `eos_score`: floating-point `[.b=B, .dout=1]`, a raw stopping logit from the last
    /// transformer position. The host generation loop interprets it using its policy.
    /// `velocity`: floating-point `[.b=B, .ld=32]`, the flow head's latent velocity for the
    /// supplied candidate and interval. The sampler applies the numerical update.
    /// EOS follows the voice dtype (uncached/prefill) or frame dtype (decode);
    /// velocity follows the candidate dtype. Both tensors preserve batch ordering.
    /// Executable results use `Bufferized(Output)`; the caller releases both buffers.
    pub const Output = struct {
        eos_score: zml.Tensor,
        active_length: zml.Tensor,
        frame: zml.Tensor,
    };

    /// Expects the root checkpoint view.
    pub fn init(store: zml.io.TensorStore.View) Gradium_Phonon_TTS_For_Conditional_Generation {
        const here = @typeName(@This()) ++ ".init";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        return .{
            .model = .init(store),
            .flow_head = .init(store.withPrefix("flow_lm.flow_net")),
            .eos_head = .init(store.withPrefix("flow_lm.out_eos")),
        };
    }

    /// Predicts an EOS score and flow velocity from the complete sequence prefix.
    ///
    /// `token_ids`: integer `[B, S]`, normally i64, the unpadded sequence from
    /// the tokenizer and generation loop. S>0; the prompt ends at audio BOS (ID 4)
    /// initially, then gains one ID 6 per completed generated frame.
    /// `voice`: floating-point `[B, P, 512]`, selected speaker features loaded from the voice
    /// asset as `[B, 512, P]` and transposed by the caller. P matches the number
    /// of ID-5 markers in each sequence, in frame order.
    /// `audio_latents`: floating-point `[B, A, 32]`, the generation loop's complete normalized
    /// audio history, one completed frame per ID-6 marker in order; A=0 initially.
    /// `candidate`: floating-point `[B, 32]`, the sampler's current candidate for the next
    /// frame in normalized latent space; freshly drawn noise in the one-step sampler.
    /// `start_time`: floating-point `[B, 1]`, starting flow progress supplied by the sampler.
    /// `end_time`: floating-point `[B, 1]`, ending flow progress supplied by the sampler.
    /// Times normally lie in [0, 1]; the current sampler uses 0 and 1. They describe
    /// flow progress, rather than audio timestamps. All inputs share batch size B.
    ///
    /// Returns EOS logits `[B, 1]` and velocities `[B, 32]`, both conditioned
    /// on only the last position of the transformer output. The caller applies the velocity
    /// update, stopping policy, and feedback of completed frames. EOS uses
    /// `voice.dtype()`; velocity uses `candidate.dtype()`. The transformer converts
    /// history and parameters to the voice dtype; the flow head converts its
    /// condition and times to the candidate dtype.
    ///
    /// Dimension tags:
    /// - `.b`: batch; independent prompts, candidates and sampling intervals.
    /// - `.s`: sequence; full-prefix positions, reduced to the final position for both heads.
    /// - `.p`: prompt frames; ordered voice frames, one per audio-prompt token.
    /// - `.ad`: 512 voice-prompt features per frame.
    /// - `.d`: 512 final transformer features conditioning both heads.
    /// - `.a`: completed generated frames in the full history; zero initially.
    /// - `.ld`: 32 normalized history/candidate/velocity components.
    /// - `.t`: one scalar flow time per batch row, for each interval endpoint.
    /// - `.dout`: one raw EOS score per batch row; no sigmoid or threshold applied.
    ///
    /// Inputs: integer IDs `[b,s]`, floating-point voice `[b,p,ad=512]`, complete
    /// history `[b,a,ld=32]`, candidate `[b,ld=32]`, and times `[b,t=1]` each.
    /// Outputs: eos_score `[b,dout=1]` in the voice dtype and velocity `[b,ld=32]`
    /// in the candidate dtype.
    pub fn forward_uncached(
        self: Gradium_Phonon_TTS_For_Conditional_Generation,
        token_ids: zml.Tensor,
        voice: zml.Tensor,
        audio_latents: zml.Tensor,
        candidate: zml.Tensor,
        start_time: zml.Tensor,
        end_time: zml.Tensor,
        active_length: zml.Tensor,
        frame: zml.Tensor,
    ) Output {
        const here = @typeName(@This()) ++ ".forward_uncached";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), "Gradium_Phonon_TTS_For_Conditional_Generation", .{});
        defer compiler.popLocation();

        const hidden = self.model.forward_uncached(token_ids, voice, audio_latents);
        const condition = last_position(hidden);

        const predict_output = self.predict(condition, candidate, start_time, end_time);
        return .{
            .eos_score = predict_output.eos_score,
            .active_length = active_length.addConstant(1).reuseBuffer(active_length),
            .frame = candidate.add(predict_output.velocity).reuseBuffer(frame),
        };
    }

    /// Returns an EOS score, latent velocity, and updated attention history.
    ///
    /// `eos_score` is `[.b=B, .dout=1]` and `velocity` is `[.b=B, .ld=32]`,
    /// with the same meanings as `Output`. `cache` contains rotated keys and
    /// unrotated values, each floating-point `[.layer=6, .b=B, .k=C, .h=8, .hd=64]`.
    /// Each cache tensor preserves its input storage dtype.
    /// The cached wrapper donates input cache storage to these returned tensors.
    /// Raw executable callers must destroy the old donated handles and retain the
    /// returned handles exactly once; the cache capacity C stays fixed.
    pub const CachedOutput = struct {
        eos_score: zml.Tensor,
        velocity: zml.Tensor,
        cache: transformer.KV_Cache,
    };

    /// Processes the complete prompt once and predicts the first audio frame.
    ///
    /// `ids`: integer `[B, S]`, normally i64, the tokenizer's unpadded prompt
    /// ending at audio BOS (ID 4), uploaded by the generation loop. S must be nonzero.
    /// `voice`: floating-point `[B, P, 512]`, selected speaker features loaded from the voice
    /// asset and transposed by the host; one ordered frame per ID-5 marker.
    /// `candidate`: floating-point `[B, 32]`, the sampler's candidate for the next normalized
    /// audio frame, initially noise; passed through to the flow head.
    /// `start`: floating-point `[B, 1]`, the sampler-supplied interval's starting flow progress.
    /// `end`: floating-point `[B, 1]`, the sampler-supplied interval's ending flow progress.
    /// The current one-step sampler supplies 0 and 1 and computes candidate + velocity.
    /// `cache`: host-allocated floating-point K/V storage, each `[6, B, C, 8, 64]`, initially
    /// filled with finite zeros. C must be at least the prompt length S.
    /// All inputs share B. There are no completed generated frames yet.
    ///
    /// Returns EOS logits `[B, 1]`, latent velocities `[B, 32]`, and updated cache.
    /// Only the last prompt position conditions the heads. The cache contains all
    /// prompt positions and donates its storage to the returned handles. EOS uses
    /// `voice.dtype()`; velocity uses `candidate.dtype()`. The flow head converts
    /// the condition and times to the candidate dtype.
    ///
    /// Dimension tags:
    /// - `.b`: sequences; `.s`: prompt positions; `.p`: voice frames; `.ad`: 512 voice features.
    /// - `.d`: 512 transformer features; `.dout`: one EOS score; `.ld`: 32 latent coordinates.
    /// - `.t`: one flow time per row; cache `.layer`=6, `.k`=C, `.h`=8, `.hd`=64.
    pub fn forward_cached_prefill(
        self: Gradium_Phonon_TTS_For_Conditional_Generation,
        ids: zml.Tensor,
        voice: zml.Tensor,
        candidate: zml.Tensor,
        start: zml.Tensor,
        end: zml.Tensor,
        cache: transformer.KV_Cache,
    ) CachedOutput {
        const here = @typeName(@This()) ++ ".forward_cached_prefill";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        // Prefill has voice context but no completed generated frames.
        const no_audio = zml.Tensor.constantTensor(.init(.{ .b = ids.dim(0), .a = 0, .ld = 32 }, voice.dtype()), &.{});
        const position = zml.Tensor.scalar(0, .i32);
        const contextualized = self.model.forward_cached(ids, voice, no_audio, position, cache);
        const condition = last_position(contextualized.hidden);
        const prediction = self.predict(condition, candidate, start, end);
        return .{
            .eos_score = prediction.eos_score,
            .velocity = prediction.velocity,
            .cache = contextualized.cache.reuseBuffer(cache),
        };
    }

    /// Consumes one completed audio frame and predicts the next using cached context.
    ///
    /// `frame`: floating-point `[B, 1, 32]`, the latest completed normalized latent, selected
    /// from accumulated output history by the generation loop. It becomes the
    /// transformer's single new input position, rather than the new sampler candidate.
    /// `position`: scalar i32 `[]`, that frame's absolute sequence slot, supplied
    /// by the generation loop. The first decode slot follows the prompt; later calls
    /// advance by one. The host must keep it within the cache capacity.
    /// `candidate`: floating-point `[B, 32]`, the sampler's candidate for the next normalized
    /// audio frame, initially noise; passed through to the flow head.
    /// `start`: floating-point `[B, 1]`, the sampler-supplied interval's starting flow progress.
    /// `end`: floating-point `[B, 1]`, the sampler-supplied interval's ending flow progress.
    /// The current one-step sampler supplies 0 and 1 and computes candidate + velocity.
    /// `cache`: floating-point K/V history, each `[6, B, C, 8, 64]`, returned by the previous
    /// prefill/decode call. All earlier sequence slots must already be populated.
    /// Frame, candidate, times, and cache share batch size B.
    ///
    /// Returns EOS logits `[B, 1]`, latent velocities `[B, 32]`, and updated cache.
    /// The transformer produces `[B, 1, 512]` and the heads use its single position.
    /// Exactly this new slot is written per layer; cache storage is donated to the
    /// returned handles. EOS uses `frame.dtype()`; velocity uses `candidate.dtype()`.
    /// The flow head converts the condition and times to the candidate dtype.
    /// The caller updates the candidate and applies stopping rules.
    ///
    /// Dimension tags:
    /// - `.b`: sequences; frame `.a`=1, hidden `.s`=1; `.d`=512 transformer features.
    /// - `.ld`: 32 latent coordinates; `.t`: one flow time; `.dout`: one EOS score.
    /// - Cache: `.layer`=6, `.k`=C slots, `.h`=8 heads, `.hd`=64 head features.
    pub fn forward_cached_decode(
        self: Gradium_Phonon_TTS_For_Conditional_Generation,
        frame: zml.Tensor,
        position: zml.Tensor,
        candidate: zml.Tensor,
        start: zml.Tensor,
        end: zml.Tensor,
        cache: transformer.KV_Cache,
    ) CachedOutput {
        std.debug.assert(frame.dtype().isFloat() and frame.rank() == 3 and frame.dim(1) == 1 and frame.dim(2) == 32);

        const here = @typeName(@This()) ++ ".forward_cached_decode";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        // Feed the previous completed frame as one ID-6 position. Voice already
        // lives in the cached prefix, so no new voice frames are supplied.
        const ids_shape: zml.Shape = .init(.{ .b = frame.dim(0), .s = 1 }, .i64);
        const ids = zml.Tensor.scalar(6, .i64).broad(ids_shape);
        const no_voice = zml.Tensor.constantTensor(.init(.{ .b = frame.dim(0), .p = 0, .ad = 512 }, frame.dtype()), &.{});
        const contextualized = self.model.forward_cached(ids, no_voice, frame, position, cache);
        const condition = contextualized.hidden.squeeze(.s);
        const prediction = self.predict(condition, candidate, start, end);
        return .{
            .eos_score = prediction.eos_score,
            .velocity = prediction.velocity,
            .cache = contextualized.cache.reuseBuffer(cache),
        };
    }

    /// Select the final contextual feature vector: [B,.s,512] -> [B,.d=512].
    /// Only that position conditions the next frame's EOS and flow predictions.
    fn last_position(hidden: zml.Tensor) zml.Tensor {
        std.debug.assert(hidden.dim(.s) > 0);
        return hidden.slice(.s, .{ .start = hidden.dim(.s) - 1, .len = 1 }).squeeze(.s);
    }

    pub const Prediction_Output = struct {
        eos_score: zml.Tensor,
        velocity: zml.Tensor,
    };

    /// Context [B,512], candidate [B,32], and interval endpoints [B,1] ->
    /// EOS [B,1] in context dtype and velocity [B,32] in candidate dtype.
    fn predict(
        self: Gradium_Phonon_TTS_For_Conditional_Generation,
        condition: zml.Tensor,
        candidate: zml.Tensor,
        start: zml.Tensor,
        end: zml.Tensor,
    ) Prediction_Output {
        return .{
            .eos_score = self.eos_head.forward(condition),
            .velocity = self.flow_head.forward(candidate, condition, start, end),
        };
    }

    pub fn load(
        self: *const Gradium_Phonon_TTS_For_Conditional_Generation,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
    ) !zml.Bufferized(Gradium_Phonon_TTS_For_Conditional_Generation) {
        const here = @typeName(@This()) ++ ".load";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        // Delegate to children so their layer arrays are loaded and cleaned up correctly.
        var buffers: zml.Bufferized(Gradium_Phonon_TTS_For_Conditional_Generation) = undefined;
        buffers.model = try self.model.load(allocator, io, platform, store);
        errdefer transformer.Gradium_Phonon_TTS_Model.unloadBuffers(&buffers.model);
        buffers.flow_head = try self.flow_head.load(allocator, io, platform, store);
        errdefer flow_head.Gradium_Phonon_TTS_Flow_Head.unloadBuffers(&buffers.flow_head);
        buffers.eos_head = try self.eos_head.load(allocator, io, platform, store);
        return buffers;
    }

    pub fn unload(self: *zml.Bufferized(Gradium_Phonon_TTS_For_Conditional_Generation)) void {
        const here = @typeName(@This()) ++ ".unload";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        transformer.Gradium_Phonon_TTS_Model.unloadBuffers(&self.model);
        flow_head.Gradium_Phonon_TTS_Flow_Head.unloadBuffers(&self.flow_head);
        eos_head.EOS_Head.unloadBuffers(&self.eos_head);
    }
};

test Gradium_Phonon_TTS_For_Conditional_Generation {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    var rng: std.Random.DefaultPrng = .init(42);
    const random = rng.random();

    var body: transformer.Gradium_Phonon_TTS_Model = .{
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
    for (&body.layers) |*layer| {
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

    var head: flow_head.Gradium_Phonon_TTS_Flow_Head = .{
        .input_proj = .{
            .projection = .init(
                .fromShape(.init(.{ .d = 512, .ld = 32 }, .bf16)),
                .fromShape(.init(.{ .d = 512 }, .bf16)),
                .ld,
            ),
        },
        .cond_proj = .{
            .projection = .init(
                .fromShape(.init(.{ .dout = 512, .d = 512 }, .bf16)),
                .fromShape(.init(.{ .dout = 512 }, .bf16)),
                .d,
            ),
        },
        .timestep_embeddings = undefined,
        .layers = undefined,
        .final_layer = .{
            .adaln_proj = .init(
                .fromShape(.init(.{ .m = 1024, .d = 512 }, .bf16)),
                .fromShape(.init(.{ .m = 1024 }, .bf16)),
                .d,
            ),
            .proj = .init(
                .fromShape(.init(.{ .ld = 32, .d = 512 }, .bf16)),
                .fromShape(.init(.{ .ld = 32 }, .bf16)),
                .d,
            ),
        },
    };
    for (&head.timestep_embeddings) |*embedding| {
        embedding.* = .{
            .freqs = .fromShape(.init(.{ .f = 128 }, .bf16)),
            .fc1 = .init(
                .fromShape(.init(.{ .d = 512, .f = 256 }, .bf16)),
                .fromShape(.init(.{ .d = 512 }, .bf16)),
                .f,
            ),
            .fc2 = .init(
                .fromShape(.init(.{ .dout = 512, .d = 512 }, .bf16)),
                .fromShape(.init(.{ .dout = 512 }, .bf16)),
                .d,
            ),
            .norm_weight = .fromShape(.init(.{ .d = 512 }, .bf16)),
        };
    }
    for (&head.layers) |*block| {
        block.* = .{
            .norm_weight = .fromShape(.init(.{ .d = 512 }, .bf16)),
            .norm_bias = .fromShape(.init(.{ .d = 512 }, .bf16)),
            .fc1 = .init(
                .fromShape(.init(.{ .dout = 512, .d = 512 }, .bf16)),
                .fromShape(.init(.{ .dout = 512 }, .bf16)),
                .d,
            ),
            .fc2 = .init(
                .fromShape(.init(.{ .dout = 512, .d = 512 }, .bf16)),
                .fromShape(.init(.{ .dout = 512 }, .bf16)),
                .d,
            ),
            .adaln_proj = .init(
                .fromShape(.init(.{ .m = 1536, .d = 512 }, .bf16)),
                .fromShape(.init(.{ .m = 1536 }, .bf16)),
                .d,
            ),
        };
    }

    const model: Gradium_Phonon_TTS_For_Conditional_Generation = .{
        .model = body,
        .flow_head = head,
        .eos_head = .{ .projection = .init(
            .fromShape(.init(.{ .dout = 1, .d = 512 }, .bf16)),
            .fromShape(.init(.{ .dout = 1 }, .bf16)),
            .d,
        ) },
    };

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

    const candidate_shape: zml.Shape = .init(.{ .b = 2, .ld = 32 }, .f32);
    var candidate: [2][32]f32 = undefined;
    for (&candidate) |*batch| {
        for (batch) |*value| value.* = random.float(f32) * 2 - 1;
    }
    var candidate_buffer = try zml.Buffer.fromBytes(io, platform, candidate_shape, std.mem.asBytes(&candidate));
    defer candidate_buffer.deinit();
    const time_shape: zml.Shape = .init(.{ .b = 2, .t = 1 }, .f32);
    var start: [2][1]f32 = undefined;
    var end: [2][1]f32 = undefined;
    for (&start, &end) |*from, *to| {
        from[0] = random.float(f32);
        to[0] = from[0] + random.float(f32) * (1 - from[0]);
    }
    var start_buffer = try zml.Buffer.fromBytes(io, platform, time_shape, std.mem.asBytes(&start));
    defer start_buffer.deinit();
    var end_buffer = try zml.Buffer.fromBytes(io, platform, time_shape, std.mem.asBytes(&end));
    defer end_buffer.deinit();

    // Bufferize each child and its array elements, as in the production loader.
    var weight_buffers: zml.Bufferized(Gradium_Phonon_TTS_For_Conditional_Generation) = .{
        .model = .{
            .embed_tokens = try zml.mem.bufferize(allocator, @TypeOf(model.model.embed_tokens), &model.model.embed_tokens),
            .input_projection = try zml.mem.bufferize(allocator, @TypeOf(model.model.input_projection), &model.model.input_projection),
            .norm = try zml.mem.bufferize(allocator, zml.nn.LayerNorm, &model.model.norm),
            .layers = undefined,
        },
        .flow_head = .{
            .input_proj = try zml.mem.bufferize(allocator, @TypeOf(model.flow_head.input_proj), &model.flow_head.input_proj),
            .cond_proj = try zml.mem.bufferize(allocator, @TypeOf(model.flow_head.cond_proj), &model.flow_head.cond_proj),
            .final_layer = try zml.mem.bufferize(allocator, @TypeOf(model.flow_head.final_layer), &model.flow_head.final_layer),
            .timestep_embeddings = undefined,
            .layers = undefined,
        },
        .eos_head = try zml.mem.bufferize(allocator, eos_head.EOS_Head, &model.eos_head),
    };
    for (&model.model.layers, &weight_buffers.model.layers) |*layer, *buffers| {
        buffers.* = try zml.mem.bufferize(allocator, @TypeOf(layer.*), layer);
    }
    for (&model.flow_head.timestep_embeddings, &weight_buffers.flow_head.timestep_embeddings) |*embedding, *buffers| {
        buffers.* = try zml.mem.bufferize(allocator, @TypeOf(embedding.*), embedding);
    }
    for (&model.flow_head.layers, &weight_buffers.flow_head.layers) |*block, *buffers| {
        buffers.* = try zml.mem.bufferize(allocator, @TypeOf(block.*), block);
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

    var length_buffer = try zml.Buffer.scalar(io, platform, ids_shape.dim(.s), .i32);
    defer length_buffer.deinit();
    var frame_buffer = try zml.Buffer.uninitialized(io, platform, candidate_shape, .{});
    defer frame_buffer.deinit();

    var exe = try platform.compile(allocator, io, model, .forward_uncached, .{
        zml.Tensor.fromShape(ids_shape),
        zml.Tensor.fromShape(voice_shape),
        zml.Tensor.fromShape(audio_shape),
        zml.Tensor.fromShape(candidate_shape),
        zml.Tensor.fromShape(time_shape),
        zml.Tensor.fromShape(time_shape),
        zml.Tensor.fromShape(.scalar(.i32)),
        zml.Tensor.fromShape(candidate_shape),
    }, .{});
    defer exe.deinit();
    var runner = try exe.runner(allocator);
    defer runner.deinit(allocator);
    var output: zml.Bufferized(Gradium_Phonon_TTS_For_Conditional_Generation.Output) = undefined;
    runner.run(io, .{ weight_buffers, ids_buffer, voice_buffer, audio_buffer, candidate_buffer, start_buffer, end_buffer, length_buffer, frame_buffer }, .{&output}, .{ .wait = true });
    // The runner consumed the donated handles; retain their replacements.
    length_buffer = output.active_length;
    frame_buffer = output.frame;
    defer output.eos_score.deinit();
    try std.testing.expectEqual(@as(usize, 2), exe.donated_input_indices.len);
    try std.testing.expectEqual(@as(i32, @intCast(ids_shape.dim(.s) + 1)), try length_buffer.getValue(i32, io));

    const expected_eos = test_data.eos_score;
    var expected_frame: [2][32]f32 = undefined;
    for (&expected_frame, candidate, test_data.velocity) |*frame, noise, velocity| {
        for (frame, noise, velocity) |*value, initial, update| value.* = initial + update;
    }
    try zml.testing.expectClose(io, zml.Slice.init(.init(.{ .b = 2, .dout = 1 }, .f32), std.mem.asBytes(&expected_eos)), output.eos_score, .{
        .absolute_tolerance = 1e-4,
        .relative_tolerance = 1e-4,
        .minimum_close_fraction = 1,
    });
    // The synthetic flow produces values in the thousands, including near-zero cancellations.
    try zml.testing.expectClose(io, zml.Slice.init(candidate_shape, std.mem.asBytes(&expected_frame)), output.frame, .{
        .absolute_tolerance = 1e-2,
        .relative_tolerance = 1e-4,
        .minimum_close_fraction = 1,
    });

    for ([_]zml.DataType{ .f16, .bf16, .f32, .f64 }) |dtype| {
        var typed_exe = try platform.compile(allocator, io, model, .forward_uncached, .{
            zml.Tensor.fromShape(ids_shape),
            zml.Tensor.fromShape(voice_shape.withDtype(dtype)),
            zml.Tensor.fromShape(audio_shape),
            zml.Tensor.fromShape(candidate_shape.withDtype(dtype)),
            zml.Tensor.fromShape(time_shape),
            zml.Tensor.fromShape(time_shape),
            zml.Tensor.fromShape(.scalar(.i32)),
            zml.Tensor.fromShape(candidate_shape.withDtype(dtype)),
        }, .{});
        defer typed_exe.deinit();
        try std.testing.expect(typed_exe.output_shapes[0].eqlWithTags(.init(.{ .b = 2, .dout = 1 }, dtype)));
        try std.testing.expect(typed_exe.output_shapes[1].eqlWithTags(.scalar(.i32)));
        try std.testing.expect(typed_exe.output_shapes[2].eqlWithTags(candidate_shape.withDtype(dtype)));
    }
    var mixed_exe = try platform.compile(allocator, io, model, .forward_uncached, .{
        zml.Tensor.fromShape(ids_shape),
        zml.Tensor.fromShape(voice_shape.withDtype(.f64)),
        zml.Tensor.fromShape(audio_shape.withDtype(.f16)),
        zml.Tensor.fromShape(candidate_shape.withDtype(.bf16)),
        zml.Tensor.fromShape(time_shape.withDtype(.f64)),
        zml.Tensor.fromShape(time_shape.withDtype(.f16)),
        zml.Tensor.fromShape(.scalar(.i32)),
        zml.Tensor.fromShape(candidate_shape.withDtype(.bf16)),
    }, .{});
    defer mixed_exe.deinit();
    try std.testing.expect(mixed_exe.output_shapes[0].eqlWithTags(.init(.{ .b = 2, .dout = 1 }, .f64)));
    try std.testing.expect(mixed_exe.output_shapes[2].eqlWithTags(candidate_shape.withDtype(.bf16)));
}

test "Gradium_Phonon_TTS_For_Conditional_Generation: cached" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    var rng: std.Random.DefaultPrng = .init(42);
    const random = rng.random();

    var body: transformer.Gradium_Phonon_TTS_Model = .{
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
    for (&body.layers) |*layer| {
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

    var head: flow_head.Gradium_Phonon_TTS_Flow_Head = .{
        .input_proj = .{
            .projection = .init(
                .fromShape(.init(.{ .d = 512, .ld = 32 }, .bf16)),
                .fromShape(.init(.{ .d = 512 }, .bf16)),
                .ld,
            ),
        },
        .cond_proj = .{
            .projection = .init(
                .fromShape(.init(.{ .dout = 512, .d = 512 }, .bf16)),
                .fromShape(.init(.{ .dout = 512 }, .bf16)),
                .d,
            ),
        },
        .timestep_embeddings = undefined,
        .layers = undefined,
        .final_layer = .{
            .adaln_proj = .init(
                .fromShape(.init(.{ .m = 1024, .d = 512 }, .bf16)),
                .fromShape(.init(.{ .m = 1024 }, .bf16)),
                .d,
            ),
            .proj = .init(
                .fromShape(.init(.{ .ld = 32, .d = 512 }, .bf16)),
                .fromShape(.init(.{ .ld = 32 }, .bf16)),
                .d,
            ),
        },
    };
    for (&head.timestep_embeddings) |*embedding| {
        embedding.* = .{
            .freqs = .fromShape(.init(.{ .f = 128 }, .bf16)),
            .fc1 = .init(
                .fromShape(.init(.{ .d = 512, .f = 256 }, .bf16)),
                .fromShape(.init(.{ .d = 512 }, .bf16)),
                .f,
            ),
            .fc2 = .init(
                .fromShape(.init(.{ .dout = 512, .d = 512 }, .bf16)),
                .fromShape(.init(.{ .dout = 512 }, .bf16)),
                .d,
            ),
            .norm_weight = .fromShape(.init(.{ .d = 512 }, .bf16)),
        };
    }
    for (&head.layers) |*block| {
        block.* = .{
            .norm_weight = .fromShape(.init(.{ .d = 512 }, .bf16)),
            .norm_bias = .fromShape(.init(.{ .d = 512 }, .bf16)),
            .fc1 = .init(
                .fromShape(.init(.{ .dout = 512, .d = 512 }, .bf16)),
                .fromShape(.init(.{ .dout = 512 }, .bf16)),
                .d,
            ),
            .fc2 = .init(
                .fromShape(.init(.{ .dout = 512, .d = 512 }, .bf16)),
                .fromShape(.init(.{ .dout = 512 }, .bf16)),
                .d,
            ),
            .adaln_proj = .init(
                .fromShape(.init(.{ .m = 1536, .d = 512 }, .bf16)),
                .fromShape(.init(.{ .m = 1536 }, .bf16)),
                .d,
            ),
        };
    }

    const model: Gradium_Phonon_TTS_For_Conditional_Generation = .{
        .model = body,
        .flow_head = head,
        .eos_head = .{ .projection = .init(
            .fromShape(.init(.{ .dout = 1, .d = 512 }, .bf16)),
            .fromShape(.init(.{ .dout = 1 }, .bf16)),
            .d,
        ) },
    };

    // Each batch row has two voice markers, two text tokens, BOS, then two audio markers.
    var ids: [2][7]i64 = .{ .{ 5, 7, 5, 7, 4, 6, 6 }, .{ 7, 5, 7, 5, 4, 6, 6 } };
    for (&ids) |*batch| {
        for (batch) |*id| {
            if (id.* == 7) id.* = random.intRangeLessThan(i64, 7, 4001);
        }
    }

    const voice_shape: zml.Shape = .init(.{ .b = 2, .p = 2, .ad = 512 }, .f32);
    var voice: [2][2][512]f32 = undefined;
    for (&voice) |*batch| {
        for (batch) |*frame| {
            for (frame) |*value| value.* = random.float(f32) * 2 - 1;
        }
    }
    var voice_buffer = try zml.Buffer.fromBytes(io, platform, voice_shape, std.mem.asBytes(&voice));
    defer voice_buffer.deinit();

    var audio: [2][2][32]f32 = undefined;
    for (&audio) |*batch| {
        for (batch) |*frame| {
            for (frame) |*value| value.* = random.float(f32) * 2 - 1;
        }
    }

    const candidate_shape: zml.Shape = .init(.{ .b = 2, .ld = 32 }, .f32);
    var candidate: [2][32]f32 = undefined;
    for (&candidate) |*batch| {
        for (batch) |*value| value.* = random.float(f32) * 2 - 1;
    }
    var candidate_buffer = try zml.Buffer.fromBytes(io, platform, candidate_shape, std.mem.asBytes(&candidate));
    defer candidate_buffer.deinit();
    const time_shape: zml.Shape = .init(.{ .b = 2, .t = 1 }, .f32);
    var start: [2][1]f32 = undefined;
    var end: [2][1]f32 = undefined;
    for (&start, &end) |*from, *to| {
        from[0] = random.float(f32);
        to[0] = from[0] + random.float(f32) * (1 - from[0]);
    }
    var start_buffer = try zml.Buffer.fromBytes(io, platform, time_shape, std.mem.asBytes(&start));
    defer start_buffer.deinit();
    var end_buffer = try zml.Buffer.fromBytes(io, platform, time_shape, std.mem.asBytes(&end));
    defer end_buffer.deinit();

    // Bufferize each child and its array elements, as in the production loader.
    var weight_buffers: zml.Bufferized(Gradium_Phonon_TTS_For_Conditional_Generation) = .{
        .model = .{
            .embed_tokens = try zml.mem.bufferize(allocator, @TypeOf(model.model.embed_tokens), &model.model.embed_tokens),
            .input_projection = try zml.mem.bufferize(allocator, @TypeOf(model.model.input_projection), &model.model.input_projection),
            .norm = try zml.mem.bufferize(allocator, zml.nn.LayerNorm, &model.model.norm),
            .layers = undefined,
        },
        .flow_head = .{
            .input_proj = try zml.mem.bufferize(allocator, @TypeOf(model.flow_head.input_proj), &model.flow_head.input_proj),
            .cond_proj = try zml.mem.bufferize(allocator, @TypeOf(model.flow_head.cond_proj), &model.flow_head.cond_proj),
            .final_layer = try zml.mem.bufferize(allocator, @TypeOf(model.flow_head.final_layer), &model.flow_head.final_layer),
            .timestep_embeddings = undefined,
            .layers = undefined,
        },
        .eos_head = try zml.mem.bufferize(allocator, eos_head.EOS_Head, &model.eos_head),
    };
    for (&model.model.layers, &weight_buffers.model.layers) |*layer, *buffers| {
        buffers.* = try zml.mem.bufferize(allocator, @TypeOf(layer.*), layer);
    }
    for (&model.flow_head.timestep_embeddings, &weight_buffers.flow_head.timestep_embeddings) |*embedding, *buffers| {
        buffers.* = try zml.mem.bufferize(allocator, @TypeOf(embedding.*), embedding);
    }
    for (&model.flow_head.layers, &weight_buffers.flow_head.layers) |*block, *buffers| {
        buffers.* = try zml.mem.bufferize(allocator, @TypeOf(block.*), block);
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

    const initial_cache = transformer.KV_Cache.init(2, 9);
    var cache_buffers = try initial_cache.initBuffer(allocator, io, platform);
    defer transformer.KV_Cache.deinitBuffer(&cache_buffers);
    const prompt_shape: zml.Shape = .init(.{ .b = 2, .s = 5 }, .i64);
    var prefill_exe = try platform.compile(allocator, io, model, .forward_cached_prefill, .{
        zml.Tensor.fromShape(prompt_shape),
        zml.Tensor.fromShape(voice_shape),
        zml.Tensor.fromShape(candidate_shape),
        zml.Tensor.fromShape(time_shape),
        zml.Tensor.fromShape(time_shape),
        initial_cache,
    }, .{});
    defer prefill_exe.deinit();
    var prefill_runner = try prefill_exe.runner(allocator);
    defer prefill_runner.deinit(allocator);
    const frame_shape: zml.Shape = .init(.{ .b = 2, .a = 1, .ld = 32 }, .f32);
    var decode_exe = try platform.compile(allocator, io, model, .forward_cached_decode, .{
        zml.Tensor.fromShape(frame_shape),
        zml.Tensor.fromShape(.scalar(.i32)),
        zml.Tensor.fromShape(candidate_shape),
        zml.Tensor.fromShape(time_shape),
        zml.Tensor.fromShape(time_shape),
        initial_cache,
    }, .{});
    defer decode_exe.deinit();
    var decode_runner = try decode_exe.runner(allocator);
    defer decode_runner.deinit(allocator);

    // Compare prefill and two successive decode predictions with their full prefixes.
    inline for (0..3) |step| {
        const prefix_shape: zml.Shape = .init(.{ .b = 2, .s = 5 + step }, .i64);
        var prefix: [2][5 + step]i64 = undefined;
        for (&prefix, ids) |*batch, source| @memcpy(batch, source[0 .. 5 + step]);
        var prefix_buffer = try zml.Buffer.fromBytes(io, platform, prefix_shape, std.mem.asBytes(&prefix));
        defer prefix_buffer.deinit();
        const history_shape: zml.Shape = .init(.{ .b = 2, .a = step, .ld = 32 }, .f32);
        var history: [2][step][32]f32 = undefined;
        for (&history, audio) |*batch, source| @memcpy(batch, source[0..step]);
        var history_buffer = try zml.Buffer.fromBytes(io, platform, history_shape, std.mem.asBytes(&history));
        defer history_buffer.deinit();

        var length_buffer = try zml.Buffer.scalar(io, platform, prefix_shape.dim(.s), .i32);
        defer length_buffer.deinit();
        var output_frame_buffer = try zml.Buffer.uninitialized(io, platform, candidate_shape, .{});
        defer output_frame_buffer.deinit();

        var uncached_exe = try platform.compile(allocator, io, model, .forward_uncached, .{
            zml.Tensor.fromShape(prefix_shape),
            zml.Tensor.fromShape(voice_shape),
            zml.Tensor.fromShape(history_shape),
            zml.Tensor.fromShape(candidate_shape),
            zml.Tensor.fromShape(time_shape),
            zml.Tensor.fromShape(time_shape),
            zml.Tensor.fromShape(.scalar(.i32)),
            zml.Tensor.fromShape(candidate_shape),
        }, .{});
        defer uncached_exe.deinit();
        var uncached_runner = try uncached_exe.runner(allocator);
        defer uncached_runner.deinit(allocator);
        var uncached_output: zml.Bufferized(Gradium_Phonon_TTS_For_Conditional_Generation.Output) = undefined;
        uncached_runner.run(io, .{ weight_buffers, prefix_buffer, voice_buffer, history_buffer, candidate_buffer, start_buffer, end_buffer, length_buffer, output_frame_buffer }, .{&uncached_output}, .{ .wait = true });
        length_buffer = uncached_output.active_length;
        output_frame_buffer = uncached_output.frame;
        defer uncached_output.eos_score.deinit();
        try std.testing.expectEqual(@as(i32, @intCast(prefix_shape.dim(.s) + 1)), try length_buffer.getValue(i32, io));

        var cached_output: zml.Bufferized(Gradium_Phonon_TTS_For_Conditional_Generation.Prediction_Output) = undefined;
        if (step == 0) {
            prefill_runner.run(io, .{ weight_buffers, prefix_buffer, voice_buffer, candidate_buffer, start_buffer, end_buffer, cache_buffers }, .{ &cached_output, &cache_buffers }, .{ .wait = true });
        } else {
            var frame: [2][1][32]f32 = undefined;
            for (&frame, audio) |*batch, source| batch[0] = source[step - 1];
            var frame_buffer = try zml.Buffer.fromBytes(io, platform, frame_shape, std.mem.asBytes(&frame));
            defer frame_buffer.deinit();
            const position: i32 = 4 + step;
            var position_buffer = try zml.Buffer.fromBytes(io, platform, .scalar(.i32), std.mem.asBytes(&position));
            defer position_buffer.deinit();
            decode_runner.run(io, .{ weight_buffers, frame_buffer, position_buffer, candidate_buffer, start_buffer, end_buffer, cache_buffers }, .{ &cached_output, &cache_buffers }, .{ .wait = true });
        }
        defer zml.Buffer.deinitAll(Gradium_Phonon_TTS_For_Conditional_Generation.Prediction_Output, &cached_output);

        // Cached and full-prefix matmul/reduction shapes can round differently.
        try zml.testing.expectClose(io, uncached_output.eos_score, cached_output.eos_score, .{
            .absolute_tolerance = 1e-3,
            .relative_tolerance = 1e-4,
            .minimum_close_fraction = 1,
        });
        // The flow head amplifies small transformer differences in these random weights.
        const cached_velocity = try cached_output.velocity.getValue([2][32]f32, io);
        var expected_frame: [2][32]f32 = undefined;
        for (&expected_frame, candidate, cached_velocity) |*frame, noise, velocity| {
            for (frame, noise, velocity) |*value, initial, update| value.* = initial + update;
        }
        try zml.testing.expectClose(io, uncached_output.frame, zml.Slice.init(candidate_shape, std.mem.asBytes(&expected_frame)), .{
            .absolute_tolerance = 1e-2,
            .relative_tolerance = 1e-4,
            .minimum_close_fraction = 1,
        });
    }

    for ([_]zml.DataType{ .f16, .bf16, .f32, .f64 }) |dtype| {
        const mixed_cache: transformer.KV_Cache = .{
            .k = .fromShape(initial_cache.k.shape().withDtype(.f16)),
            .v = .fromShape(initial_cache.v.shape().withDtype(.bf16)),
        };
        const candidate_dtype: zml.DataType = if (dtype == .f64) .bf16 else dtype;
        var typed_prefill = try platform.compile(allocator, io, model, .forward_cached_prefill, .{
            zml.Tensor.fromShape(prompt_shape),
            zml.Tensor.fromShape(voice_shape.withDtype(dtype)),
            zml.Tensor.fromShape(candidate_shape.withDtype(candidate_dtype)),
            zml.Tensor.fromShape(time_shape),
            zml.Tensor.fromShape(time_shape),
            mixed_cache,
        }, .{});
        defer typed_prefill.deinit();
        try std.testing.expect(typed_prefill.output_shapes[0].eqlWithTags(.init(.{ .b = 2, .dout = 1 }, dtype)));
        try std.testing.expect(typed_prefill.output_shapes[1].eqlWithTags(candidate_shape.withDtype(candidate_dtype)));
        try std.testing.expect(typed_prefill.output_shapes[2].eqlWithTags(mixed_cache.k.shape()));
        try std.testing.expect(typed_prefill.output_shapes[3].eqlWithTags(mixed_cache.v.shape()));

        var typed_decode = try platform.compile(allocator, io, model, .forward_cached_decode, .{
            zml.Tensor.fromShape(frame_shape.withDtype(dtype)),
            zml.Tensor.fromShape(.scalar(.i32)),
            zml.Tensor.fromShape(candidate_shape.withDtype(dtype)),
            zml.Tensor.fromShape(time_shape),
            zml.Tensor.fromShape(time_shape),
            mixed_cache,
        }, .{});
        defer typed_decode.deinit();
        try std.testing.expect(typed_decode.output_shapes[0].eqlWithTags(.init(.{ .b = 2, .dout = 1 }, dtype)));
        try std.testing.expect(typed_decode.output_shapes[1].eqlWithTags(candidate_shape.withDtype(dtype)));
        try std.testing.expect(typed_decode.output_shapes[2].eqlWithTags(mixed_cache.k.shape()));
        try std.testing.expect(typed_decode.output_shapes[3].eqlWithTags(mixed_cache.v.shape()));
    }
}
