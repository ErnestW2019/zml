const std = @import("std");
const zml = @import("zml");
const test_data = @import("transformer_test_data.zig");

test {
    std.testing.refAllDecls(@This());
}

/// Fixed-capacity attention history, [2, B, C, 8, 64], tagged .layer/.b/.k/.h/.hd.
/// K already includes RoPE. The caller zero-initializes both buffers, supplies an
/// absolute write position, and ensures position+chunk_length <= C (the fixed cache capacity). Each retains its dtype.
/// The attention mask keeps only the current position and 250 predecessors.
pub const KV_Cache = struct {
    k: zml.Tensor,
    v: zml.Tensor,

    pub fn init(batch: i64, capacity: i64, dtype: zml.DataType) KV_Cache {
        std.debug.assert(batch > 0 and capacity > 0 and dtype.isFloat());
        const shape: zml.Shape = .init(.{ .layer = 2, .b = batch, .k = capacity, .h = 8, .hd = 64 }, dtype);
        return .{ .k = .fromShape(shape), .v = .fromShape(shape) };
    }
};

pub const Cached_Output = struct { hidden: zml.Tensor, cache: KV_Cache };

/// Feature-wise LayerNorm used before attention and before the MLP, epsilon 1e-5.
/// Checkpoint vectors are [512]. ZML computes normalization statistics in f32.
pub const Layer_Norm = struct {
    weight: zml.Tensor,
    bias: zml.Tensor,

    pub fn init(store: zml.io.TensorStore.View) Layer_Norm {
        const here = @typeName(@This()) ++ ".init";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        return .{ .weight = store.createReplicatedTensor("weight", .{.d}), .bias = store.createReplicatedTensor("bias", .{.d}) };
    }

    /// `input`: floating [B, S, 512], residual features from the upsampler or the
    /// preceding transformer branch. Returns normalized features of the same
    /// shape and dtype. Tags .b independent sequences, .s positions, .d features.
    pub fn forward(self: Layer_Norm, input: zml.Tensor) zml.Tensor {
        const here = @typeName(@This()) ++ ".forward";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        std.debug.assert(input.dtype().isFloat());
        const norm: zml.nn.LayerNorm = .{ .weight = self.weight.convert(input.dtype()), .bias = self.bias.convert(input.dtype()), .eps = 1e-5 };
        return norm.forward(input.withTags(.{ .b, .s, .d }));
    }
};

/// Learned per-feature multiplier on each transformer branch before residual addition.
pub const Layer_Scale = struct {
    scale: zml.Tensor,
    pub fn init(store: zml.io.TensorStore.View) Layer_Scale {
        const here = @typeName(@This()) ++ ".init";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        return .{ .scale = store.createReplicatedTensor("scale", .{.d}) };
    }

    /// `input`: floating [B, S, 512], projected attention or MLP output.
    /// Returns the same shape/dtype scaled by checkpoint [512] parameters.
    /// Tags .b batch, .s positions, .d features; only .d selects the multiplier.
    pub fn forward(self: Layer_Scale, input: zml.Tensor) zml.Tensor {
        const here = @typeName(@This()) ++ ".forward";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        std.debug.assert(input.dtype().isFloat());
        const x = input.withTags(.{ .b, .s, .d });
        return x.mul(self.scale.convert(x.dtype()).broad(x.shape()));
    }
};

/// Bias-free 512 -> 2048 -> 512 feed-forward branch. gradium/phonon-mimi uses
/// exact GELU (erf), not the tanh approximation in the library's default config.
pub const MLP = struct {
    fc1: zml.nn.Linear,
    fc2: zml.nn.Linear,
    pub fn init(store: zml.io.TensorStore.View) MLP {
        const here = @typeName(@This()) ++ ".init";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        return .{
            .fc1 = .init(store.createReplicatedTensor("fc1.weight", .{ .ff, .d }), null, .d),
            .fc2 = .init(store.createReplicatedTensor("fc2.weight", .{ .d, .ff }), null, .ff),
        };
    }

    /// `input`: floating [B, S, 512], post-attention normalized residual features.
    /// Returns [B, S, 512] in input.dtype(), before LayerScale and residual addition.
    /// Tags .b batch, .s positions, .d model features, .ff intermediate features.
    /// erf is evaluated in f32 internally, then converted back to the input dtype.
    pub fn forward(self: MLP, input: zml.Tensor) zml.Tensor {
        const here = @typeName(@This()) ++ ".forward";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        std.debug.assert(input.dtype().isFloat());
        const features = input.withTags(.{ .b, .s, .d });
        const expanded = self.fc1.forward(features, features.dtype());
        const activated = gelu_erf(expanded);
        return self.fc2.forward(activated, features.dtype());
    }

    /// Exact GELU on [B,.s,.ff=2048]. erf runs in f32 and is converted back.
    fn gelu_erf(features: zml.Tensor) zml.Tensor {
        const erf_term = features.scale(1.0 / @sqrt(2.0)).erf().convert(features.dtype());
        return features.scale(0.5).mul(erf_term.addConstant(1));
    }
};

/// Eight-head causal attention with split-half RoPE (theta 10000) and a 251-slot
/// window. Separate Q/K/V checkpoint projections already use the HF head layout.
/// Parameters and selected cache history convert to the input floating dtype;
/// ZML's RoPE trigonometry and softmax use f32 internally.
pub const Attention = struct {
    q_proj: zml.nn.Linear,
    k_proj: zml.nn.Linear,
    v_proj: zml.nn.Linear,
    o_proj: zml.nn.Linear,

    pub fn init(store: zml.io.TensorStore.View) Attention {
        const here = @typeName(@This()) ++ ".init";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        return .{
            .q_proj = .init(store.createReplicatedTensor("q_proj.weight", .{ .out, .d }), null, .d),
            .k_proj = .init(store.createReplicatedTensor("k_proj.weight", .{ .out, .d }), null, .d),
            .v_proj = .init(store.createReplicatedTensor("v_proj.weight", .{ .out, .d }), null, .d),
            .o_proj = .init(store.createReplicatedTensor("o_proj.weight", .{ .out, .d }), null, .d),
        };
    }

    /// `input`: floating [B, S, 512], pre-attention normalized decoder features.
    /// Positions start at zero. Returns [B, S, 512] in input.dtype(), before the
    /// caller's LayerScale/residual. Tags .b batch, .s time, .d features; internally
    /// .q queries, .k keys, .h eight heads, .hd 64 coordinates per head.
    pub fn forward_uncached(self: Attention, input: zml.Tensor) zml.Tensor {
        const here = @typeName(@This()) ++ ".forward_uncached";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        const hidden = input.withTags(.{ .b, .s, .d });
        const positions = zml.Tensor.arange(.{ .end = hidden.dim(.s) }, .i32).withTags(.{.s});
        const projected = self.project_qkv(hidden);
        const rotated = apply_rope(projected, positions);
        const mask = sliding_window_mask(positions, hidden.dim(.s), hidden.dtype());
        return self.attend_and_project(rotated.queries, rotated.keys, rotated.values, mask);
    }

    /// `input`: floating [B, chunk_length, 512], only the new normalized features
    /// from Transformer_Layer. One new latent supplies 16 transformer positions.
    /// `position`: scalar i32 absolute index of the first new position.
    /// `cache`: zero-initialized/previously returned K/V [2,B,capacity,8,64].
    /// `layer_index`: host index 0 or 1. Caller ensures position >= 0 and
    /// position+chunk_length <= capacity. Earlier history must be populated.
    /// Returns [B,chunk_length,512] features and updated history. Hidden follows
    /// input.dtype(); K/V retain their storage dtypes. Tags .b/.s/.d on input,
    /// .q for new queries, .k for capacity, .layer/.h/.hd for history.
    pub fn forward_cached(self: Attention, input: zml.Tensor, position: zml.Tensor, cache: KV_Cache, layer_index: usize) Cached_Output {
        const here = @typeName(@This()) ++ ".forward_cached";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        const hidden = input.withTags(.{ .b, .s, .d });
        const chunk_length = hidden.dim(.s);
        const capacity = cache.k.dim(.k);
        std.debug.assert(position.rank() == 0 and position.dtype() == .i32 and layer_index < 2);
        std.debug.assert(cache.k.dtype().isFloat() and cache.v.dtype().isFloat());
        std.debug.assert(hidden.dim(.b) == cache.k.dim(.b) and chunk_length <= capacity);

        const positions = absolute_positions(position, chunk_length);
        const projected = self.project_qkv(hidden);
        const rotated = apply_rope(projected, positions);
        const updated_cache = update_cache(cache, layer_index, position, rotated.keys, rotated.values);
        const history = read_cached_layer(updated_cache, layer_index, hidden.dtype());
        const mask = sliding_window_mask(positions, capacity, hidden.dtype());
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

    /// Separate checkpoint projections, each [B,.s,512] -> [B,.s,.h=8,.hd=64].
    fn project_qkv(self: Attention, hidden: zml.Tensor) QKV {
        std.debug.assert(hidden.dtype().isFloat() and hidden.dim(.d) == 512);
        return .{
            .queries = self.q_proj.forward(hidden, hidden.dtype()).splitAxis(.out, .{ .h = 8, .hd = 64 }),
            .keys = self.k_proj.forward(hidden, hidden.dtype()).splitAxis(.out, .{ .h = 8, .hd = 64 }),
            .values = self.v_proj.forward(hidden, hidden.dtype()).splitAxis(.out, .{ .h = 8, .hd = 64 }),
        };
    }

    /// Rotate Q/K at absolute i32 positions [.s]. Mimi's separate checkpoint
    /// weights use split-half real/imaginary coordinates. Values are not rotated.
    fn apply_rope(projected: QKV, positions: zml.Tensor) QKV {
        const options: zml.nn.RopeOpts = .{
            .layout = .real_im_pass,
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
    /// Preserve other layers, capacity, and storage dtypes. The caller manages
    /// buffer ownership; this helper describes only the tensor update.
    fn update_cache(cache: KV_Cache, layer_index: usize, position: zml.Tensor, keys: zml.Tensor, values: zml.Tensor) KV_Cache {
        const layer = zml.Tensor.scalar(@as(i32, @intCast(layer_index)), .i32);
        return .{
            .k = cache.k.dynamicUpdateSlice(.{ .layer = layer, .k = position }, keys.convert(cache.k.dtype())),
            .v = cache.v.dynamicUpdateSlice(.{ .layer = layer, .k = position }, values.convert(cache.v.dtype())),
        };
    }

    const Cached_Layer = struct { keys: zml.Tensor, values: zml.Tensor };

    /// Select one layer's full [B, capacity, 8, 64] history and convert it to the
    /// compute dtype. The attention mask, rather than this read, excludes invisible slots.
    fn read_cached_layer(cache: KV_Cache, layer_index: usize, dtype: zml.DataType) Cached_Layer {
        const start: i64 = @intCast(layer_index);
        return .{
            .keys = cache.k.slice(.layer, .{ .start = start, .end = start + 1 }).squeeze(.layer).convert(dtype),
            .values = cache.v.slice(.layer, .{ .start = start, .end = start + 1 }).squeeze(.layer).convert(dtype),
        };
    }

    /// Absolute query positions [.s] -> additive [.q,.k] mask. Each query
    /// sees itself and its 250 predecessors; future and older positions are blocked.
    fn sliding_window_mask(positions: zml.Tensor, capacity: i64, dtype: zml.DataType) zml.Tensor {
        const shape: zml.Shape = .init(.{ .q = positions.dim(.s), .k = capacity }, .i32);
        const queries = positions.rename(.{ .s = .q }).broad(shape);
        const keys = zml.Tensor.iota(shape, .k);
        const distance = queries.sub(keys);
        const past_or_present = distance.cmp(.GE, zml.Tensor.scalar(0, .i32));
        const within_window = distance.cmp(.LT, zml.Tensor.scalar(251, .i32));
        const visible = past_or_present.logical(.AND, within_window);
        return visible.select(zml.Tensor.scalar(0, dtype), zml.Tensor.scalar(-std.math.inf(f32), dtype));
    }

    /// Attend from [B, new_positions, 8, 64] queries to keys/values, then merge
    /// heads and project to [B, new_positions, 512], tagged .b/.s/.d.
    /// .out distinguishes the output projection's result from its contracted .d axis.
    fn attend_and_project(self: Attention, queries: zml.Tensor, keys: zml.Tensor, values: zml.Tensor, mask: zml.Tensor) zml.Tensor {
        const attended = zml.nn.sdpa(queries, keys, values, .{ .attn_mask = mask });
        const features = attended.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s });
        return self.o_proj.forward(features, queries.dtype()).rename(.{ .out = .d });
    }
};

/// One pre-norm transformer layer: attention and MLP each have their own learned
/// LayerScale and residual connection. Parameters come from decoder_transformer.layers.N.
pub const Transformer_Layer = struct {
    self_attn: Attention,
    mlp: MLP,
    input_layernorm: Layer_Norm,
    post_attention_layernorm: Layer_Norm,
    self_attn_layer_scale: Layer_Scale,
    mlp_layer_scale: Layer_Scale,

    pub fn init(store: zml.io.TensorStore.View) Transformer_Layer {
        const here = @typeName(@This()) ++ ".init";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        return .{
            .self_attn = .init(store.withPrefix("self_attn")),
            .mlp = .init(store.withPrefix("mlp")),
            .input_layernorm = .init(store.withPrefix("input_layernorm")),
            .post_attention_layernorm = .init(store.withPrefix("post_attention_layernorm")),
            .self_attn_layer_scale = .init(store.withPrefix("self_attn_layer_scale")),
            .mlp_layer_scale = .init(store.withPrefix("mlp_layer_scale")),
        };
    }

    /// `input`: floating [B, S, 512], upsampler output or the preceding layer's
    /// residual features. Returns the same shape/dtype, tags .b batch/.s time/.d features.
    pub fn forward_uncached(self: Transformer_Layer, input: zml.Tensor) zml.Tensor {
        const here = @typeName(@This()) ++ ".forward_uncached";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        const hidden = input.withTags(.{ .b, .s, .d });
        const normalized = self.input_layernorm.forward(hidden);
        const attention = self.self_attn.forward_uncached(normalized);
        const residual = hidden.add(self.self_attn_layer_scale.forward(attention));
        return self.add_mlp_residual(residual);
    }

    /// `input`: floating [B,chunk_length,512], features for only the new positions;
    /// `position`: scalar i32 absolute first index; `cache`: K/V [2,B,C,8,64],
    /// with C the fixed capacity; `layer_index`: 0 or 1.
    /// Caller supplies zero-initialized history then carries returned history
    /// forward, within capacity. Hidden and cache tensors preserve their dtypes.
    pub fn forward_cached(self: Transformer_Layer, input: zml.Tensor, position: zml.Tensor, cache: KV_Cache, layer_index: usize) Cached_Output {
        const here = @typeName(@This()) ++ ".forward_cached";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        const hidden = input.withTags(.{ .b, .s, .d });
        const normalized = self.input_layernorm.forward(hidden);
        const attention = self.self_attn.forward_cached(normalized, position, cache, layer_index);
        const residual = hidden.add(self.self_attn_layer_scale.forward(attention.hidden));
        return .{ .hidden = self.add_mlp_residual(residual), .cache = attention.cache };
    }

    /// Normalize, transform, and scale the MLP branch before adding [B,.s,.d=512] residual.
    fn add_mlp_residual(self: Transformer_Layer, residual: zml.Tensor) zml.Tensor {
        const normalized = self.post_attention_layernorm.forward(residual);
        const update = self.mlp.forward(normalized);
        return residual.add(self.mlp_layer_scale.forward(update));
    }
};

/// The two decoder transformer layers. There is no final LayerNorm.
pub const Transformer = struct {
    layers: [2]Transformer_Layer,
    pub fn init(store: zml.io.TensorStore.View) Transformer {
        const here = @typeName(@This()) ++ ".init";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        return .{ .layers = .{ .init(store.withPrefix("layers.0")), .init(store.withPrefix("layers.1")) } };
    }

    // ZML bufferize does not descend fixed arrays; initialize each layer explicitly.
    pub fn bufferize(self: Transformer, allocator: std.mem.Allocator) !zml.Bufferized(Transformer) {
        const here = @typeName(@This()) ++ ".bufferize";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        var buffers: zml.Bufferized(Transformer) = undefined;
        for (&self.layers, &buffers.layers) |*layer, *target| target.* = try zml.mem.bufferize(allocator, Transformer_Layer, layer);
        return buffers;
    }

    /// `input`: floating [B, S, 512], upsampled latent projection transposed to
    /// sequence-first layout. Returns contextual features for SEANet, same shape
    /// and dtype, tagged .b batch/.s positions/.d features.
    pub fn forward_uncached(self: Transformer, input: zml.Tensor) zml.Tensor {
        const here = @typeName(@This()) ++ ".forward_uncached";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        var hidden = input;
        for (self.layers) |layer| hidden = layer.forward_uncached(hidden);
        return hidden;
    }

    /// `input`: floating [B,chunk_length,512], only the new upsampled features;
    /// `position`: scalar i32
    /// count of previously processed upsampled positions (16 per latent frame).
    /// `cache`: K/V [2,B,C,8,64], initially zero, then returned history. Requires
    /// position+chunk_length <= C (the fixed cache capacity). Returns new features and both layers' updated history,
    /// each tensor preserving its input dtype. Tags .b/.s/.d and .layer/.k/.h/.hd.
    pub fn forward_cached(self: Transformer, input: zml.Tensor, position: zml.Tensor, cache: KV_Cache) Cached_Output {
        const here = @typeName(@This()) ++ ".forward_cached";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        var result: Cached_Output = .{ .hidden = input, .cache = cache };
        for (self.layers, 0..) |layer, index| result = layer.forward_cached(result.hidden, position, result.cache, index);
        return result;
    }
};

test "Layer_Norm" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    var rng: std.Random.DefaultPrng = .init(42);
    const random = rng.random();
    const layer: Layer_Norm = .{
        .weight = .fromShape(.init(.{ .d = 512 }, .f32)),
        .bias = .fromShape(.init(.{ .d = 512 }, .f32)),
    };
    const input_shape: zml.Shape = .init(.{ .b = 2, .s = 3, .d = 512 }, .f32);
    var input: [2][3][512]f32 = undefined;
    for (std.mem.bytesAsSlice(f32, std.mem.asBytes(&input))) |*value| value.* = random.float(f32) * 2 - 1;
    var input_buffer = try zml.Buffer.fromBytes(io, platform, input_shape, std.mem.asBytes(&input));
    defer input_buffer.deinit();
    var weight_buffers = try zml.mem.bufferize(allocator, Layer_Norm, &layer);
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
    var exe = try platform.compile(allocator, io, layer, .forward, .{zml.Tensor.fromShape(input_shape)}, .{});
    defer exe.deinit();
    var output = try exe.eval(allocator, io, .{ weight_buffers, input_buffer });
    defer output.deinit();
    const output_shape: zml.Shape = .init(.{ .b = 2, .s = 3, .d = 512 }, .f32);
    try std.testing.expect(output.shape().eqlWithTags(output_shape));
    const expected = test_data.layer_norm;
    try zml.testing.expectClose(io, zml.Slice.init(output_shape, std.mem.asBytes(&expected)), output, .{
        .absolute_tolerance = 1e-5,
        .relative_tolerance = 1e-4,
        .minimum_close_fraction = 1,
    });
    for ([_]zml.DataType{ .f16, .bf16, .f64 }) |dtype| {
        var typed_exe = try platform.compile(allocator, io, layer, .forward, .{zml.Tensor.fromShape(input_shape.withDtype(dtype))}, .{});
        defer typed_exe.deinit();
        try std.testing.expect(typed_exe.output_shapes[0].eqlWithTags(output_shape.withDtype(dtype)));
    }
}

test "Layer_Scale" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    var rng: std.Random.DefaultPrng = .init(42);
    const random = rng.random();
    const layer: Layer_Scale = .{
        .scale = .fromShape(.init(.{ .d = 512 }, .f32)),
    };
    const input_shape: zml.Shape = .init(.{ .b = 2, .s = 3, .d = 512 }, .f32);
    var input: [2][3][512]f32 = undefined;
    for (std.mem.bytesAsSlice(f32, std.mem.asBytes(&input))) |*value| value.* = random.float(f32) * 2 - 1;
    var input_buffer = try zml.Buffer.fromBytes(io, platform, input_shape, std.mem.asBytes(&input));
    defer input_buffer.deinit();
    var weight_buffers = try zml.mem.bufferize(allocator, Layer_Scale, &layer);
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
    var exe = try platform.compile(allocator, io, layer, .forward, .{zml.Tensor.fromShape(input_shape)}, .{});
    defer exe.deinit();
    var output = try exe.eval(allocator, io, .{ weight_buffers, input_buffer });
    defer output.deinit();
    const output_shape: zml.Shape = .init(.{ .b = 2, .s = 3, .d = 512 }, .f32);
    try std.testing.expect(output.shape().eqlWithTags(output_shape));
    const expected = test_data.layer_scale;
    try zml.testing.expectClose(io, zml.Slice.init(output_shape, std.mem.asBytes(&expected)), output, .{
        .absolute_tolerance = 1e-5,
        .relative_tolerance = 1e-4,
        .minimum_close_fraction = 1,
    });
    for ([_]zml.DataType{ .f16, .bf16, .f64 }) |dtype| {
        var typed_exe = try platform.compile(allocator, io, layer, .forward, .{zml.Tensor.fromShape(input_shape.withDtype(dtype))}, .{});
        defer typed_exe.deinit();
        try std.testing.expect(typed_exe.output_shapes[0].eqlWithTags(output_shape.withDtype(dtype)));
    }
}

test "MLP" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    var rng: std.Random.DefaultPrng = .init(42);
    const random = rng.random();
    const layer: MLP = .{
        .fc1 = .init(.fromShape(.init(.{ .ff = 2048, .d = 512 }, .f32)), null, .d),
        .fc2 = .init(.fromShape(.init(.{ .d = 512, .ff = 2048 }, .f32)), null, .ff),
    };
    const input_shape: zml.Shape = .init(.{ .b = 2, .s = 3, .d = 512 }, .f32);
    var input: [2][3][512]f32 = undefined;
    for (std.mem.bytesAsSlice(f32, std.mem.asBytes(&input))) |*value| value.* = random.float(f32) * 2 - 1;
    var input_buffer = try zml.Buffer.fromBytes(io, platform, input_shape, std.mem.asBytes(&input));
    defer input_buffer.deinit();
    var weight_buffers = try zml.mem.bufferize(allocator, MLP, &layer);
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
    var exe = try platform.compile(allocator, io, layer, .forward, .{zml.Tensor.fromShape(input_shape)}, .{});
    defer exe.deinit();
    var output = try exe.eval(allocator, io, .{ weight_buffers, input_buffer });
    defer output.deinit();
    const output_shape: zml.Shape = .init(.{ .b = 2, .s = 3, .d = 512 }, .f32);
    try std.testing.expect(output.shape().eqlWithTags(output_shape));
    const expected = test_data.mlp;
    try zml.testing.expectClose(io, zml.Slice.init(output_shape, std.mem.asBytes(&expected)), output, .{
        .absolute_tolerance = 1e-5,
        .relative_tolerance = 1e-4,
        .minimum_close_fraction = 1,
    });
    for ([_]zml.DataType{ .f16, .bf16, .f64 }) |dtype| {
        var typed_exe = try platform.compile(allocator, io, layer, .forward, .{zml.Tensor.fromShape(input_shape.withDtype(dtype))}, .{});
        defer typed_exe.deinit();
        try std.testing.expect(typed_exe.output_shapes[0].eqlWithTags(output_shape.withDtype(dtype)));
    }
}

test "Attention" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    var rng: std.Random.DefaultPrng = .init(42);
    const random = rng.random();
    const layer: Attention = .{
        .q_proj = .init(.fromShape(.init(.{ .out = 512, .d = 512 }, .f32)), null, .d),
        .k_proj = .init(.fromShape(.init(.{ .out = 512, .d = 512 }, .f32)), null, .d),
        .v_proj = .init(.fromShape(.init(.{ .out = 512, .d = 512 }, .f32)), null, .d),
        .o_proj = .init(.fromShape(.init(.{ .out = 512, .d = 512 }, .f32)), null, .d),
    };
    const input_shape: zml.Shape = .init(.{ .b = 2, .s = 3, .d = 512 }, .f32);
    var input: [2][3][512]f32 = undefined;
    for (std.mem.bytesAsSlice(f32, std.mem.asBytes(&input))) |*value| value.* = random.float(f32) * 2 - 1;
    var input_buffer = try zml.Buffer.fromBytes(io, platform, input_shape, std.mem.asBytes(&input));
    defer input_buffer.deinit();
    var weight_buffers = try zml.mem.bufferize(allocator, Attention, &layer);
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
    const output_shape: zml.Shape = .init(.{ .b = 2, .s = 3, .d = 512 }, .f32);
    try std.testing.expect(output.shape().eqlWithTags(output_shape));
    const expected = test_data.attention;
    try zml.testing.expectClose(io, zml.Slice.init(output_shape, std.mem.asBytes(&expected)), output, .{
        .absolute_tolerance = 1e-5,
        .relative_tolerance = 1e-4,
        .minimum_close_fraction = 1,
    });
    for ([_]zml.DataType{ .f16, .bf16, .f64 }) |dtype| {
        var typed_exe = try platform.compile(allocator, io, layer, .forward_uncached, .{zml.Tensor.fromShape(input_shape.withDtype(dtype))}, .{});
        defer typed_exe.deinit();
        try std.testing.expect(typed_exe.output_shapes[0].eqlWithTags(output_shape.withDtype(dtype)));
        const cache: KV_Cache = .{
            .k = .fromShape(.init(.{ .layer = 2, .b = 2, .k = 5, .h = 8, .hd = 64 }, .f16)),
            .v = .fromShape(.init(.{ .layer = 2, .b = 2, .k = 5, .h = 8, .hd = 64 }, .bf16)),
        };
        var cached_exe = try platform.compile(allocator, io, layer, .forward_cached, .{
            zml.Tensor.fromShape(input_shape.withDtype(dtype)), zml.Tensor.fromShape(.scalar(.i32)), cache, @as(usize, 1),
        }, .{});
        defer cached_exe.deinit();
        try std.testing.expect(cached_exe.output_shapes[0].eqlWithTags(output_shape.withDtype(dtype)));
        try std.testing.expect(cached_exe.output_shapes[1].eqlWithTags(cache.k.shape()));
        try std.testing.expect(cached_exe.output_shapes[2].eqlWithTags(cache.v.shape()));
    }
}

test "Transformer_Layer" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    var rng: std.Random.DefaultPrng = .init(42);
    const random = rng.random();
    const layer: Transformer_Layer = .{
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
    };
    const input_shape: zml.Shape = .init(.{ .b = 2, .s = 3, .d = 512 }, .f32);
    var input: [2][3][512]f32 = undefined;
    for (std.mem.bytesAsSlice(f32, std.mem.asBytes(&input))) |*value| value.* = random.float(f32) * 2 - 1;
    var input_buffer = try zml.Buffer.fromBytes(io, platform, input_shape, std.mem.asBytes(&input));
    defer input_buffer.deinit();
    var weight_buffers = try zml.mem.bufferize(allocator, Transformer_Layer, &layer);
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
    const output_shape: zml.Shape = .init(.{ .b = 2, .s = 3, .d = 512 }, .f32);
    try std.testing.expect(output.shape().eqlWithTags(output_shape));
    const expected = test_data.transformer_layer;
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

test "Transformer" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    var rng: std.Random.DefaultPrng = .init(42);
    const random = rng.random();
    const layer: Transformer = .{
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
    };
    const input_shape: zml.Shape = .init(.{ .b = 2, .s = 3, .d = 512 }, .f32);
    var input: [2][3][512]f32 = undefined;
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
    const output_shape: zml.Shape = .init(.{ .b = 2, .s = 3, .d = 512 }, .f32);
    try std.testing.expect(output.shape().eqlWithTags(output_shape));
    const expected = test_data.transformer;
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

test "Attention: cached" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    var rng: std.Random.DefaultPrng = .init(42);
    const random = rng.random();
    const layer: Attention = .{
        .q_proj = .init(.fromShape(.init(.{ .out = 512, .d = 512 }, .f32)), null, .d),
        .k_proj = .init(.fromShape(.init(.{ .out = 512, .d = 512 }, .f32)), null, .d),
        .v_proj = .init(.fromShape(.init(.{ .out = 512, .d = 512 }, .f32)), null, .d),
        .o_proj = .init(.fromShape(.init(.{ .out = 512, .d = 512 }, .f32)), null, .d),
    };
    const input_shape: zml.Shape = .init(.{ .b = 2, .s = 253, .d = 512 }, .f32);
    const input = try allocator.create([2][253][512]f32);
    defer allocator.destroy(input);
    for (std.mem.bytesAsSlice(f32, std.mem.asBytes(input))) |*value| value.* = random.float(f32) * 2 - 1;
    var input_buffer = try zml.Buffer.fromBytes(io, platform, input_shape, std.mem.asBytes(input));
    defer input_buffer.deinit();
    var weight_buffers = try zml.mem.bufferize(allocator, Attention, &layer);
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
    const cache: KV_Cache = .init(2, 256, .f32);
    var cache_buffers: zml.Bufferized(KV_Cache) = undefined;
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
    const cached_output = try allocator.create([2][253][512]f32);
    defer allocator.destroy(cached_output);
    inline for (.{ .{ .start = 0, .end = 250 }, .{ .start = 250, .end = 251 }, .{ .start = 251, .end = 253 } }) |step| {
        const count = step.end - step.start;
        const chunk_shape: zml.Shape = .init(.{ .b = 2, .s = count, .d = 512 }, .f32);
        const chunk = try allocator.create([2][count][512]f32);
        defer allocator.destroy(chunk);
        for (chunk, input) |*batch, *source| @memcpy(batch, source[step.start..step.end]);
        var chunk_buffer = try zml.Buffer.fromBytes(io, platform, chunk_shape, std.mem.asBytes(chunk));
        defer chunk_buffer.deinit();
        const position: i32 = step.start;
        var position_buffer = try zml.Buffer.fromBytes(io, platform, .scalar(.i32), std.mem.asBytes(&position));
        defer position_buffer.deinit();
        var exe = try platform.compile(allocator, io, layer, .forward_cached, .{ zml.Tensor.fromShape(chunk_shape), zml.Tensor.fromShape(.scalar(.i32)), cache, @as(usize, 1) }, .{});
        defer exe.deinit();
        var runner = try exe.runner(allocator);
        defer runner.deinit(allocator);
        var result: zml.Bufferized(Cached_Output) = undefined;
        runner.run(io, .{ weight_buffers, chunk_buffer, position_buffer, cache_buffers, @as(usize, 1) }, .{&result}, .{ .wait = true });
        defer result.hidden.deinit();
        zml.Buffer.deinitAll(KV_Cache, &cache_buffers);
        cache_buffers = result.cache;
        const values_slice = try result.hidden.toSliceAlloc(allocator, io);
        defer values_slice.free(allocator);
        const values: *const [2][count][512]f32 = @ptrCast(@alignCast(values_slice.data().ptr));
        for (cached_output, values) |*batch, *source| @memcpy(batch[step.start..step.end], source);
    }
    const output_shape: zml.Shape = .init(.{ .b = 2, .s = 253, .d = 512 }, .f32);
    try std.testing.expect(uncached_output.shape().eqlWithTags(output_shape));
    try zml.testing.expectClose(io, uncached_output, zml.Slice.init(output_shape, std.mem.asBytes(cached_output)), .{
        .absolute_tolerance = 1e-5,
        .relative_tolerance = 1e-4,
        .minimum_close_fraction = 1,
    });
}

test "Transformer_Layer: cached" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    var rng: std.Random.DefaultPrng = .init(42);
    const random = rng.random();
    const layer: Transformer_Layer = .{
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
    };
    const input_shape: zml.Shape = .init(.{ .b = 2, .s = 3, .d = 512 }, .f32);
    var input: [2][3][512]f32 = undefined;
    for (std.mem.bytesAsSlice(f32, std.mem.asBytes(&input))) |*value| value.* = random.float(f32) * 2 - 1;
    var input_buffer = try zml.Buffer.fromBytes(io, platform, input_shape, std.mem.asBytes(&input));
    defer input_buffer.deinit();
    var weight_buffers = try zml.mem.bufferize(allocator, Transformer_Layer, &layer);
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
    const cache: KV_Cache = .init(2, 5, .f32);
    var cache_buffers: zml.Bufferized(KV_Cache) = undefined;
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
    var cached_output: [2][3][512]f32 = undefined;
    inline for (.{ .{ .start = 0, .end = 1 }, .{ .start = 1, .end = 3 } }) |step| {
        const count = step.end - step.start;
        const chunk_shape: zml.Shape = .init(.{ .b = 2, .s = count, .d = 512 }, .f32);
        var chunk: [2][count][512]f32 = undefined;
        for (&chunk, input) |*batch, source| @memcpy(batch, source[step.start..step.end]);
        var chunk_buffer = try zml.Buffer.fromBytes(io, platform, chunk_shape, std.mem.asBytes(&chunk));
        defer chunk_buffer.deinit();
        const position: i32 = step.start;
        var position_buffer = try zml.Buffer.fromBytes(io, platform, .scalar(.i32), std.mem.asBytes(&position));
        defer position_buffer.deinit();
        var exe = try platform.compile(allocator, io, layer, .forward_cached, .{ zml.Tensor.fromShape(chunk_shape), zml.Tensor.fromShape(.scalar(.i32)), cache, @as(usize, 1) }, .{});
        defer exe.deinit();
        var runner = try exe.runner(allocator);
        defer runner.deinit(allocator);
        var result: zml.Bufferized(Cached_Output) = undefined;
        runner.run(io, .{ weight_buffers, chunk_buffer, position_buffer, cache_buffers, @as(usize, 1) }, .{&result}, .{ .wait = true });
        defer result.hidden.deinit();
        zml.Buffer.deinitAll(KV_Cache, &cache_buffers);
        cache_buffers = result.cache;
        const values = try result.hidden.getValue([2][count][512]f32, io);
        for (&cached_output, values) |*batch, source| @memcpy(batch[step.start * 1 .. step.end * 1], &source);
    }
    const output_shape: zml.Shape = .init(.{ .b = 2, .s = 3, .d = 512 }, .f32);
    try std.testing.expect(uncached_output.shape().eqlWithTags(output_shape));
    try zml.testing.expectClose(io, uncached_output, zml.Slice.init(output_shape, std.mem.asBytes(&cached_output)), .{
        .absolute_tolerance = 1e-5,
        .relative_tolerance = 1e-4,
        .minimum_close_fraction = 1,
    });
}

test "Transformer: cached" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    var rng: std.Random.DefaultPrng = .init(42);
    const random = rng.random();
    const layer: Transformer = .{
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
    };
    const input_shape: zml.Shape = .init(.{ .b = 2, .s = 253, .d = 512 }, .f32);
    const input = try allocator.create([2][253][512]f32);
    defer allocator.destroy(input);
    for (std.mem.bytesAsSlice(f32, std.mem.asBytes(input))) |*value| value.* = random.float(f32) * 2 - 1;
    var input_buffer = try zml.Buffer.fromBytes(io, platform, input_shape, std.mem.asBytes(input));
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
    const cache: KV_Cache = .init(2, 256, .f32);
    var cache_buffers: zml.Bufferized(KV_Cache) = undefined;
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
    const cached_output = try allocator.create([2][253][512]f32);
    defer allocator.destroy(cached_output);
    inline for (.{ .{ .start = 0, .end = 1 }, .{ .start = 1, .end = 250 }, .{ .start = 250, .end = 253 } }) |step| {
        const count = step.end - step.start;
        const chunk_shape: zml.Shape = .init(.{ .b = 2, .s = count, .d = 512 }, .f32);
        const chunk = try allocator.create([2][count][512]f32);
        defer allocator.destroy(chunk);
        for (chunk, input) |*batch, *source| @memcpy(batch, source[step.start..step.end]);
        var chunk_buffer = try zml.Buffer.fromBytes(io, platform, chunk_shape, std.mem.asBytes(chunk));
        defer chunk_buffer.deinit();
        const position: i32 = step.start;
        var position_buffer = try zml.Buffer.fromBytes(io, platform, .scalar(.i32), std.mem.asBytes(&position));
        defer position_buffer.deinit();
        var exe = try platform.compile(allocator, io, layer, .forward_cached, .{ zml.Tensor.fromShape(chunk_shape), zml.Tensor.fromShape(.scalar(.i32)), cache }, .{});
        defer exe.deinit();
        var runner = try exe.runner(allocator);
        defer runner.deinit(allocator);
        var result: zml.Bufferized(Cached_Output) = undefined;
        runner.run(io, .{ weight_buffers, chunk_buffer, position_buffer, cache_buffers }, .{&result}, .{ .wait = true });
        defer result.hidden.deinit();
        zml.Buffer.deinitAll(KV_Cache, &cache_buffers);
        cache_buffers = result.cache;
        const values_slice = try result.hidden.toSliceAlloc(allocator, io);
        defer values_slice.free(allocator);
        const values: *const [2][count][512]f32 = @ptrCast(@alignCast(values_slice.data().ptr));
        for (cached_output, values) |*batch, *source| @memcpy(batch[step.start..step.end], source);
    }
    const output_shape: zml.Shape = .init(.{ .b = 2, .s = 253, .d = 512 }, .f32);
    try std.testing.expect(uncached_output.shape().eqlWithTags(output_shape));
    try zml.testing.expectClose(io, uncached_output, zml.Slice.init(output_shape, std.mem.asBytes(cached_output)), .{
        .absolute_tolerance = 1e-5,
        .relative_tolerance = 1e-4,
        .minimum_close_fraction = 1,
    });
}

test "Attention: sliding window boundary" {
    const Window_Attention = struct {
        attention: Attention,
        pub fn forward_uncached(self: @This(), input: zml.Tensor) zml.Tensor {
            return self.attention.forward_uncached(input).slice(.s, .{ .start = 249 });
        }
    };
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    var rng: std.Random.DefaultPrng = .init(42);
    const random = rng.random();
    const layer: Window_Attention = .{
        .attention = .{
            .q_proj = .init(.fromShape(.init(.{ .out = 512, .d = 512 }, .f32)), null, .d),
            .k_proj = .init(.fromShape(.init(.{ .out = 512, .d = 512 }, .f32)), null, .d),
            .v_proj = .init(.fromShape(.init(.{ .out = 512, .d = 512 }, .f32)), null, .d),
            .o_proj = .init(.fromShape(.init(.{ .out = 512, .d = 512 }, .f32)), null, .d),
        },
    };
    const input_shape: zml.Shape = .init(.{ .b = 2, .s = 253, .d = 512 }, .f32);
    const input = try allocator.create([2][253][512]f32);
    defer allocator.destroy(input);
    for (std.mem.bytesAsSlice(f32, std.mem.asBytes(input))) |*value| value.* = random.float(f32) * 2 - 1;
    var input_buffer = try zml.Buffer.fromBytes(io, platform, input_shape, std.mem.asBytes(input));
    defer input_buffer.deinit();
    var weight_buffers = try zml.mem.bufferize(allocator, Window_Attention, &layer);
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
    const output_shape: zml.Shape = .init(.{ .b = 2, .s = 4, .d = 512 }, .f32);
    try std.testing.expect(output.shape().eqlWithTags(output_shape));
    const expected = test_data.attention_window;
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
