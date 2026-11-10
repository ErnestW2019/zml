//! Host-side Phonon tokenizer, including its JSON-defined text normalization.

// TODO: all allocations in here are a mess of wastefulness, but very low prio

const std = @import("std");
const zml = @import("zml");

const util = @import("../util.zig");

test {
    std.testing.refAllDecls(@This());
}

const c = @import("c");

const Tokenizer = @This();

inner: zml.tokenizer.Tokenizer,
definition: std.json.Parsed(std.json.Value),
normalizers: []Normalizer,
allocator: std.mem.Allocator,

/// Loads a local tokenizer JSON definition through the supplied IO.
pub fn init(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !Tokenizer {
    const here = @typeName(@This()) ++ ".init";

    var span = zml.tracer.span_at(@src(), here, .{});
    defer span.end();

    const bytes = blk: {
        var span_read = zml.tracer.span_at(@src(), here ++ "/read", .{});
        defer span_read.end();

        break :blk try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited);
    };
    defer allocator.free(bytes);

    var span_process = zml.tracer.span_at(@src(), here ++ "/process", .{});
    defer span_process.end();

    var definition = try std.json.parseFromSlice(std.json.Value, allocator, bytes, .{ .allocate = .alloc_always });
    errdefer definition.deinit();

    const steps = definition.value.object.get("normalizer").?.object.get("normalizers").?.array.items;
    const normalizers = try allocator.alloc(Normalizer, steps.len);
    errdefer allocator.free(normalizers);

    var initialized: usize = 0;
    errdefer for (normalizers[0..initialized]) |normalizer| normalizer.deinit();
    for (steps, normalizers) |step, *normalizer| {
        normalizer.* = try .init(step);
        initialized += 1;
    }

    // Match the checkpoint's literal special tokens before normalizing text.
    for (definition.value.object.get("added_tokens").?.array.items) |token| {
        for ([_][]const u8{ "normalized", "single_word", "lstrip", "rstrip" }) |flag| {
            if (token.object.get(flag).?.bool) return error.UnsupportedAddedToken;
        }
    }

    // IREE cannot load the 191-step pipeline or its lookaround regexes.
    // Apply those rules here; retain IREE for Unigram segmentation and IDs.
    var backend_definition = definition.value;

    const original_normalizer = backend_definition.object.get("normalizer").?;
    backend_definition.object.getPtr("normalizer").?.* = .null;

    const backend_json = try std.json.Stringify.valueAlloc(allocator, backend_definition, .{});
    defer allocator.free(backend_json);

    backend_definition.object.getPtr("normalizer").?.* = original_normalizer;

    const inner = try zml.tokenizer.Tokenizer.fromBytes(allocator, backend_json);
    return .{ .inner = inner, .definition = definition, .normalizers = normalizers, .allocator = allocator };
}

pub fn deinit(self: *Tokenizer) void {
    const here = @typeName(@This()) ++ ".deinit";

    var span = zml.tracer.span_at(@src(), here, .{});
    defer span.end();

    self.inner.deinit();
    for (self.normalizers) |normalizer| normalizer.deinit();
    self.allocator.free(self.normalizers);
    self.definition.deinit();
}

/// Tokenizes one UTF-8 string on the CPU. The string must already contain
/// any conversation formatting and expanded audio placeholders.
///
/// Returns an owned i64 tensor matching Python's input_ids dtype:
/// - `.b`: batch, fixed to 1 because this method accepts a single string.
/// - `.s`: sequence, one integer token ID per position, with no padding.
/// Shape: `[.b=1, .s=number_of_tokens]`. Free with `result.free(allocator)`.
pub fn encode(self: *const Tokenizer, allocator: std.mem.Allocator, text: []const u8) !zml.Slice {
    const here = @typeName(@This()) ++ ".encode";

    var span = zml.tracer.span_at(@src(), here, .{});
    defer span.end();

    var ids: std.ArrayList(i64) = .empty;
    defer ids.deinit(allocator);

    var start: usize = 0;
    var position: usize = 0;
    const added_tokens = self.definition.value.object.get("added_tokens").?.array.items;
    while (position < text.len) {
        var match: ?std.json.Value = null;
        var match_length: usize = 0;
        for (added_tokens) |token| {
            const content = token.object.get("content").?.string;
            if (content.len > match_length and std.mem.startsWith(u8, text[position..], content)) {
                match = token;
                match_length = content.len;
            }
        }
        if (match) |token| {
            try self.encode_text(allocator, text[start..position], &ids);
            try ids.append(allocator, token.object.get("id").?.integer);
            position += match_length;
            start = position;
        } else position += 1;
    }
    try self.encode_text(allocator, text[start..], &ids);

    const result = try zml.Slice.alloc(allocator, .init(.{ .b = 1, .s = ids.items.len }, .i64));
    @memcpy(result.items(i64), ids.items);

    return result;
}

fn encode_text(self: *const Tokenizer, allocator: std.mem.Allocator, text: []const u8, ids: *std.ArrayList(i64)) !void {
    const here = @typeName(@This()) ++ ".encode_text";

    if (text.len == 0) return;

    var span = zml.tracer.span_at(@src(), here, .{});
    defer span.end();

    var normalized = try allocator.dupe(u8, text);
    defer allocator.free(normalized);
    for (self.normalizers) |normalizer| {
        const next = try normalizer.apply(allocator, normalized);
        allocator.free(normalized);
        normalized = next;
    }

    var encoder = try self.inner.encoder();
    defer encoder.deinit();
    const tokens = try encoder.encodeAlloc(allocator, normalized);
    defer allocator.free(tokens);
    for (tokens) |id| try ids.append(allocator, @intCast(id));
}

pub fn encode_prompt(self: *const Tokenizer, gpa: std.mem.Allocator, text: []const u8, voice_frames: usize) !zml.Slice {
    var prompt: std.Io.Writer.Allocating = .init(gpa);
    defer prompt.deinit();
    for (0..voice_frames) |_| try prompt.writer.writeAll("<|audio_prompt|>");
    try prompt.writer.writeAll(std.mem.trim(u8, text, " \t\r\n\x0b\x0c"));
    try prompt.writer.writeAll("<|audio_bos|>");
    return self.encode(gpa, prompt.written());
}

/// Tokenizes on the CPU and transfers the `[.b=1, .s]` i64 IDs to the
/// execution platform. The caller owns the returned buffer and must deinit it.
pub fn encode_buffer(self: *const Tokenizer, gpa: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, text: []const u8) !zml.Buffer {
    const tokens = try self.encode(gpa, text);
    defer tokens.free(gpa);
    return .fromSlice(io, platform, tokens);
}

test Tokenizer {
    const io = std.testing.io;
    const gpa = std.testing.allocator;

    var arena_impl: std.heap.ArenaAllocator = .init(gpa);
    defer arena_impl.deinit();
    const arena = arena_impl.allocator();

    const tokenizer_path = try util.assetPath(arena, util.gradium_phonon_31m_en_tokenizer);
    var tokenizer = try Tokenizer.init(gpa, io, tokenizer_path);
    defer tokenizer.deinit();

    // Fixed cases exercise normalization, byte fallback, and audio control tokens.
    const Case = struct { text: []const u8, ids: []const i64 };
    for ([_]Case{
        .{ .text = "", .ids = &.{} },
        .{ .text = "  hello", .ids = &.{ 3248, 663 } },
        .{ .text = "café", .ids = &.{ 1463, 961, 1053, 663 } },
        .{ .text = "“hello”\tworld", .ids = &.{ 660, 3915, 1448, 749, 805, 3915, 956, 663 } },
        .{ .text = "cost $5.00: wow!", .ids = &.{ 1547, 662, 674, 1403, 856, 663, 720, 720, 661, 2529, 1367 } },
        .{ .text = "<|audio_prompt|>hello<|audio_bos|>", .ids = &.{ 5, 3248, 663, 4 } },
        .{ .text = "こんにちは", .ids = &.{ 660, 631, 533, 551, 631, 534, 551, 631, 533, 575, 631, 533, 565, 631, 533, 579, 663 } },
        .{ .text = "тест", .ids = &.{ 660, 612, 566, 1558, 1943, 2096, 663 } },
        .{ .text = "hi 😀 there", .ids = &.{ 2048, 715, 663 } },
    }) |case| {
        const tokens = try tokenizer.encode(gpa, case.text);
        defer tokens.free(gpa);
        try std.testing.expectEqualSlices(i64, case.ids, tokens.constItems(i64));
    }

    const input = "<|audio_prompt|><|audio_prompt|><|audio_prompt|>Bonjour Corentin dit Cerisier<|audio_bos|>";
    const expected = [_]i64{ 5, 5, 5, 1538, 697, 1709, 939, 2887, 945, 765, 995, 674, 840, 707, 983, 2160, 663, 4 };
    const tokens = try tokenizer.encode(gpa, input);
    defer tokens.free(gpa);
    try std.testing.expectEqual(@as(i64, 1), tokens.shape.dim(.b));
    try std.testing.expectEqual(@as(i64, expected.len), tokens.shape.dim(.s));
    try std.testing.expectEqualSlices(i64, &expected, tokens.constItems(i64));

    const prompt_tokens = try tokenizer.encode_prompt(gpa, "Bonjour Corentin dit Cerisier", 3);
    defer prompt_tokens.free(gpa);
    try std.testing.expectEqualSlices(i64, &expected, prompt_tokens.constItems(i64));

    const short_prompt = try tokenizer.encode_prompt(gpa, " \thello\n ", 3);
    defer short_prompt.free(gpa);
    try std.testing.expectEqualSlices(i64, &.{ 5, 5, 5, 3248, 663, 4 }, short_prompt.constItems(i64));

    const no_voice_prompt = try tokenizer.encode_prompt(gpa, "hello", 0);
    defer no_voice_prompt.free(gpa);
    try std.testing.expectEqualSlices(i64, &.{ 3248, 663, 4 }, no_voice_prompt.constItems(i64));

    var buffer = try tokenizer.encode_buffer(gpa, io, zml.testing.env(), input);
    defer buffer.deinit();
    const transferred = try buffer.toSliceAlloc(gpa, io);
    defer transferred.free(gpa);
    try std.testing.expectEqualSlices(i64, &expected, transferred.constItems(i64));
}

// ---

/// One host-side tokenizer JSON normalization step; no neural tensors or weights.
/// init consumes a parsed Replace, Strip or Prepend rule. State is either a compiled
/// PCRE2 pattern plus replacement UTF-8 bytes, or a UTF-8 prefix. Text slices borrow
/// storage from Tokenizer.definition or static literals; compiled patterns are owned
/// and freed by deinit.
/// apply consumes UTF-8 text []const u8 and an allocator, returning owned []u8 of
/// variable byte length after replacement/stripping/prepending. The caller frees it.
/// Regex scratch/output storage is allocated on the host during apply.
const Normalizer = union(enum) {
    /// Host replacement payload: pattern is an owned compiled PCRE2 expression;
    /// content is replacement UTF-8 text borrowed from the parsed JSON definition.
    /// Normalizer.apply maps input text bytes to replacement-result bytes.
    replace: struct { pattern: *anyopaque, content: []const u8 },
    prepend: []const u8,

    fn init(value: std.json.Value) !Normalizer {
        const object = value.object;
        const kind = object.get("type").?.string;
        if (std.mem.eql(u8, kind, "Prepend")) return .{ .prepend = object.get("prepend").?.string };
        var pattern: []const u8 = undefined;
        var literal = false;
        var content: []const u8 = "";
        if (std.mem.eql(u8, kind, "Replace")) {
            const spec = object.get("pattern").?.object;
            literal = spec.contains("String");
            pattern = (spec.get(if (literal) "String" else "Regex") orelse return error.InvalidNormalizer).string;
            content = object.get("content").?.string;
        } else if (std.mem.eql(u8, kind, "Strip")) {
            const left = object.get("strip_left").?.bool;
            const right = object.get("strip_right").?.bool;
            pattern = if (left and right) "^\\s+|\\s+$" else if (left) "^\\s+" else if (right) "\\s+$" else "(?!)";
        } else return error.UnsupportedNormalizer;
        const compiled = c.phonon_regex_compile(pattern.ptr, pattern.len, literal) orelse return error.InvalidNormalizerRegex;
        return .{ .replace = .{ .pattern = compiled, .content = content } };
    }

    fn deinit(self: Normalizer) void {
        switch (self) {
            .replace => |rule| c.phonon_regex_free(rule.pattern),
            .prepend => {},
        }
    }

    fn apply(self: Normalizer, allocator: std.mem.Allocator, text: []const u8) ![]u8 {
        switch (self) {
            .prepend => |prefix| return std.mem.concat(allocator, u8, &.{ prefix, text }),
            .replace => |rule| {
                var capacity = text.len + rule.content.len + 16;
                while (true) {
                    const output = try allocator.alloc(u8, capacity);
                    defer allocator.free(output);
                    var length = capacity;
                    switch (c.phonon_regex_replace(rule.pattern, text.ptr, text.len, rule.content.ptr, rule.content.len, output.ptr, &length)) {
                        0 => return allocator.dupe(u8, output[0..length]),
                        1 => capacity = length,
                        else => return error.NormalizationFailed,
                    }
                }
            },
        }
    }
};
