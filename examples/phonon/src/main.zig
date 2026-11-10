const std = @import("std");
const zml = @import("zml");

const util = @import("util.zig");
const Audio_Player = @import("Audio_Player.zig");
const mimi = @import("mimi/root.zig");
const Text_To_Speech_Uncached = @import("phonon/root.zig").Text_To_Speech_Uncached;
const Text_To_Speech_Cached = @import("phonon/root.zig").Text_To_Speech_Cached;

pub const std_options: std.Options = .{ .log_level = .info };

// Retain the Zig half of Bazel's startup hook.
comptime {
    _ = zml.bazel;
    _ = @import("mimi/root.zig");
}

test {
    std.testing.refAllDecls(@This());
}

// TODO: things yet to cover:
// * batch inputs (i.e. multiple prompts at once)
// * different base models (i.e. something other than phonon-31m-en)
// * different voice features
// * not f32?
// * now that it's ported, we could grab the asset snapshots from the zig graph instead? then there's *really* no python
// * how does one benchmark these things?
// * what's with the implicit errors all over the place?
//
// And, more generally, the big 4:
//
// * Streaming
//   Mimi now decodes each completed latent into a waveform chunk during
//   generation. A miniaudio queue feeds the playback callback; inference
//   stalls produce silence when the queue runs dry.
//
// * Batching
//   The flow head can process B > 1, but requests have different prompt lengths
//   and finish at different times. The loop must decide which requests run
//   together, track their cache positions and EOS countdowns separately, and
//   handle a finished row while others continue. Our current single-prompt loop
//   avoids those scheduling questions.
//
// * Latency
//   Prefill determines how soon the first frame can arrive; repeated decode
//   determines whether we can sustain playback. The loop controls compilation
//   reuse, device/host transfers, synchronization, and how much work happens per
//   call. For example, our current code reads velocity back and adds it to noise
//   on the host; performing that addition in the graph could reduce host work
//   and transfers.
//
// * Cancellation
//   When someone interrupts speech, the loop must stop scheduling further
//   steps, discard queued audio, and release the request’s buffers safely. If
//   a device call is already running, cancellation may mean letting that call
//   finish and ignoring its result. Cache donation makes ownership during that
//   cleanup particularly relevant.

// TODO: pipelining
// TODO: dont go to host for no reason
// TODO: batching
// TODO: KV caching (but care for the mem nonsense)
// TODO: smaller floats
// TODO: wait=false
// TODO: cache compilations
// TODO: zio? why? how?
// TODO: sharding?
// TODO: donate and reuse buffers

const use_tts_cache = false;
const use_sta_cache = false;

const TTS = if (use_tts_cache) Text_To_Speech_Cached else Text_To_Speech_Uncached;
const STA = if (use_sta_cache) mimi.Audio_Decoder_Cached else mimi.Audio_Decoder_Uncached;

pub fn main(init: std.process.Init) !void {
    const here = @typeName(@This()) ++ ".main";

    const gpa = init.gpa;
    const io = init.io; // TODO: we should try and see what happens with zio though
    const arena_impl = init.arena;
    const arena = arena_impl.allocator();

    var stdout_buffer: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(io, &stdout_buffer);

    const args = try init.minimal.args.toSlice(arena);
    const no_play = args.len > 1 and std.mem.eql(u8, args[1], "--no-play");
    const text_start: usize = if (no_play) 2 else 1;
    const text = if (args.len > text_start) try std.mem.join(arena, " ", args[text_start..]) else default_prompt;

    const platform: *zml.Platform = try .auto(gpa, io, .{});
    defer platform.deinit(gpa, io);

    const profiler_options: zml.Platform.ProfilerOptions = .{};
    var profiler = try platform.profiler(gpa, io, profiler_options);
    defer profiler.deinit();

    try profiler.start();
    defer {
        if ((profiler.stop() catch unreachable)) |profile| {
            std.log.info("Profile dumped: {s} and {s}", .{
                profile.protobuf_path,
                profile.perfetto_path,
            });
        }
    }

    var span_init = zml.tracer.span_at(@src(), here ++ "/init", .{});
    {}

    var tts = try TTS.init(io, gpa, arena, platform, text);
    defer tts.deinit();

    {
        try stdout.interface.print("Token IDs: {any}\n", .{tts.ids_list.items});
        try stdout.interface.flush();
    }

    // TODO: no reason for this to init on the same thread right?
    var audio_decoder = try STA.init(gpa, io, platform, 1, 1, @intCast(tts.max_frames));
    defer audio_decoder.deinit();

    // TODO: no reason for this to init on the same thread right?
    const player_opt: ?Audio_Player = if (no_play) null else try .init(mimi.sampling_rate);
    defer if (player_opt) |audio| audio.deinit();

    var waveform: std.ArrayList(f32) = .empty;
    defer waveform.deinit(gpa);

    {}
    span_init.end();

    var iteration: usize = 0;
    while (!tts.done) : (iteration += 1) {
        var span_iteration = zml.tracer.span_at(@src(), here ++ "/iteration", .{ .iteration = iteration });
        defer span_iteration.end();

        const out = try tts.step();

        // TODO: clearly there's yet another clear lack of pipelining here
        {
            // TODO: clearly there are missed pipelining opportunities here?
            // TODO: similarly, why would we go back through the host in the first place?
            // -> im being told that we can re-use GPU handles across programs!

            const frame = tts.latents_list.items[tts.latents_list.items.len - 32 ..];
            const chunk = try audio_decoder.decode(zml.Slice.init(
                .init(.{ .b = 1, .s = 1, .ld = 32 }, .f32),
                std.mem.sliceAsBytes(frame),
            ));
            defer chunk.free(gpa);

            try waveform.appendSlice(gpa, chunk.constItems(f32));
            if (player_opt) |audio| {
                var span_push_waveform = zml.tracer.span_at(@src(), here ++ "/push_waveform", .{});
                defer span_push_waveform.end();

                try audio.write(chunk.constItems(f32));
            }
        }

        try stdout.interface.print("Frame {d}: EOS {d:.5}, sequence length {d}\n", .{ tts.frames(), out.eos_score, tts.ids_list.items.len });
        try stdout.interface.flush();
    }

    if (false) {
        if (player_opt) |player| {
            var span_push_waveforms = zml.tracer.span_at(@src(), here ++ "/push_waveforms", .{});
            defer span_push_waveforms.end();

            try player.write(waveform.items);
            try player.finish();
        }
    }

    if (false) {
        // Bazel runs from its runfiles tree; save beside the caller's working directory.
        const output_path = try std.fs.path.join(arena, &.{ init.environ_map.get("BUILD_WORKING_DIRECTORY") orelse ".", util.generated_latents_path });
        const wav_path = try std.fs.path.join(arena, &.{ init.environ_map.get("BUILD_WORKING_DIRECTORY") orelse ".", "phonon.wav" });

        try save_latents(gpa, io, output_path, tts.latents_list.items);
        try save_wav(io, wav_path, waveform.items);

        try stdout.interface.print("Saved {s}: {d} mono samples at {d} Hz.\n", .{ wav_path, waveform.items.len, mimi.sampling_rate });
        try stdout.interface.print("Generated normalized latents [1,{d},32] f32; stopped by {s}. Saved {s}\n", .{
            tts.frames(), if (tts.stopped_by_eos) "EOS" else "frame limit", output_path,
        });
        try stdout.interface.flush();
    }
}

/// Save normalized TTS history, explicitly labelled to avoid decoder-space confusion.
fn save_latents(allocator: std.mem.Allocator, io: std.Io, path: []const u8, values: []const f32) !void {
    const header = try std.fmt.allocPrint(allocator, "{{\"__metadata__\":{{\"latent_space\":\"normalized TTS; apply latent_std and latent_mean before decoding\"}},\"audio_latents\":{{\"dtype\":\"F32\",\"shape\":[1,{d},32],\"data_offsets\":[0,{d}]}}}}", .{ values.len / 32, values.len * 4 });
    defer allocator.free(header);
    const padding = (8 - header.len % 8) % 8;
    const file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    var buffer: [4096]u8 = undefined;
    var writer = file.writer(io, &buffer);
    var length: [8]u8 = undefined;
    std.mem.writeInt(u64, &length, @intCast(header.len + padding), .little);
    try writer.interface.writeAll(&length);
    try writer.interface.writeAll(header);
    try writer.interface.splatByteAll(' ', padding);
    // Safetensors stores little-endian values, independently of the host.
    for (values) |value| {
        var bytes: [4]u8 = undefined;
        std.mem.writeInt(u32, &bytes, @bitCast(value), .little);
        try writer.interface.writeAll(&bytes);
    }
    try writer.interface.flush();
}

/// Save mono PCM16. Keep floating-point decoder output intact until this final
/// quantization step; clamp only to the representable WAV sample range.
fn save_wav(io: std.Io, path: []const u8, samples: []const f32) !void {
    const byte_count = try std.math.mul(u32, std.math.cast(u32, samples.len) orelse return error.AudioTooLong, 2);
    const riff_size = try std.math.add(u32, byte_count, 36);
    const file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    var buffer: [8192]u8 = undefined;
    var writer = file.writer(io, &buffer);
    var header: [44]u8 = undefined;
    @memcpy(header[0..4], "RIFF");
    std.mem.writeInt(u32, header[4..8], riff_size, .little);
    @memcpy(header[8..16], "WAVEfmt ");
    std.mem.writeInt(u32, header[16..20], 16, .little);
    std.mem.writeInt(u16, header[20..22], 1, .little);
    std.mem.writeInt(u16, header[22..24], 1, .little);
    std.mem.writeInt(u32, header[24..28], mimi.sampling_rate, .little);
    std.mem.writeInt(u32, header[28..32], mimi.sampling_rate * 2, .little);
    std.mem.writeInt(u16, header[32..34], 2, .little);
    std.mem.writeInt(u16, header[34..36], 16, .little);
    @memcpy(header[36..40], "data");
    std.mem.writeInt(u32, header[40..44], byte_count, .little);
    try writer.interface.writeAll(&header);
    for (samples) |sample| {
        if (!std.math.isFinite(sample)) return error.InvalidAudio;
        const quantized: i16 = @intFromFloat(@round(std.math.clamp(sample * 32768, -32768, 32767)));
        var bytes: [2]u8 = undefined;
        std.mem.writeInt(i16, &bytes, quantized, .little);
        try writer.interface.writeAll(&bytes);
    }
    try writer.interface.flush();
}

const default_prompt =
    \\In the last autumn of the northern kingdoms, when the leaves of the elder trees turned silver before they fell, a rider came out of the wilderness bearing neither banner nor blade. His cloak was stiff with frost, his horse stumbled beneath him, and in his right hand he carried a lantern whose flame burned blue.
    \\
    \\At the western gate of Caer Dûn, the watchmen called for him to halt.
    \\
    \\“Name yourself,” said their captain, “and speak your business before these walls.”
    \\
    \\But the rider looked beyond them toward the mountains, where the first snow lay pale upon the heights.
    \\
    \\“Is the king awake?”
    \\
    \\“The king has been dead these seven winters.”
    \\
    \\Then the rider bowed his head, and for a moment the only sound was the wind moving through the iron teeth of the gate.
    \\
    \\“Then wake his daughter,” he said. “And bid your people leave their doors unbarred. Before the moon has crossed the highest tower, there will be many coming who have no strength left to knock.”
    \\
    \\They brought him through streets where the evening fires were being lit. Children followed at a distance, whispering at the strange lantern, and an old woman gathering water dropped her bucket when she saw its flame.
    \\
    \\She had seen such a light once before.
    \\
    \\It had been carried home beside her father’s body.
    \\
    \\Within the hall, Lady Mereth stood beneath the empty throne. She wore no crown. A plain circlet held back her hair, and there was ink upon the fingers of the hand that rested on her sword. Before the rider entered, she had been counting the grain remaining in the eastern storehouses.
    \\
    \\He knelt with difficulty.
    \\
    \\“My lady. The watch at Stonewater has fallen.”
    \\
    \\Mereth dismissed the servants. She waited until the doors had closed.
    \\
    \\“How many escaped?”
    \\
    \\“Thirty when we left the river. Seventeen when we reached the forest.”
    \\
    \\“And Captain Edran?”
    \\
    \\The rider set the lantern on the floor between them.
    \\
    \\Mereth looked at it for a long while. Then she took up the little brass cup beside her chair and poured water for him herself.
    \\
    \\“Tell me.”
    \\
    \\He told her of the river freezing in a single night, though the weather had been mild. He told her of lights moving beneath the ice, keeping pace with the sentries on the bank. At dawn, a figure had appeared upon the farther shore. It wore the armour of a king buried three hundred years before, and wherever it walked, the reeds bent toward it.
    \\
    \\Behind that figure came the host.
    \\
    \\They carried shields split by ancient blows. Some wore burial cloths beneath their mail. The archers of Stonewater loosed until their fingers bled, but the advancing ranks never quickened and never broke.
    \\
    \\“We held the bridge until evening,” the rider said. “Edran stayed to fire the timbers. He gave me the lantern before he went down to the gate.”
    \\
    \\Mereth turned away. Beyond the narrow window, the lamps of the city shone against the gathering dark.
    \\
    \\Her brother had promised to return before the harvest feast.
    \\
    \\At last she asked, “How long?”
    \\
    \\“Three days, if the river delays them. Two, if it does not.”
    \\
    \\From somewhere below the hall came the sound of laughter. The kitchen boys were arguing over a game, unaware that their voices carried through the stairwell.
    \\
    \\“Have you eaten?” she asked.
    \\
    \\The rider shook his head.
    \\
    \\She called the servants back.
    \\
    \\Before midnight, the bells of Caer Dûn were ringing. Messengers rode south with fresh horses waiting at every village. The smithies opened their shutters, and sparks rose into the cold air. In the market square, men who had quarrelled over fences for twenty years stood together in the armoury line.
    \\
    \\Mereth ordered the mountain road cleared for the children and the old. Every wagon was to carry food on the outward journey and return for those who could not walk.
    \\
    \\At the stables, a boy of sixteen caught the bridle of her horse.
    \\
    \\“Will the southern lords come?”
    \\
    \\“I have asked them.”
    \\
    \\“My father says they have forgotten their oaths.”
    \\
    \\“Your father may be right.”
    \\
    \\The boy looked down. Mereth dismounted and took the bridle from his hands.
    \\
    \\“What is your name?”
    \\
    \\“Tavin, my lady.”
    \\
    \\“Well, Tavin. There are forty horses here that must be ready before sunrise. Will you help me?”
    \\
    \\He nodded, and together they began.
    \\
    \\On the second morning, snow came over the northern hills.
    \\
    \\On the second evening, the last refugees from Stonewater reached the gate. Among them was a little girl carrying a wooden bowl. She would give it to no one. Her mother had told her to keep it safe, and the mother had not come.
    \\
    \\Mereth found a place for her beside the kitchen hearth.
    \\
    \\That night, the sentries heard singing beyond the fields.
    \\
    \\It was faint at first, scarcely more than a murmur beneath the wind. Then the words grew clear, and the oldest watchman upon the wall began to weep. He knew the song. His grandmother had sung it while she worked, though he had never heard the final verse.
    \\
    \\Below the battlements, shapes moved between the abandoned farmhouses.
    \\
    \\Mereth climbed the eastern tower before dawn. She had slept an hour in a chair. Her sword belt sat crookedly over her coat, and someone had pressed a piece of bread into her hand as she left the hall.
    \\
    \\The captain of the watch stood beside the parapet.
    \\
    \\“No riders from the south,” he said.
    \\
    \\She looked along the empty road. Then she broke the bread in two and offered him half.
    \\
    \\Together they watched the fields grow pale.
    \\
    \\Far below, the enemy halted. A tall figure came forward alone, its crown white with frost. It lifted its face toward the walls, and when it spoke, its voice was the voice of her father.
    \\
    \\“Open the gate, Mereth. I have come home.”
    \\
    \\Her hand tightened upon the stone.
    \\
    \\Behind her, a young sentry made a small, frightened sound.
    \\
    \\Mereth drew her sword. The blade trembled once before she steadied it.
    \\
    \\“My father,” she said, “would have known the way to the door.”
    \\
    \\She raised the sword above her head.
    \\
    \\Along the wall, bowstrings creaked.
    \\
    \\And from the south, faint but rising through the morning mist, a horn began to blow.
;
