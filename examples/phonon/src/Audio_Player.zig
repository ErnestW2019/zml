const std = @import("std");
const c = @import("c");
const Audio_Player = @This();

/// Owns a stationary miniaudio device and a one-second SPSC queue. The calling
/// thread produces mono f32 samples; the device callback consumes them. miniaudio
/// converts the requested sample rate/format to the hardware's native format.
handle: *c.phonon_audio,

pub fn init(sample_rate: u32) !Audio_Player {
    return initBackend(sample_rate, false);
}

fn initBackend(sample_rate: u32, null_backend: bool) !Audio_Player {
    var handle: ?*c.phonon_audio = null;
    try check(c.phonon_audio_create(sample_rate, @intFromBool(null_backend), &handle));
    return .{ .handle = handle.? };
}

/// Copies mono waveform samples into the queue and starts playback on the first
/// nonempty write. Blocks the producer when full; the callback never waits.
pub fn write(self: Audio_Player, samples: []const f32) !void {
    try check(c.phonon_audio_write(self.handle, samples.ptr, samples.len));
}

/// Marks EOF, drains queued audio plus a short silent tail, then stops playback.
/// Call once after the final write. Subsequent writes are rejected.
pub fn finish(self: Audio_Player) !void {
    try check(c.phonon_audio_finish(self.handle));
}

/// Stops playback immediately and releases all resources, including on errors.
pub fn deinit(self: Audio_Player) void {
    c.phonon_audio_destroy(self.handle);
}

fn check(result: c_int) !void {
    if (result != 0) {
        std.log.err("miniaudio failed ({d})", .{result});
        return error.AudioPlaybackFailed;
    }
}

test "Audio_Player: queue backpressure and drain" {
    const player = try initBackend(24000, true);
    defer player.deinit();
    // More than one queue capacity exercises wraparound and producer waiting.
    const samples = try std.testing.allocator.alloc(f32, 25000);
    defer std.testing.allocator.free(samples);
    @memset(samples, 0);
    try player.write(samples[0..1920]);
    try player.write(samples[1920..]);
    try player.finish();
    try std.testing.expect(c.phonon_audio_write(player.handle, samples.ptr, 1) != 0);
}

test "Audio_Player: empty stream" {
    const player = try initBackend(24000, true);
    defer player.deinit();
    try player.finish();
}
