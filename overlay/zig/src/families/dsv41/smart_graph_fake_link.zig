// Modified for the DeepSeek 1017 TP3 integration; upstream copyrights retained. See CREDITS.md and provenance.json.
//! Test-only in-memory transport for the ACTUAL Runtime, with deadlines.
const std = @import("std");
const w = @import("smart_graph_protocol.zig");
pub threadlocal var available: u64 = 8 * 1024 * 1024 * 1024;
pub threadlocal var rss: u64 = 1024 * 1024 * 1024;
pub const Slot = struct {
    full: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    len: usize = 0,
    bytes: [w.capacity]u8 = undefined,
};
pub const Channel = struct { up: Slot = .{}, down: Slot = .{}, failed: std.atomic.Value(bool) = std.atomic.Value(bool).init(false) };
pub const Link = struct {
    channel: *Channel,
    head: bool,
    pub fn monotonicNs() u64 {
        return @intCast(std.Io.Clock.awake.now(std.Io.Threaded.global_single_threaded.io()).toNanoseconds());
    }
    fn wait(s: Link, slot: *Slot, full: bool, deadline: u64) !void {
        const until = @min(deadline, monotonicNs() + 500 * std.time.ns_per_ms);
        while (slot.full.load(.acquire) != full) {
            if (s.channel.failed.load(.acquire)) return error.PeerDown;
            if (monotonicNs() >= until) return error.StartupTimeout;
            std.atomic.spinLoopHint();
        }
    }
    pub fn sendDeadline(s: Link, bytes: []const u8, deadline: u64) !void {
        if (s.channel.failed.load(.acquire)) return error.PeerDown;
        const slot = if (s.head) &s.channel.down else &s.channel.up;
        try s.wait(slot, false, deadline);
        @memcpy(slot.bytes[0..bytes.len], bytes);
        slot.len = bytes.len;
        slot.full.store(true, .release);
    }
    pub fn recvDeadline(s: Link, out: []u8, deadline: u64) ![]u8 {
        const slot = if (s.head) &s.channel.up else &s.channel.down;
        try s.wait(slot, true, deadline);
        if (slot.len > out.len) return error.FrameTooLarge;
        const n = slot.len;
        @memcpy(out[0..n], slot.bytes[0..n]);
        slot.full.store(false, .release);
        return out[0..n];
    }
};
