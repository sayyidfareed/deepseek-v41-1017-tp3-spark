// Modified for the DeepSeek 1017 TP3 integration; upstream copyrights retained. See CREDITS.md and provenance.json.
//! Characterize the real host's decode allowance with delayed asynchronous prefill.
//! The fake's wait represents device work already queued by the preceding chunk.
//! This tests scheduling, not CUDA performance or a historical request.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Box = @import("lane_host_tier_test.zig").Box;
const be = lanes.backend;
const io = std.testing.io;
const gpa = std.testing.allocator;
threadlocal var virtual_ns: i96 = 1;

fn now(_: ?*anyopaque, _: std.Io.Clock) std.Io.Timestamp {
    return .{ .nanoseconds = virtual_ns };
}

const Delayed = struct {
    target: lanes.fake.Fake = .{ .gpa = gpa },
    table: be.Backend.VTable = undefined,
    fence: bool,
    pending: bool = false,
    seen: [4]u64 = @splat(0),
    visits: usize = 0,

    fn backend(d: *Delayed) be.Backend {
        const b = d.target.chunkedBackend();
        d.table = b.vtable.*;
        d.table.prefill_part = prefill;
        d.table.verify = verify;
        return .{ .ptr = &d.target, .vtable = &d.table };
    }
    fn parent(ptr: *anyopaque) *Delayed {
        const target: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
        return @fieldParentPtr("target", target);
    }
    fn complete(d: *Delayed) !void {
        if (!d.pending) return;
        d.pending = false;
        virtual_ns += 60 * std.time.ns_per_ms;
    }
    fn prefill(ptr: *anyopaque, s: *lanes.Stream, chunks: usize) anyerror!bool {
        const d = parent(ptr);
        if (s.prompt_len == 16) {
            if (d.visits >= d.seen.len) return error.ExtraPrefillVisit;
            d.seen[d.visits] = d.target.rounds;
            d.visits += 1;
        }
        const done = try d.target.chunkedBackend().vtable.prefill_part.?(ptr, s, chunks);
        if (s.prompt_len == 16 and !done) {
            d.pending = true;
            // Reference: finish the preceding chunk before returning its boundary
            // to the host. The candidate must retain async solo prefill.
            if (d.fence and s.prefill_contended) try d.complete();
        }
        return done;
    }
    fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const d = parent(ptr);
        try d.complete();
        virtual_ns += std.time.ns_per_ms;
        try d.target.backend().verify(windows, out);
    }
};

fn observe(fence: bool) ![3]u64 {
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var d: Delayed = .{ .fence = fence };
    defer d.target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, d.backend(), clock.clock());
    defer core.deinit();
    core.bounded_admission = true;
    var a: Box = .{};
    defer a.deinit();
    var b: Box = .{};
    defer b.deinit();
    var timed = io.vtable.*;
    timed.now = now;
    const host_io: std.Io = .{ .userdata = io.userdata, .vtable = &timed };
    var host = LaneHost.init(gpa, host_io, &core, .{ .lanes = 2 });
    host.interleave_ns = 20 * std.time.ns_per_ms;
    try host.start();
    defer host.stop();
    const engine = host.engine();
    const pa = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6 };
    const ra: api.Request = .{ .prompt = &pa, .max_tokens = 100000 };
    try engine.submit(1, &ra, a.sink());
    while (a.count() == 0) try std.Io.sleep(io, .fromMilliseconds(1), .awake);
    const pb = [_]u32{ 8, 6, 7, 5, 3, 0, 9, 2, 1, 4, 8, 3, 7, 1, 2, 0 };
    const rb: api.Request = .{ .prompt = &pb, .max_tokens = 1 };
    try engine.submit(2, &rb, b.sink());
    try std.testing.expectEqual(api.Reason.length, b.wait());
    engine.cancel(1);
    try std.testing.expectEqual(api.Reason.cancelled, a.wait());
    try b.exact(&pb);
    try std.testing.expectEqual(@as(usize, 4), d.visits);
    return .{ d.seen[1] - d.seen[0], d.seen[2] - d.seen[1], d.seen[3] - d.seen[2] };
}

test "pending prefill can consume the decode allowance; fencing recovers it" {
    const deferred = try observe(false);
    const fenced = try observe(true);
    std.debug.print("prefill-debt decode rounds: deferred={any}, fenced={any}\n", .{ deferred, fenced });
    for (deferred) |rounds| try std.testing.expectEqual(@as(u64, 1), rounds);
    for (fenced) |rounds| try std.testing.expectEqual(@as(u64, 20), rounds);
}
