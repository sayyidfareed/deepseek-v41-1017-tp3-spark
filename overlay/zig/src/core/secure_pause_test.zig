// Modified for the DeepSeek 1017 TP3 integration; upstream copyrights retained. See CREDITS.md and provenance.json.
const std = @import("std");
const api = @import("engine_api.zig");
const lanes = @import("lanes");

test "secure pause parks the actual lane thread and resumes the same core" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    try host.start();
    defer host.stop();
    const engine = host.engine();
    try host.secure_pause.acquire(&host, 1000);
    defer if (host.secure_pause.requested.load(.acquire)) host.secure_pause.release(&host);
    const Box = struct {
        count: std.atomic.Value(usize) = .init(0),
        done: std.atomic.Value(bool) = .init(false),
        fn event(p: *anyopaque, _: api.Id, e: *const api.Event) void {
            const b: *@This() = @ptrCast(@alignCast(p));
            switch (e.*) {
                .tokens => |tokens| { _ = b.count.fetchAdd(tokens.len, .monotonic); },
                .finished => b.done.store(true, .release),
                else => {},
            }
        }
    };
    var box: Box = .{};
    const req: api.Request = .{ .prompt = &.{ 1, 2, 3, 4 }, .max_tokens = 4 };
    try std.testing.expectError(error.Busy, engine.submit(1, &req, .{ .ctx = &box, .event = Box.event }));
    try std.testing.expectEqual(@as(usize, 0), box.count.load(.acquire));
    try std.testing.expect(host.core == &core);
    host.secure_pause.release(&host);
    try engine.submit(2, &req, .{ .ctx = &box, .event = Box.event });
    for (0..1000) |_| {
        if (box.done.load(.acquire)) break;
        try std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake);
    }
    // Always cancel/drain before test locals go out of scope, including failure.
    if (!box.done.load(.acquire)) { engine.cancel(2); while (!box.done.load(.acquire)) try std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake); }
    try std.testing.expectEqual(@as(usize, 4), box.count.load(.acquire));
    try std.testing.expect(host.core == &core);
}

test "secure pause refuses a host that has no running lane thread" {
    var host: api.LaneHost = undefined;
    host.io = std.testing.io;
    host.mutex = .init;
    host.closing = false;
    host.thread = null;
    host.secure_pause = .{};
    try std.testing.expectError(error.SecurePauseUnavailable, host.secure_pause.acquire(&host, 10));
    try std.testing.expect(!host.secure_pause.requested.load(.acquire));
}
