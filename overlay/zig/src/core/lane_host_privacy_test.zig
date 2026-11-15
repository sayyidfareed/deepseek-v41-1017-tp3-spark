// Modified for the DeepSeek 1017 TP3 integration; upstream copyrights retained. See CREDITS.md and provenance.json.
//! Shared scheduling proof only: synthetic backend, not secure GPU qualification.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");

test "mixed C4 schedules together but exposes only ordinary request traffic" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    const Probe = struct {
        host: ?*api.LaneHost = null,
        max_batch: usize = 0,
        saw_mixed: bool = false,
        ownership_correct: bool = true,
        public_running_correct: bool = true,
        fn verify(ptr: *anyopaque, windows: []const lanes.backend.Window, out: []lanes.backend.Verified) anyerror!void {
            const fake: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
            const p: *@This() = @ptrCast(@alignCast(fake.prefill_hook_ctx.?));
            p.max_batch = @max(p.max_batch, windows.len);
            var ordinary = false;
            var private = false;
            for (windows) |w| {
                if (w.stream.prompt()[0] == 91) private = true else ordinary = true;
                p.ownership_correct = p.ownership_correct and (w.stream.secure_request == (w.stream.prompt()[0] == 91));
            }
            p.saw_mixed = p.saw_mixed or (ordinary and private);
            if (windows.len == 4) {
                var status: api.Status = .{};
                p.host.?.engine().status(&status, &.{});
                p.public_running_correct = p.public_running_correct and status.running == 3;
            }
            try fake.backend().vtable.verify(ptr, windows, out);
        }
    };
    var probe: Probe = .{};
    target.prefill_hook_ctx = &probe;
    var table = target.backend().vtable.*;
    table.verify = Probe.verify;
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, .{ .ptr = &target, .vtable = &table }, clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 4 });
    probe.host = &host;
    const Box = struct {
        done: std.atomic.Value(bool) = .init(false),
        tokens: usize = 0,
        reason: api.Reason = .failed,
        fn event(ctx: *anyopaque, _: api.Id, e: *const api.Event) void {
            const b: *@This() = @ptrCast(@alignCast(ctx));
            switch (e.*) {
                .tokens => |t| b.tokens += t.len,
                .finished => |f| { b.reason = f.reason; b.done.store(true, .release); },
                else => {},
            }
        }
    };
    const prompts = [_][8]u32{
        .{ 1, 2, 3, 4, 5, 6, 7, 8 }, .{ 2, 2, 3, 4, 5, 6, 7, 8 },
        .{ 3, 2, 3, 4, 5, 6, 7, 8 }, .{ 91, 2, 3, 4, 5, 6, 7, 8 },
    };
    var requests: [4]api.Request = undefined;
    var boxes: [4]Box = @splat(.{});
    for (&requests, &boxes, 0..) |*r, *b, i| {
        r.* = .{ .prompt = &prompts[i], .max_tokens = 96, .secure_request = i == 3 };
        try host.engine().submit(@intCast(i), r, .{ .ctx = b, .event = Box.event });
    }
    var status: api.Status = .{};
    host.engine().status(&status, &.{});
    const public_waiting = status.waiting;
    try host.start();
    var all_done = false;
    for (0..5000) |_| {
        all_done = true;
        for (&boxes) |*b| all_done = all_done and b.done.load(.acquire);
        if (all_done) break;
        try std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake);
    }
    host.stop(); // join before reading backend or host-owned data
    try std.testing.expect(all_done);
    for (&boxes) |*b| {
        try std.testing.expectEqual(api.Reason.length, b.reason);
        try std.testing.expectEqual(@as(usize, 96), b.tokens);
    }
    try std.testing.expect(probe.saw_mixed);
    try std.testing.expect(probe.ownership_correct);
    try std.testing.expect(probe.public_running_correct);
    try std.testing.expectEqual(@as(usize, 4), probe.max_batch);
    try std.testing.expectEqual(@as(u32, 3), public_waiting);
    try std.testing.expectEqual(@as(u64, 24), host.host_counts.computed_tokens);
}


test "private round scratch is erased before completion and ordinary neighboring memory survives" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var bytes: [8192]u8 align(16) = @splat(0x77);
    var fixed = std.heap.FixedBufferAllocator.init(&bytes);
    const a = fixed.allocator();
    var core = lanes.Engine.init(a, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    const ordinary = try a.alloc(u8, 64);
    @memset(ordinary, 0xab);
    const secret = try core.arena.allocator().alloc(u8, 128);
    @memset(secret, 0x91);
    core.clearPrivateScratch();
    try std.testing.expectEqual(@as(usize, 0), core.arena.queryCapacity());
    try std.testing.expect(std.mem.allEqual(u8, secret, 0));
    try std.testing.expect(std.mem.allEqual(u8, ordinary, 0xab));
    a.free(ordinary);
}

test "private admission refuses a content recorder before model work" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var recorder = lanes.events.Log.init(gpa);
    defer recorder.deinit();
    core.log = &recorder;
    var stream = try lanes.Stream.init(gpa, .{ .secure_request = true, .id = "private", .prompt = &.{91}, .max_new = 4 });
    defer stream.deinit(gpa);
    try std.testing.expectError(error.SecureTracingEnabled, core.addStream(&stream));
    try std.testing.expectError(error.SecureTracingEnabled, core.fillStream(&stream, true));
    try std.testing.expectEqual(@as(usize, 0), recorder.events.items.len);
    try std.testing.expectEqual(@as(usize, 0), core.activeCount());
}


test "private queued draft buffers are erased when forced tokens replace them" {
    var memory: [65536]u8 = @splat(0xa5);
    var fixed = std.heap.FixedBufferAllocator.init(&memory);
    const a = fixed.allocator();
    var cfg = try lanes.Config.init(std.testing.allocator, .{ .exact_width = 8 }, 8, 7);
    defer cfg.deinit(std.testing.allocator);
    var target: lanes.fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var stream = try lanes.Stream.init(a, .{ .secure_request = true, .id = "private", .prompt = &.{91}, .max_new = 4 });
    defer stream.deinit(a);
    const tokens = try a.dupe(u32, &.{ 91, 92 });
    const parents = try a.dupe(i32, &.{ -1, 0 });
    const neighbor = try a.alloc(u8, 64);
    @memset(neighbor, 0xab);
    stream.next = .{ .count = 2, .tokens = tokens, .parents = parents };
    try stream.appendForced(a, &.{31});
    const planned = try lanes.windows.plan(&core, &stream, null);
    try std.testing.expectEqual(lanes.windows.Kind.forced, planned.kind);
    try std.testing.expect(stream.next == null);
    try std.testing.expect(std.mem.allEqual(u32, tokens, 0));
    try std.testing.expect(std.mem.allEqual(i32, parents, 0));
    try std.testing.expect(std.mem.allEqual(u8, neighbor, 0xab));
    core.clearPrivateScratch();
}
