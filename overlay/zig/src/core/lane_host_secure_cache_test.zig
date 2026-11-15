// Modified for the DeepSeek 1017 TP3 integration; upstream copyrights retained. See CREDITS.md and provenance.json.
//! Real host command loop and admission registry; synthetic backend only.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");

test "secure cache forget runs on the lane thread and leaves unrelated admissions usable" {
    const a = std.testing.allocator;
    var cfg = try lanes.Config.init(a, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(a);
    var fake: lanes.fake.Fake = .{ .gpa = a };
    defer fake.deinit();
    const Seen = struct {
        scope: ?[32]u8 = null,
        calls: usize = 0,
        polled: bool = false,
        fn forget(ptr: *anyopaque, scope: ?*const [32]u8) !usize {
            const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
            const s: *@This() = @ptrCast(@alignCast(f.prefill_hook_ctx.?));
            s.calls += 1;
            s.scope = if (scope) |v| v.* else null;
            return 1;
        }
        fn poll(ptr: *anyopaque) !void {
            const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
            const s: *@This() = @ptrCast(@alignCast(f.prefill_hook_ctx.?));
            s.polled = true;
        }
    };
    var seen: Seen = .{};
    fake.prefill_hook_ctx = &seen;
    var vtable = fake.backend().vtable.*;
    vtable.private_cache_forget = Seen.forget;
    vtable.private_cache_poll = Seen.poll;
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, .{ .ptr = &fake, .vtable = &vtable }, clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(a, std.testing.io, &core, .{ .lanes = 4 });
    try host.start();
    var left: api.SecureCacheTicket = .{ .scope = @splat(1) };
    var right: api.SecureCacheTicket = .{ .scope = @splat(2) };
    const engine = host.engine();
    try engine.vtable.secure_cache_enter.?(engine.ctx, &left);
    try engine.vtable.secure_cache_enter.?(engine.ctx, &right);
    const count = try engine.vtable.secure_forget.?(engine.ctx, &left.scope);
    const scoped_correct = left.revoked.load(.acquire) and !right.revoked.load(.acquire);
    const all_count = try engine.vtable.secure_forget.?(engine.ctx, null);
    const all_correct = right.revoked.load(.acquire);
    engine.vtable.secure_cache_leave.?(engine.ctx, &left);
    engine.vtable.secure_cache_leave.?(engine.ctx, &right);
    host.mutex.lockUncancelable(host.io);
    const cancellations = host.cancels.items.len;
    host.mutex.unlock(host.io);
    host.stop();
    try std.testing.expectEqual(@as(usize, 1), count);
    try std.testing.expectEqual(@as(usize, 1), all_count);
    try std.testing.expect(scoped_correct and all_correct);
    try std.testing.expectEqual(@as(usize, 2), seen.calls);
    try std.testing.expect(seen.polled and seen.scope == null);
    try std.testing.expect(std.mem.allEqual(u8, &left.scope, 0));
    try std.testing.expect(std.mem.allEqual(u8, &right.scope, 0));
    try std.testing.expectEqual(@as(usize, 0), cancellations);
}

test "a real host backend failure revokes every secure ticket and clears retained points" {
    const a = std.testing.allocator;
    var cfg = try lanes.Config.init(a, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(a);
    var fake: lanes.fake.Fake = .{ .gpa = a };
    defer fake.deinit();
    const Seen = struct {
        retained: usize = 2,
        cleared: bool = false,
        done: std.atomic.Value(bool) = .init(false),
        failed: bool = false,
        injected: bool = false,
        fn prefill(ptr: *anyopaque, _: *lanes.Stream) !void {
            const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
            const s: *@This() = @ptrCast(@alignCast(f.prefill_hook_ctx.?));
            s.injected = true;
            return error.SyntheticBackendFailure;
        }
        fn fill(_: *anyopaque, _: *lanes.Stream, _: usize) !bool { return error.SyntheticBackendFailure; }
        fn forget(ptr: *anyopaque, scope: ?*const [32]u8) !usize {
            const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
            const s: *@This() = @ptrCast(@alignCast(f.prefill_hook_ctx.?));
            if (scope != null) return error.ExpectedAllScopes;
            const count = s.retained;
            s.retained = 0;
            s.cleared = true;
            return count;
        }
        fn event(ptr: *anyopaque, _: api.Id, value: *const api.Event) void {
            const s: *@This() = @ptrCast(@alignCast(ptr));
            if (value.* == .finished) {
                s.failed = value.finished.reason == .failed;
                s.done.store(true, .release);
            }
        }
    };
    var seen: Seen = .{};
    fake.prefill_hook_ctx = &seen;
    var vtable = fake.backend().vtable.*;
    vtable.prefill = Seen.prefill;
    vtable.prefill_part = Seen.fill;
    vtable.private_cache_forget = Seen.forget;
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, .{ .ptr = &fake, .vtable = &vtable }, clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(a, std.testing.io, &core, .{ .lanes = 4 });
    try host.start();
    defer host.stop();
    const engine = host.engine();
    var first: api.SecureCacheTicket = .{ .scope = @splat(1) };
    var second: api.SecureCacheTicket = .{ .scope = @splat(2) };
    try engine.vtable.secure_cache_enter.?(engine.ctx, &first);
    defer engine.vtable.secure_cache_leave.?(engine.ctx, &first);
    try engine.vtable.secure_cache_enter.?(engine.ctx, &second);
    defer engine.vtable.secure_cache_leave.?(engine.ctx, &second);
    try engine.submit(1, &.{ .prompt = &.{ 2, 7, 1, 8 }, .max_tokens = 32 },
        .{ .ctx = &seen, .event = Seen.event });
    for (0..2000) |_| {
        if (seen.done.load(.acquire)) break;
        std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    }
    try std.testing.expect(seen.done.load(.acquire));
    try std.testing.expect(seen.injected and seen.failed and seen.cleared and seen.retained == 0);
    try std.testing.expect(first.revoked.load(.acquire) and second.revoked.load(.acquire));
}
