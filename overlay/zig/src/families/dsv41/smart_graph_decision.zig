// Modified for the DeepSeek 1017 TP3 integration; upstream copyrights retained. See CREDITS.md and provenance.json.
//! The same decision function used by the serving transport and CPU drills.
const std = @import("std");
const p = @import("smart_graph_policy.zig");
const wire = @import("smart_graph_protocol.zig");
pub fn choose(s: *const p.Ledger, c: p.Config, f: *wire.Frame, recovery_pending: bool) void {
    if (!f.ok) {
        f.action = .eager;
        return;
    }
    const offer = f.phase == .offer;
    const slot = offer and !s.resident.contains(f.leaf.group) and s.resident.count() >= c.limit;
    const need = c.reserve + if (offer) c.transient + s.peak(c) else c.hysteresis;
    if (f.available == 0) {
        f.action = if (offer) .eager else .keep;
        return;
    }
    if (f.available >= need and !slot) {
        f.action = if (!offer) .keep else if (s.worthCapture(f.leaf, if (s.misses > 0) s.control_ns / s.misses else 0)) .capture else .eager;
        return;
    }
    // Destruction can be asynchronous. A previous batch must reach a normal
    // synchronized primitive boundary before another batch is authorized.
    if (recovery_pending) {
        f.action = if (offer) .eager else .keep;
        return;
    }
    const deficit = need -| f.available;
    const plan = s.plan(f.leaf.group, deficit, slot, f.available < c.reserve / 2);
    if (plan.n == 0) {
        f.action = if (offer) .eager else .keep;
        return;
    }
    if (offer and s.expected(f.leaf.group) <= 1 and f.available >= c.reserve / 2) {
        var only_hot = true;
        for (plan.keys[0..plan.n]) |k| if (s.resident.get(k).?.aged(s.clock) <= 2) {
            only_hot = false;
        };
        if (only_hot) {
            f.action = .eager;
            return;
        }
    }
    f.action = .retire;
    f.plan = plan;
}
test "ample memory preserves captures but unknown availability never authorizes one" {
    var s = try p.Ledger.init(std.testing.allocator);
    defer s.deinit();
    const c: p.Config = .{ .enabled = true };
    var f: wire.Frame = .{ .phase = .offer, .available = 8 * 1024 * p.mib };
    choose(&s, c, &f, false);
    try std.testing.expectEqual(wire.Action.capture, f.action);
    f.available = 0;
    choose(&s, c, &f, false);
    try std.testing.expectEqual(wire.Action.eager, f.action);
    f.available = 8 * 1024 * p.mib;
    f.ok = false;
    choose(&s, c, &f, false);
    try std.testing.expectEqual(wire.Action.eager, f.action);
}
test "all fake ranks execute the exact head victim set and generation before recapture" {
    var ranks: [4]p.Ledger = undefined;
    for (&ranks) |*s| s.* = try p.Ledger.init(std.testing.allocator);
    defer for (&ranks) |*s| s.deinit();
    for (&ranks) |*s| {
        for (0..20) |i| {
            s.tick();
            s.captured(.{ .group = .{ .kind = .round, .rows = 6, .bucket = i + 1 } }, 1000, 16 * p.mib);
        }
        s.clock += 128;
    }
    var f: wire.Frame = .{ .phase = .probe, .available = 1200 * p.mib };
    choose(&ranks[0], .{ .enabled = true }, &f, false);
    try std.testing.expectEqual(wire.Action.retire, f.action);
    try std.testing.expect(f.plan.n > 0 and f.plan.n <= 8);
    var buf: [wire.capacity]u8 = undefined;
    const encoded = try f.encode(&buf);
    for (&ranks) |*s| {
        const decision = try wire.Frame.decode(encoded);
        for (decision.plan.keys[0..decision.plan.n]) |key| s.retire(key);
        s.generation += 1;
        try std.testing.expectEqual(ranks[0].inventory(), s.inventory());
    }
    f.plan = .{};
    f.action = .keep;
    choose(&ranks[0], .{ .enabled = true }, &f, true);
    try std.testing.expectEqual(wire.Action.keep, f.action);
}
test "current group is protected even under emergency and retirement cannot thrash immediately" {
    var s = try p.Ledger.init(std.testing.allocator);
    defer s.deinit();
    const k: p.Leaf = .{ .group = .{ .kind = .round, .rows = 6, .bucket = 1024 } };
    s.tick();
    s.captured(k, 1000, p.mib);
    var f: wire.Frame = .{ .phase = .offer, .leaf = k, .available = 100 * p.mib };
    choose(&s, .{ .enabled = true }, &f, false);
    try std.testing.expectEqual(wire.Action.eager, f.action);
    try std.testing.expectEqual(@as(usize, 1), s.resident.count());
}
