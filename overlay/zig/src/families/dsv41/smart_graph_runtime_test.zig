// Modified for the DeepSeek 1017 TP3 integration; upstream copyrights retained. See CREDITS.md and provenance.json.
//! Four real Runtime instances with a fake transport/CUDA owner; no GPU.
const std = @import("std");
const control = @import("smart_graph_controller.zig");
const p = @import("smart_graph_policy.zig");
const fake = @import("smart_graph_fake_link.zig");
const Scenario = enum { normal, host_oom, capture_failure, disconnect, config_mismatch, disabled };
const Runner = struct {
    rank: u32,
    peers: [3]?fake.Link,
    scenario: Scenario,
    owners: [32]u64 = @splat(0),
    retired: u64 = 0,
    captured: u64 = 0,
    eager: u64 = 0,
    generation: u64 = 0,
    inventory: u64 = 0,
    failed: bool = false,
    err: ?anyerror = null,
    finished: bool = false,
    fn mask(ptr: *anyopaque, key: p.Group) u64 {
        const s: *Runner = @ptrCast(@alignCast(ptr));
        return s.owners[@intCast(key.bucket)];
    }
    fn retire(ptr: *anyopaque, key: p.Group) !void {
        const s: *Runner = @ptrCast(@alignCast(ptr));
        const slot = &s.owners[@intCast(key.bucket)];
        if (slot.* == 0) return error.MissingFakeOwner;
        fake.available += @as(u64, @popCount(slot.*)) * 16 * p.mib;
        slot.* = 0;
        s.retired += 1;
    }
    fn run(s: *Runner) void {
        s.work() catch |err| {
            s.err = err;
        };
        s.finished = true;
    }
    fn work(s: *Runner) !void {
        fake.available = 1536 * p.mib;
        fake.rss = 1024 * p.mib;
        var cfg: p.Config = .{ .enabled = s.scenario != .disabled, .limit = 4, .poll_primitives = 16 };
        if (s.scenario == .config_mismatch and s.rank == 3) cfg.reserve += 64 * p.mib;
        var rt = try control.Runtime.open(std.heap.page_allocator, cfg, 0x123, s.rank, 4, s.peers[0..if (s.rank == 0) 3 else 1], s, retire, mask);
        defer rt.close();
        defer {
            s.failed = rt.failed;
            if (rt.ledger) |*l| {
                s.generation = l.generation;
                s.inventory = l.inventory();
                s.eager = l.eager;
            }
        }
        if (!cfg.enabled) {
            try std.testing.expectEqual(null, rt.ledger);
            for (0..128) |_| try rt.boundary(true, .{});
            const leaf: p.Leaf = .{ .group = .{ .kind = .round, .rows = 6, .bucket = 1 } };
            try std.testing.expect(try rt.prepare(leaf, true));
            try rt.captured(leaf, true);
            s.captured = 1;
            return;
        }
        for (0..128) |i| {
            const group: p.Group = .{ .kind = .round, .rows = 6, .bucket = 1 + (i / 3) % 16 };
            try rt.boundary(true, group);
            rt.use(group);
            for (0..3) |piece| {
                const leaf: p.Leaf = .{ .group = group, .piece = @intCast(piece), .greedy = piece == 2 and i % 2 == 0 };
                if (s.owners[@intCast(group.bucket)] & leaf.mask() != 0) {
                    rt.hit();
                    continue;
                }
                if (s.scenario == .disconnect and s.rank == 2 and i == 3) s.peers[0].?.channel.failed.store(true, .release);
                const ready = !(s.scenario == .host_oom and i == 0 and piece == 0 and s.rank == 2);
                if (!try rt.prepare(leaf, ready)) continue;
                if (s.scenario == .capture_failure and i == 3 and s.rank == 2) {
                    try rt.captured(leaf, false);
                    return error.FailureWasNotRejected;
                }
                fake.available -= 16 * p.mib;
                fake.rss += 16 * p.mib;
                try rt.captured(leaf, true);
                s.owners[@intCast(group.bucket)] |= leaf.mask();
                s.captured += 1;
            }
        }
    }
};
fn campaign(scenario: Scenario) ![4]Runner {
    var channels: [3]fake.Channel = @splat(.{});
    var runners: [4]Runner = undefined;
    runners[0] = .{ .rank = 0, .peers = .{ .{ .channel = &channels[0], .head = true }, .{ .channel = &channels[1], .head = true }, .{ .channel = &channels[2], .head = true } }, .scenario = scenario };
    for (1..4) |i| runners[i] = .{ .rank = @intCast(i), .peers = .{ .{ .channel = &channels[i - 1], .head = false }, null, null }, .scenario = scenario };
    var threads: [4]std.Thread = undefined;
    for (&threads, 0..) |*t, i| t.* = try std.Thread.spawn(.{}, Runner.run, .{&runners[i]});
    for (threads) |t| t.join();
    return runners;
}
test "actual controller: four ranks retain retire recapture and learn with identical owners" {
    const ranks = try campaign(.normal);
    for (ranks) |s| {
        try std.testing.expect(s.finished);
        try std.testing.expectEqual(null, s.err);
        try std.testing.expect(!s.failed);
        try std.testing.expectEqual(ranks[0].generation, s.generation);
        try std.testing.expectEqual(ranks[0].inventory, s.inventory);
        try std.testing.expectEqualSlices(u64, &ranks[0].owners, &s.owners);
        try std.testing.expect(s.retired > 0);
    }
}
test "actual controller: one host allocation failure chooses eager on all ranks" {
    const ranks = try campaign(.host_oom);
    for (ranks) |s| {
        try std.testing.expectEqual(null, s.err);
        try std.testing.expect(s.eager > 0);
        try std.testing.expectEqual(ranks[0].captured, s.captured);
    }
}
test "actual controller: a partial capture failure poisons every rank without eager retry" {
    const ranks = try campaign(.capture_failure);
    for (ranks) |s| {
        try std.testing.expect(s.failed);
        try std.testing.expect(s.err != null);
        try std.testing.expectEqual(@as(u64, 4), s.captured);
    }
}
test "actual controller: lost control link fails in bounded time" {
    const ranks = try campaign(.disconnect);
    for (ranks) |s| {
        try std.testing.expect(s.finished);
        try std.testing.expect(s.err != null);
    }
}
test "actual controller: startup mismatch is rejected before any capture" {
    const ranks = try campaign(.config_mismatch);
    for (ranks) |s| {
        try std.testing.expect(s.err != null);
        try std.testing.expectEqual(@as(u64, 0), s.captured);
    }
}
test "disabled controller has no metadata or per primitive control traffic" {
    const ranks = try campaign(.disabled);
    for (ranks) |s| {
        try std.testing.expectEqual(null, s.err);
        try std.testing.expectEqual(@as(u64, 0), s.generation);
        try std.testing.expectEqual(@as(u64, 1), s.captured);
    }
}
