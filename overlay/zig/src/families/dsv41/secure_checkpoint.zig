// Modified for the DeepSeek 1017 TP3 integration; upstream copyrights retained. See CREDITS.md and provenance.json.
//! Memory-only checkpoint copy plan. No spill store, files, token logging or
//! ownership table. The caller owns the source slot and independent destination
//! allocation until every copy and its fence have completed on every rank.
const std = @import("std");
const state = @import("spill_state.zig");

pub const Span = struct { live: u64, kept: u64, bytes: usize };
pub const Plan = struct {
    spans: [state.max_planes]Span = undefined,
    len: usize = 0,
    bytes: usize = 0,

    /// Replicated layout only, matching the current qualified secure engine.
    /// Charge the entire independent checkpoint, not just its small slot state.
    pub fn size(layout: *const state.StateLayout, at: usize) !usize {
        if (layout.owner_layout != .replicated) return error.PrivateShardedLayoutUnsupported;
        if (at == 0 or layout.geo.block_tokens == 0 or at % layout.geo.block_tokens != 0)
            return error.BadPrivateCheckpoint;
        var total: usize = 0;
        for (layout.blockPlanes()) |p| {
            if (p.ratio == 0 or at % p.ratio != 0) return error.BadPrivateCheckpoint;
            total = try std.math.add(usize, total, try std.math.mul(usize, at / p.ratio, p.row));
        }
        for (layout.smallPlanes()) |p|
            total = try std.math.add(usize, total, try std.math.mul(usize, p.rows, p.row));
        return total;
    }

    /// Validate every range before any copy; destination must be an independent
    /// allocation of exactly the charged size. It may not alias a live plane.
    pub fn make(dev: anytype, layout: *const state.StateLayout, slot: usize,
        base: usize, at: usize, buffer: u64, bytes: usize) !Plan
    {
        const needed = try size(layout, at);
        if (bytes != needed or buffer == 0 or slot >= dev.slots() or
            base % layout.geo.block_tokens != 0 or
            try std.math.add(usize, base, at) > dev.capacity()) return error.BadPrivateCheckpoint;
        const end = try std.math.add(u64, buffer, bytes);
        var plan: Plan = .{ .bytes = needed };
        var offset: usize = 0;
        for (layout.blockPlanes()) |p| {
            if (base % p.ratio != 0) return error.BadPrivateCheckpoint;
            const row0 = dev.blockPlane(p);
            if (row0 == 0) return error.BadPrivateCheckpoint;
            try disjoint(row0, try std.math.mul(usize, dev.capacity() / p.ratio, p.row), buffer, end);
            const address = try std.math.add(u64, row0, try std.math.mul(usize, base / p.ratio, p.row));
            const length = try std.math.mul(usize, at / p.ratio, p.row);
            try plan.add(address, buffer, end, offset, length);
            offset = try std.math.add(usize, offset, length);
        }
        for (layout.smallPlanes()) |p| {
            const length = try std.math.mul(usize, p.rows, p.row);
            try disjoint(dev.slotPlane(p, 0), try std.math.mul(usize, dev.slots(), length), buffer, end);
            try plan.add(dev.slotPlane(p, slot), buffer, end, offset, length);
            offset = try std.math.add(usize, offset, length);
        }
        if (offset != needed) return error.BadPrivateCheckpoint;
        return plan;
    }

    fn disjoint(live: u64, bytes: usize, buffer: u64, end: u64) !void {
        if (live == 0 or bytes == 0) return error.BadPrivateCheckpoint;
        const live_end = try std.math.add(u64, live, bytes);
        if (live < end and buffer < live_end) return error.PrivateCheckpointAliasesLive;
    }

    fn add(p: *Plan, live: u64, buffer: u64, end: u64, offset: usize, bytes: usize) !void {
        if (live == 0 or bytes == 0 or p.len == p.spans.len) return error.BadPrivateCheckpoint;
        const live_end = try std.math.add(u64, live, bytes);
        if (live < end and buffer < live_end) return error.PrivateCheckpointAliasesLive;
        const kept = try std.math.add(u64, buffer, offset);
        if (try std.math.add(u64, kept, bytes) > end) return error.BadPrivateCheckpoint;
        p.spans[p.len] = .{ .live = live, .kept = kept, .bytes = bytes };
        p.len += 1;
    }

    /// Even a partial-copy failure must fence before the caller erases/frees the
    /// destination. A failed fence is an engine failure, never a cold-cache miss.
    pub fn copy(p: *const Plan, dev: anytype, save: bool) !void {
        for (p.spans[0..p.len]) |span| {
            dev.copy(if (save) span.kept else span.live,
                if (save) span.live else span.kept, span.bytes) catch |err| {
                try dev.fence();
                return err;
            };
        }
        try dev.fence();
    }
};

/// The retained allocation is never freed or reused before this returns. Zero
/// covers padding as well as every used plane. Errors propagate to the owner.
pub fn erase(dev: anytype, address: u64, bytes: usize) !void {
    if (address == 0 or bytes == 0) return error.BadPrivateCheckpoint;
    _ = try std.math.add(u64, address, bytes);
    try dev.fence();
    dev.fill(address, 0, bytes) catch |err| {
        try dev.fence();
        return err;
    };
    try dev.fence();
}

const Fake = struct {
    memory: [65536]u8 = @splat(0xa5),
    copies: usize = 0,
    fences: usize = 0,
    fail_copy: ?usize = null,
    fail_fence: bool = false,
    bad_plane: bool = false,
    fn slots(_: *Fake) usize { return 4; }
    fn capacity(_: *Fake) usize { return 16; }
    fn blockPlane(d: *Fake, p: state.Plane) u64 {
        return if (d.bad_plane) 0 else 64 + @as(u64, @intFromEnum(p.kind)) * 4096;
    }
    fn slotPlane(_: *Fake, p: state.Plane, slot: usize) u64 {
        return 64 + @as(u64, @intFromEnum(p.kind)) * 4096 + slot * p.slotBytes();
    }
    fn copy(d: *Fake, dst: u64, src: u64, bytes: usize) !void {
        if (d.fail_copy == d.copies) return error.CopyFailed;
        @memcpy(d.memory[@intCast(dst)..][0..bytes], d.memory[@intCast(src)..][0..bytes]);
        d.copies += 1;
    }
    fn fence(d: *Fake) !void {
        d.fences += 1;
        if (d.fail_fence) return error.FenceFailed;
    }
    fn fill(d: *Fake, address: u64, value: u8, bytes: usize) !void {
        @memset(d.memory[@intCast(address)..][0..bytes], value);
    }
};

fn fixture() !state.StateLayout {
    return state.StateLayout.init(&.{.{ .ratio = 2, .kv = true, .idx = true, .raw = true }},
        .{ .block_tokens = 4, .window = 4, .ring_extra = 2, .raw_rows = 4, .head_dim = 16, .index_head_dim = 32 }, .replicated);
}

test "private checkpoint survives source erasure and restores without touching neighbors" {
    const layout = try fixture();
    const bytes = try Plan.size(&layout, 4);
    var dev: Fake = .{};
    const plan = try Plan.make(&dev, &layout, 1, 4, 4, 40000, bytes);
    for (plan.spans[0..plan.len], 0..) |span, i|
        @memset(dev.memory[@intCast(span.live)..][0..span.bytes], @intCast(i + 1));
    const before = dev.memory;
    try plan.copy(&dev, true);
    for (plan.spans[0..plan.len]) |span| @memset(dev.memory[@intCast(span.live)..][0..span.bytes], 0);
    try plan.copy(&dev, false);
    try erase(&dev, 40000, bytes);
    for (dev.memory, 0..) |byte, i| {
        const expected: u8 = if (i >= 40000 and i < 40000 + bytes) 0 else before[i];
        try std.testing.expectEqual(expected, byte);
    }
    try std.testing.expectEqual(@as(usize, 4), dev.fences);
}

test "private checkpoint rejects invalid or aliased storage before a copy" {
    const layout = try fixture();
    const bytes = try Plan.size(&layout, 4);
    var dev: Fake = .{};
    try std.testing.expectError(error.BadPrivateCheckpoint, Plan.make(&dev, &layout, 4, 4, 4, 40000, bytes));
    try std.testing.expectError(error.BadPrivateCheckpoint, Plan.make(&dev, &layout, 0, 3, 4, 40000, bytes));
    try std.testing.expectError(error.BadPrivateCheckpoint, Plan.make(&dev, &layout, 0, 16, 4, 40000, bytes));
    try std.testing.expectError(error.BadPrivateCheckpoint, Plan.make(&dev, &layout, 0, 4, 4, 40000, bytes - 1));
    try std.testing.expectError(error.PrivateCheckpointAliasesLive, Plan.make(&dev, &layout, 0, 4, 4, 96, bytes));
    dev.bad_plane = true;
    try std.testing.expectError(error.BadPrivateCheckpoint, Plan.make(&dev, &layout, 0, 4, 4, 40000, bytes));
    try std.testing.expectEqual(@as(usize, 0), dev.copies);
}

test "private checkpoint fences partial failure and propagates a failed fence" {
    const layout = try fixture();
    const bytes = try Plan.size(&layout, 4);
    var dev: Fake = .{ .fail_copy = 1 };
    const plan = try Plan.make(&dev, &layout, 0, 4, 4, 40000, bytes);
    try std.testing.expectError(error.CopyFailed, plan.copy(&dev, true));
    try std.testing.expectEqual(@as(usize, 1), dev.fences);
    try erase(&dev, 40000, bytes);
    try std.testing.expect(std.mem.allEqual(u8, dev.memory[40000..][0..bytes], 0));
    dev.fail_fence = true;
    try std.testing.expectError(error.FenceFailed, erase(&dev, 40000, bytes));
}
