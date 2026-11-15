// Modified for the DeepSeek 1017 TP3 integration; upstream copyrights retained. See CREDITS.md and provenance.json.
//! Targeted erasure of one replicated KV extent and one stream slot. This does
//! not erase shared scratch or certify a secure request by itself. The caller
//! must own the extent, guard spill readers and synchronize before releasing it.
const std = @import("std");
const state = @import("spill_state.zig");

pub const Span = struct { address: u64, bytes: usize, value: u8 = 0 };
pub const Plan = struct {
    spans: [state.max_planes + 8]Span = undefined,
    len: usize = 0,

    fn add(p: *Plan, span: Span) !void {
        if (span.address == 0 or span.bytes == 0 or p.len == p.spans.len) return error.BadPrivateRange;
        _ = std.math.add(u64, span.address, span.bytes) catch return error.BadPrivateRange;
        p.spans[p.len] = span;
        p.len += 1;
    }

    /// Build and validate the entire operation before issuing any device write.
    /// Extra spans are slot-local drafter rings, supplied by their owning pool.
    pub fn make(dev: anytype, layout: *const state.StateLayout, slot: usize, lo: usize, hi: usize, extra: []const Span) !Plan {
        if (layout.owner_layout != .replicated) return error.PrivateShardedLayoutUnsupported;
        if (slot >= dev.slots() or lo >= hi or hi > dev.capacity()) return error.BadPrivateRange;
        if (layout.geo.block_tokens == 0 or lo % layout.geo.block_tokens != 0 or hi % layout.geo.block_tokens != 0) return error.BadPrivateRange;
        var p: Plan = .{};
        for (layout.blockPlanes()) |plane| {
            if (plane.ratio == 0 or lo % plane.ratio != 0 or hi % plane.ratio != 0) return error.BadPrivateRange;
            const offset = std.math.mul(usize, lo / plane.ratio, plane.row) catch return error.BadPrivateRange;
            const bytes = std.math.mul(usize, (hi - lo) / plane.ratio, plane.row) catch return error.BadPrivateRange;
            const address = std.math.add(u64, dev.blockPlane(plane), offset) catch return error.BadPrivateRange;
            if (dev.blockPlane(plane) == 0) return error.BadPrivateRange;
            try p.add(.{ .address = address, .bytes = bytes, .value = plane.sentinel });
        }
        for (layout.smallPlanes()) |plane| {
            const bytes = std.math.mul(usize, plane.rows, plane.row) catch return error.BadPrivateRange;
            try p.add(.{ .address = dev.slotPlane(plane, slot), .bytes = bytes, .value = plane.sentinel });
        }
        for (extra) |span| try p.add(span);
        return p;
    }

    /// A failure propagates; the caller must not release/reassign the allocation
    /// on partial erasure or acknowledge completion before device synchronization.
    pub fn apply(p: *const Plan, dev: anytype) !void {
        for (p.spans[0..p.len]) |span| try dev.fill(span.address, span.value, span.bytes);
    }
};

const Fake = struct {
    bytes: [32768]u8 = @splat(0xa5),
    writes: usize = 0,
    fail_at: ?usize = null,
    bad_plane: bool = false,
    fn slots(_: *Fake) usize { return 4; }
    fn capacity(_: *Fake) usize { return 16; }
    fn blockPlane(d: *Fake, p: state.Plane) u64 {
        return if (d.bad_plane and p.kind == .idx_codes) 0 else 64 + @as(u64, @intFromEnum(p.kind)) * 2048;
    }
    fn slotPlane(_: *Fake, p: state.Plane, s: usize) u64 {
        return 64 + @as(u64, @intFromEnum(p.kind)) * 2048 + s * p.slotBytes();
    }
    fn fill(d: *Fake, address: u64, value: u8, bytes: usize) !void {
        if (d.fail_at == d.writes) return error.DeviceFailed;
        if (address + bytes > d.bytes.len) return error.BadFakeAddress;
        @memset(d.bytes[@intCast(address)..][0..bytes], value);
        d.writes += 1;
    }
};

fn fixture() !state.StateLayout {
    return state.StateLayout.init(&.{.{ .ratio = 2, .kv = true, .idx = true, .raw = true }},
        .{ .block_tokens = 4, .window = 4, .ring_extra = 2, .raw_rows = 4, .head_dim = 16, .index_head_dim = 32 }, .replicated);
}

test "secure extent erases only its KV rows and slot including drafter rings" {
    const layout = try fixture();
    var dev: Fake = .{};
    const plan = try Plan.make(&dev, &layout, 1, 4, 8, &.{.{ .address = 24000, .bytes = 128 }});
    try plan.apply(&dev);
    try std.testing.expectEqual(plan.len, dev.writes);
    for (dev.bytes, 0..) |byte, i| {
        var expected: u8 = 0xa5;
        for (plan.spans[0..plan.len]) |span| {
            if (i >= span.address and i < span.address + span.bytes) expected = span.value;
        }
        try std.testing.expectEqual(expected, byte);
    }
    try std.testing.expectEqual(@as(usize, 8), plan.len);
}

test "secure extent validates all ranges and addresses before touching neighbors" {
    const layout = try fixture();
    var dev: Fake = .{};
    for ([_][3]usize{ .{ 4, 4, 8 }, .{ 1, 8, 4 }, .{ 1, 4, 20 }, .{ 1, 3, 8 }, .{ 1, 4, 7 } }) |bad| {
        try std.testing.expectError(error.BadPrivateRange, Plan.make(&dev, &layout, bad[0], bad[1], bad[2], &.{}));
    }
    dev.bad_plane = true;
    try std.testing.expectError(error.BadPrivateRange, Plan.make(&dev, &layout, 1, 4, 8, &.{}));
    dev.bad_plane = false;
    try std.testing.expectError(error.BadPrivateRange, Plan.make(&dev, &layout, 1, 4, 8, &.{.{ .address = std.math.maxInt(u64), .bytes = 2 }}));
    var sharded = layout;
    sharded.owner_layout = .owner_sharded;
    try std.testing.expectError(error.PrivateShardedLayoutUnsupported, Plan.make(&dev, &sharded, 1, 4, 8, &.{}));
    try std.testing.expectEqual(@as(usize, 0), dev.writes);
    try std.testing.expect(std.mem.allEqual(u8, &dev.bytes, 0xa5));
}

test "secure extent returns partial device failures instead of successful cleanup" {
    const layout = try fixture();
    var dev: Fake = .{ .fail_at = 2 };
    const plan = try Plan.make(&dev, &layout, 1, 4, 8, &.{});
    try std.testing.expectError(error.DeviceFailed, plan.apply(&dev));
    try std.testing.expectEqual(@as(usize, 2), dev.writes);
}


test "secure extent model integration compiles without a GPU call" {
    const model = @import("model.zig");
    const Check = struct {
        fn run(m: *model.Model, slot: usize, lo: usize, hi: usize) anyerror!void {
            try m.eraseStreamStorage(slot, lo, hi);
            try m.eraseSharedRoundScratch();
        }
    };
    const call: *const fn (*model.Model, usize, usize, usize) anyerror!void = &Check.run;
    try std.testing.expect(@intFromPtr(call) != 0);
}

/// Identity retained only while a private stream owns its slot. The ordered
/// rank-frame epoch distinguishes successive uses of the same address range.
pub const Ownership = struct {
    epoch: u64,
    lo: usize,
    hi: usize,
    pub fn check(o: Ownership, epoch: u64, lo: usize, hi: usize) !void {
        if (o.epoch != epoch or o.lo != lo or o.hi != hi or lo >= hi) return error.PrivateOwnerMismatch;
    }
};

pub fn acknowledgement(epoch: u64, slot: u32) [13]u8 {
    var packet: [13]u8 = undefined;
    packet[0] = 0x74;
    std.mem.writeInt(u64, packet[1..9], epoch, .little);
    std.mem.writeInt(u32, packet[9..13], slot, .little);
    return packet;
}

pub fn checkAcknowledgement(packet: []const u8, epoch: u64, slot: u32) !void {
    const expected = acknowledgement(epoch, slot);
    if (!std.mem.eql(u8, packet, &expected)) return error.PrivateEraseNotAcknowledged;
}

pub const CleanupPhase = enum(u8) { fenced = 0x76, erased = 0x74, resumed = 0x77 };
pub fn phaseAcknowledgement(phase: CleanupPhase, epoch: u64, slot: u32) [13]u8 {
    var packet = acknowledgement(epoch, slot);
    packet[0] = @intFromEnum(phase);
    return packet;
}
pub fn checkPhaseAcknowledgement(packet: []const u8, phase: CleanupPhase, epoch: u64, slot: u32) !void {
    const expected = phaseAcknowledgement(phase, epoch, slot);
    if (!std.mem.eql(u8, packet, &expected)) return error.PrivateCleanupPhaseMismatch;
}

test "secure extent cleanup phases cannot substitute for each other or another owner" {
    for ([_]CleanupPhase{ .fenced, .erased, .resumed }) |phase| {
        const packet = phaseAcknowledgement(phase, 0, 2);
        try checkPhaseAcknowledgement(&packet, phase, 0, 2);
        for ([_]CleanupPhase{ .fenced, .erased, .resumed }) |other| if (other != phase) {
            try std.testing.expectError(error.PrivateCleanupPhaseMismatch, checkPhaseAcknowledgement(&packet, other, 0, 2));
        };
        try std.testing.expectError(error.PrivateCleanupPhaseMismatch, checkPhaseAcknowledgement(&packet, phase, 1, 2));
        try std.testing.expectError(error.PrivateCleanupPhaseMismatch, checkPhaseAcknowledgement(&packet, phase, 0, 1));
        try std.testing.expectError(error.PrivateCleanupPhaseMismatch, checkPhaseAcknowledgement(packet[0..12], phase, 0, 2));
    }
}

test "secure extent cleanup refuses stale ownership and mismatched acknowledgements" {
    const owner: Ownership = .{ .epoch = 7, .lo = 4096, .hi = 8192 };
    try owner.check(7, 4096, 8192);
    try std.testing.expectError(error.PrivateOwnerMismatch, owner.check(6, 4096, 8192));
    try std.testing.expectError(error.PrivateOwnerMismatch, owner.check(7, 0, 4096));
    const ack = acknowledgement(7, 2);
    try checkAcknowledgement(&ack, 7, 2);
    try std.testing.expectError(error.PrivateEraseNotAcknowledged, checkAcknowledgement(&ack, 8, 2));
    try std.testing.expectError(error.PrivateEraseNotAcknowledged, checkAcknowledgement(&ack, 7, 1));
    try std.testing.expectError(error.PrivateEraseNotAcknowledged, checkAcknowledgement(ack[0..12], 7, 2));
}

// A readiness receipt is distinct from a request-erasure receipt.
pub fn readinessAcknowledgement(epoch: u64, rank: u32) [13]u8 {
    var packet = acknowledgement(epoch, rank);
    packet[0] = 0x75;
    return packet;
}
pub fn checkReadinessAcknowledgement(packet: []const u8, epoch: u64, rank: u32) !void {
    const expected = readinessAcknowledgement(epoch, rank);
    if (!std.mem.eql(u8, packet, &expected)) return error.PrivateRankNotReady;
}
test "secure extent readiness requires exact epoch rank and receipt kind" {
    const packet = readinessAcknowledgement(7, 1);
    try checkReadinessAcknowledgement(&packet, 7, 1);
    try std.testing.expectError(error.PrivateRankNotReady, checkReadinessAcknowledgement(&packet, 8, 1));
    try std.testing.expectError(error.PrivateRankNotReady, checkReadinessAcknowledgement(&packet, 7, 2));
    try std.testing.expectError(error.PrivateRankNotReady, checkReadinessAcknowledgement(packet[0..12], 7, 1));
    const erased = acknowledgement(7, 1);
    try std.testing.expectError(error.PrivateRankNotReady, checkReadinessAcknowledgement(&erased, 7, 1));
}
