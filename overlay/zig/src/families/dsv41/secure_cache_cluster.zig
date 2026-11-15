// Modified for the DeepSeek 1017 TP3 integration; upstream copyrights retained. See CREDITS.md and provenance.json.
//! Separate, checked all-rank RAM checkpoints. Never enters ordinary snapshot
//! or spill ownership. Called only on the engine thread between GPU operations.
const std = @import("std");
const cuda = @import("cuda");
const model = @import("model.zig");
const link = @import("link.zig");
const checkpoint = @import("secure_checkpoint.zig");
pub const ownership = @import("secure_cache.zig");
const max_entries = ownership.max_entries;

const Device = struct {
    m: *model.Model,
    pub fn slots(d: Device) usize { return d.m.streams; }
    pub fn capacity(d: Device) usize { return d.m.caches.cap; }
    pub fn blockPlane(d: Device, p: model.Model.spill.Plane) u64 { return (model.Model.SpillDev{ .m = d.m }).blockPlane(p); }
    pub fn slotPlane(d: Device, p: model.Model.spill.Plane, slot: usize) u64 { return (model.Model.SpillDev{ .m = d.m }).slotPlane(p, slot); }
    pub fn copy(d: Device, dst: u64, src: u64, bytes: usize) !void {
        try d.m.ctx.d.check(d.m.ctx.d.api.cuMemcpyDtoDAsync_v2(dst, src, bytes, d.m.stream.handle), "cuMemcpyDtoDAsync");
    }
    pub fn fill(d: Device, dst: u64, value: u8, bytes: usize) !void { try (model.Model.SpillDev{ .m = d.m }).fill(dst, value, bytes); }
    pub fn fence(d: Device) !void { try d.m.stream.synchronize(); }
};

pub const Rank = struct {
    budget: usize = 0,
    used: usize = 0,
    slots: [max_entries]?Saved = @splat(null),
    const Saved = struct { id: u64, at: usize, buf: cuda.DeviceBuffer, bytes: usize };

    pub fn save(r: *Rank, m: *model.Model, id: u64, slot: usize, base: usize, at: usize) !bool {
        if (id == 0) return error.BadPrivateCheckpoint;
        var free: ?usize = null;
        for (r.slots, 0..) |saved, i| {
            if (saved) |s| { if (s.id == id) return error.DuplicatePrivateCheckpoint; } else free = i;
        }
        const index = free orelse return false;
        const layout = try m.spillLayout();
        const bytes = try checkpoint.Plan.size(&layout, at);
        if (bytes > r.budget or r.used > r.budget - bytes) return false;
        var buf = cuda.DeviceBuffer.alloc(m.ctx.d, bytes) catch |err| switch (err) {
            error.OutOfDeviceMemory => return false,
            else => return err,
        };
        // This allocation must be erased even if the plan or partial copy fails.
        errdefer {
            checkpoint.erase(Device{ .m = m }, buf.ptr, bytes) catch std.c._exit(70);
            buf.freeChecked() catch std.c._exit(70);
        }
        const plan = try checkpoint.Plan.make(Device{ .m = m }, &layout, slot, base, at, buf.ptr, bytes);
        try plan.copy(Device{ .m = m }, true);
        r.slots[index] = .{ .id = id, .at = at, .buf = buf, .bytes = bytes };
        r.used += bytes;
        return true;
    }

    pub fn restore(r: *Rank, m: *model.Model, id: u64, slot: usize, base: usize, at: usize) !void {
        for (r.slots) |saved| if (saved) |s| if (s.id == id) {
            if (at != s.at) return error.PrivateCheckpointMismatch;
            const layout = try m.spillLayout();
            const plan = try checkpoint.Plan.make(Device{ .m = m }, &layout, slot, base, at, s.buf.ptr, s.bytes);
            return plan.copy(Device{ .m = m }, false);
        };
        return error.PrivateCheckpointMissing;
    }

    pub fn drop(r: *Rank, m: *model.Model, id: u64) !void {
        for (&r.slots) |*saved| if (saved.*) |s| if (id == 0 or s.id == id) {
            try checkpoint.erase(Device{ .m = m }, s.buf.ptr, s.bytes);
            var buf = s.buf;
            try buf.freeChecked();
            r.used -= s.bytes;
            saved.* = null;
        };
    }
};

// A receipt binds operation, ordered frame, checkpoint and physical rank. It
// contains no scope, prompt tokens, timings or other request metadata.
pub fn receipt(op: u8, epoch: u64, id: u64, rank: u32) [21]u8 {
    var out: [21]u8 = undefined;
    out[0] = op;
    std.mem.writeInt(u64, out[1..9], epoch, .little);
    std.mem.writeInt(u64, out[9..17], id, .little);
    std.mem.writeInt(u32, out[17..21], rank, .little);
    return out;
}

fn acknowledged(ln: anytype, op: u8, epoch: u64, id: u64) !bool {
    var complete = true;
    const deadline = link.Link.monotonicNs() +| 30 * std.time.ns_per_s;
    for (ln.m.followers(), 1..) |peer, rank| {
        var packet: [21]u8 = undefined;
        const got = try (peer orelse return error.PrivateRankMissing).recvDeadline(&packet, deadline);
        if (!try checkReceipt(got, op, epoch, id, @intCast(rank))) complete = false;
    }
    return complete;
}

pub fn checkReceipt(got: []const u8, op: u8, epoch: u64, id: u64, rank: u32) !bool {
    const expected = receipt(op, epoch, id, rank);
    if (std.mem.eql(u8, got, &expected)) return true;
    const refused = receipt(op | 0x80, epoch, id, rank);
    if (op == 18 and std.mem.eql(u8, got, &refused)) return false;
    return error.PrivateCacheAcknowledgementMismatch;
}

pub fn Dropper(comptime L: type) type {
    return struct {
        ln: *L,
        pub fn drop(d: @This(), id: u64) !void { try dropPoint(d.ln, id); }
    };
}

pub fn dropPoint(ln: anytype, id: u64) !void {
    ln.wire.protected = true;
    const epoch = ln.step;
    if (try ln.begin(.private_cache_drop)) {
        try ln.wire.int(ln.gpa, u64, id);
        try ln.flush();
    }
    try ln.private_points.drop(ln.m, id);
    _ = try acknowledged(ln, 20, epoch, id);
}

pub fn save(ln: anytype, s: anytype, at: usize) !void {
    const lane = ln.streams.get(s) orelse return error.UnknownStream;
    const lease = lane.cache_lease orelse return;
    if (s.secure_cache_revoked()) return;
    errdefer std.c._exit(70);
    const bytes = try checkpoint.Plan.size(&(try ln.m.spillLayout()), at);
    const ops: Dropper(@TypeOf(ln.*)) = .{ .ln = ln };
    if (!try ln.private_cache.reserve(ops, lease, at, bytes)) return;
    const id = ln.next_private_point;
    ln.next_private_point = try std.math.add(u64, id, 1);
    // From the first frame onward any partial all-rank failure is fatal. Do not
    // continue ordinary work with an uncertain peer command/cleanup state.
    ln.wire.protected = true;
    const epoch = ln.step;
    if (try ln.begin(.private_cache_save)) {
        try ln.wire.int(ln.gpa, u64, id);
        try ln.wire.int(ln.gpa, u32, @intCast(lane.slot));
        try ln.wire.int(ln.gpa, u64, lane.secure_epoch);
        try ln.wire.int(ln.gpa, u64, lane.base);
        try ln.wire.int(ln.gpa, u64, at);
        try ln.flush();
    }
    const local = try ln.private_points.save(ln.m, id, lane.slot, lane.base, at);
    const peers = try acknowledged(ln, 18, epoch, id);
    if (!local or !peers) return dropPoint(ln, id);
    if (s.secure_cache_revoked()) return dropPoint(ln, id);
    const published = ln.private_cache.publish(lease, s.prompt()[0..at], id, bytes, ln.m.now()) catch |err| {
        try dropPoint(ln, id);
        if (err == error.OutOfMemory) return;
        return err;
    };
    if (!published) try dropPoint(ln, id);
}

pub fn restore(ln: anytype, s: anytype, max_at: usize) !usize {
    const lane = ln.streams.get(s) orelse return error.UnknownStream;
    const lease = lane.cache_lease orelse return 0;
    if (s.secure_cache_revoked()) return 0;
    const point = ln.private_cache.lookup(lease, s.prompt(), max_at, ln.m.now()) orelse return 0;
    errdefer std.c._exit(70);
    ln.wire.protected = true;
    const epoch = ln.step;
    if (try ln.begin(.private_cache_restore)) {
        try ln.wire.int(ln.gpa, u64, point.id);
        try ln.wire.int(ln.gpa, u32, @intCast(lane.slot));
        try ln.wire.int(ln.gpa, u64, lane.secure_epoch);
        try ln.wire.int(ln.gpa, u64, lane.base);
        try ln.wire.int(ln.gpa, u64, point.at);
        try ln.flush();
    }
    try ln.private_points.restore(ln.m, point.id, lane.slot, lane.base, point.at);
    _ = try acknowledged(ln, 19, epoch, point.id);
    return point.at;
}

/// Follower commands validate live request ownership before touching its state.
pub fn follow(r: *Rank, m: *model.Model, reader: anytype, owners: anytype, op: u8, epoch: u64) !void {
    const id = try reader.int(u64);
    var accepted = true;
    if (op == 20) {
        if (reader.rest().len != 0) return error.BadFrame;
        try r.drop(m, id);
    } else {
        const slot = try reader.int(u32);
        const owner_epoch = try reader.int(u64);
        const base = try reader.int(u64);
        const at = try reader.int(u64);
        if (reader.rest().len != 0 or slot >= owners.len or slot >= m.streams) return error.BadFrame;
        const owner = owners[slot] orelse return error.PrivateOwnerMismatch;
        if (owner.epoch != owner_epoch or owner.lo != base or at == 0 or at > owner.hi - owner.lo)
            return error.PrivateOwnerMismatch;
        if (op == 18) accepted = try r.save(m, id, slot, @intCast(base), @intCast(at))
        else if (op == 19) try r.restore(m, id, slot, @intCast(base), @intCast(at))
        else return error.BadFrame;
    }
    const ack = receipt(if (accepted) op else op | 0x80, epoch, id, @intCast(m.rank));
    try m.peer().sendDeadline(&ack, link.Link.monotonicNs() +| 30 * std.time.ns_per_s);
}

fn environment(name: [:0]const u8) ?[]const u8 { return std.mem.span(std.c.getenv(name) orelse return null); }
pub fn checkShared(self: anytype) !void {
        if (self.shared_private_ready) return;
        if (self.private_request or self.streams.count() != 0) return error.SecureBackendBusy;
        if (self.kv != null or self.m.kvs != null) return error.PrivateShardedLayoutUnsupported;
        if (self.prof != null or self.m.prof or self.m.timer != null) return error.SecureTracingEnabled;
        _ = try @import("secure_policy.zig").Guard.dormant();
        try self.m.comm.checkSharedPrivate();
        const budget = if (environment("TF_SECURE_CACHE_BYTES")) |s| try std.fmt.parseInt(usize, s, 10) else 0;
        self.private_cache.budget = budget;
        self.private_points.budget = budget;
        const epoch = self.step;
        if (try self.begin(.private_check)) {
            try self.wire.int(self.gpa, u64, budget);
            try self.flush();
            const deadline = link.Link.monotonicNs() +| 30 * std.time.ns_per_s;
            for (self.m.followers(), 1..) |peer, rank| {
                var packet: [13]u8 = undefined;
                const got = try (peer orelse return error.PrivateRankMissing).recvDeadline(&packet, deadline);
                try @import("secure_extent.zig").checkReadinessAcknowledgement(got, epoch, @intCast(rank));
            }
        }
        self.shared_private_ready = true;

}

test "secure cache acknowledgements bind rank operation checkpoint and ordered frame" {
    const good = receipt(18, 6, 42, 2);
    try std.testing.expect(try checkReceipt(&good, 18, 6, 42, 2));
    const full = receipt(18 | 0x80, 6, 42, 2);
    try std.testing.expect(!try checkReceipt(&full, 18, 6, 42, 2));
    for ([_][21]u8{ receipt(19, 6, 42, 2), receipt(18, 7, 42, 2), receipt(18, 6, 43, 2), receipt(18, 6, 42, 1) }) |bad|
        try std.testing.expectError(error.PrivateCacheAcknowledgementMismatch, checkReceipt(&bad, 18, 6, 42, 2));
    const not_erased = receipt(20 | 0x80, 6, 42, 2);
    try std.testing.expectError(error.PrivateCacheAcknowledgementMismatch, checkReceipt(&not_erased, 20, 6, 42, 2));
    try std.testing.expectError(error.PrivateCacheAcknowledgementMismatch, checkReceipt(good[0..20], 18, 6, 42, 2));
}
