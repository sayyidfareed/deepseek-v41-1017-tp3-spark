// Modified for the DeepSeek 1017 TP3 integration; upstream copyrights retained. See CREDITS.md and provenance.json.
//! Engine-thread-only private cache ownership. Device state belongs to `ops`;
//! this table never uses the ordinary Store, disk tier, logs or public counters.
const std = @import("std");

pub const Scope = [32]u8;
pub const Lease = struct { index: usize, generation: u64 };
pub const Point = struct { id: u64, at: usize, bytes: usize };
pub const max_entries = 8;
pub const idle_ns: u64 = 15 * 60 * std.time.ns_per_s;

pub const Cache = struct {
    allocator: std.mem.Allocator,
    budget: usize,
    used: usize = 0,
    generation: u64 = 0,
    entries: [max_entries]Entry = @splat(.{}),

    const Entry = struct {
        scope: Scope = @splat(0),
        generation: u64 = 0,
        active: usize = 0,
        touched: u64 = 0,
        point: ?Point = null,
        tokens: []u32 = &.{},
    };

    fn same(a: *const Scope, b: *const Scope) bool {
        return std.crypto.timing_safe.eql(Scope, a.*, b.*);
    }

    fn entry(c: *Cache, lease: Lease) ?*Entry {
        if (lease.index >= c.entries.len or lease.generation == 0) return null;
        const e = &c.entries[lease.index];
        return if (e.generation == lease.generation) e else null;
    }

    /// Revocation precedes the checked device erase. On an erase failure the
    /// allocation stays owned, and the caller must fail the engine closed.
    fn clear(c: *Cache, ops: anytype, i: usize) !usize {
        const e = &c.entries[i];
        e.generation = 0;
        const n: usize = if (e.point != null) 1 else 0;
        if (e.point) |p| {
            try ops.drop(p.id);
            c.used -= p.bytes;
        }
        std.crypto.secureZero(u32, e.tokens);
        c.allocator.free(e.tokens);
        std.crypto.secureZero(u8, std.mem.asBytes(e));
        e.* = .{};
        return n;
    }

    pub fn forget(c: *Cache, ops: anytype, scope: ?*const Scope) !usize {
        var count: usize = 0;
        for (&c.entries, 0..) |*e, i| {
            // A failed erase keeps its point owned even after revocation.
            if (e.generation == 0 and e.point == null) continue;
            if (scope) |s| if (!same(&e.scope, s)) continue;
            count += try c.clear(ops, i);
        }
        return count;
    }

    pub fn expire(c: *Cache, ops: anytype, now: u64) !void {
        for (&c.entries, 0..) |*e, i| {
            if (e.generation != 0 and e.active == 0 and now -| e.touched >= idle_ns)
                _ = try c.clear(ops, i);
        }
    }

    /// No slot or budget means no reuse, never disk fallback. Active namespaces
    /// are not evicted for admission; a forget can still revoke their leases.
    pub fn begin(c: *Cache, ops: anytype, scope: *const Scope, now: u64) !?Lease {
        if (c.budget == 0) return null;
        try c.expire(ops, now);
        var free: ?usize = null;
        var oldest: ?usize = null;
        for (&c.entries, 0..) |*e, i| {
            if (e.generation == 0) {
                if (e.point != null) return error.PrivateCacheEraseIncomplete;
                free = i;
            } else if (same(&e.scope, scope)) {
                e.active = try std.math.add(usize, e.active, 1);
                e.touched = now;
                return .{ .index = i, .generation = e.generation };
            } else if (e.active == 0 and (oldest == null or e.touched < c.entries[oldest.?].touched)) oldest = i;
        }
        const i = free orelse oldest orelse return null;
        if (c.entries[i].generation != 0) _ = try c.clear(ops, i);
        c.generation = try std.math.add(u64, c.generation, 1);
        c.entries[i] = .{ .scope = scope.*, .generation = c.generation, .active = 1, .touched = now };
        return .{ .index = i, .generation = c.generation };
    }

    pub fn end(c: *Cache, lease: Lease, now: u64) void {
        const e = c.entry(lease) orelse return;
        std.debug.assert(e.active > 0);
        e.active -= 1;
        e.touched = now;
    }

    pub fn lookup(c: *Cache, lease: Lease, tokens: []const u32, max_at: usize, now: u64) ?Point {
        const e = c.entry(lease) orelse return null;
        const p = e.point orelse return null;
        if (p.at > max_at or p.at >= tokens.len or p.at != e.tokens.len or
            !std.mem.eql(u32, e.tokens, tokens[0..p.at])) return null;
        e.touched = now;
        return p;
    }

    /// Make room BEFORE allocating/copying the replacement. This can erase the
    /// old point but never another active namespace's point. Charge device bytes
    /// plus the host token key; use checked arithmetic, including transient copies.
    pub fn reserve(c: *Cache, ops: anytype, lease: Lease, at: usize, device_bytes: usize) !bool {
        const e = c.entry(lease) orelse return false;
        const charge = try std.math.add(usize, device_bytes, try std.math.mul(usize, at, @sizeOf(u32)));
        if (charge > c.budget or at == 0) return false;
        if (e.point) |p| {
            try ops.drop(p.id);
            c.used -= p.bytes;
            e.point = null;
            std.crypto.secureZero(u32, e.tokens);
            c.allocator.free(e.tokens);
            e.tokens = &.{};
        }
        while (c.used > c.budget - charge) {
            var oldest: ?usize = null;
            for (&c.entries, 0..) |*other, i| {
                if (i == lease.index or other.generation == 0 or other.active != 0 or other.point == null) continue;
                if (oldest == null or other.touched < c.entries[oldest.?].touched) oldest = i;
            }
            _ = try c.clear(ops, oldest orelse return false);
        }
        return true;
    }

    /// Caller owns an all-rank, fully fenced checkpoint until this succeeds.
    /// On false/error it must erase that uncommitted checkpoint on every rank.
    pub fn publish(c: *Cache, lease: Lease, tokens: []const u32, id: u64, device_bytes: usize, now: u64) !bool {
        const e = c.entry(lease) orelse return false;
        if (e.point != null or tokens.len == 0) return error.PrivateCachePublishConflict;
        const charge = try std.math.add(usize, device_bytes, try std.math.mul(usize, tokens.len, @sizeOf(u32)));
        if (charge > c.budget or c.used > c.budget - charge) return false;
        const copy = try c.allocator.dupe(u32, tokens);
        e.tokens = copy;
        e.point = .{ .id = id, .at = tokens.len, .bytes = charge };
        c.used += charge;
        e.touched = now;
        return true;
    }

    pub fn revoke(c: *Cache, ops: anytype, lease: Lease) !void {
        if (c.entry(lease) != null) _ = try c.clear(ops, lease.index);
    }
};

const Fake = struct {
    dropped: usize = 0,
    fail: bool = false,
    pub fn drop(f: *Fake, _: u64) !void {
        if (f.fail) return error.EraseFailed;
        f.dropped += 1;
    }
};

test "private cache requires same scope and exact prefix and never resurrects a revoked lease" {
    var ops: Fake = .{};
    var c: Cache = .{ .allocator = std.testing.allocator, .budget = 4096 };
    defer _ = c.forget(&ops, null) catch unreachable;
    const a: Scope = @splat(1);
    const b: Scope = @splat(2);
    const la = (try c.begin(&ops, &a, 1)).?;
    try std.testing.expect(try c.reserve(&ops, la, 2, 64));
    try std.testing.expect(try c.publish(la, &.{ 10, 20 }, 1, 64, 2));
    try std.testing.expect(c.lookup(la, &.{ 10, 20, 30 }, 2, 3) != null);
    try std.testing.expect(c.lookup(la, &.{ 10, 21, 30 }, 2, 3) == null);
    try std.testing.expect(c.lookup(la, &.{ 10, 20, 30 }, 1, 3) == null);
    const lb = (try c.begin(&ops, &b, 3)).?;
    try std.testing.expect(c.lookup(lb, &.{ 10, 20, 30 }, 2, 3) == null);
    try std.testing.expectEqual(@as(usize, 1), try c.forget(&ops, &a));
    try std.testing.expect(!try c.publish(la, &.{ 10, 20 }, 2, 64, 4));
    const next = (try c.begin(&ops, &a, 5)).?;
    try std.testing.expect(next.generation != la.generation);
    try std.testing.expect(!try c.reserve(&ops, la, 2, 64));
    c.end(la, 5); // stale release cannot change a newer lease
    try std.testing.expectEqual(@as(usize, 1), c.entries[next.index].active);
    try std.testing.expectEqual(@as(usize, 1), c.entries[lb.index].active);
}

test "private cache charges host and device storage and expires only idle namespaces" {
    var ops: Fake = .{};
    var c: Cache = .{ .allocator = std.testing.allocator, .budget = 100 };
    defer _ = c.forget(&ops, null) catch unreachable;
    const a: Scope = @splat(1);
    const b: Scope = @splat(2);
    const la = (try c.begin(&ops, &a, 1)).?;
    try std.testing.expect(!try c.reserve(&ops, la, 2, 93));
    try std.testing.expect(try c.reserve(&ops, la, 2, 90));
    try std.testing.expect(try c.publish(la, &.{ 10, 20 }, 1, 90, 2));
    const lb = (try c.begin(&ops, &b, 3)).?;
    try std.testing.expect(!try c.reserve(&ops, lb, 2, 90));
    try c.expire(&ops, idle_ns + 20);
    try std.testing.expectEqual(@as(usize, 98), c.used);
    c.end(la, idle_ns + 21);
    try std.testing.expect(try c.reserve(&ops, lb, 2, 90));
    try std.testing.expectEqual(@as(usize, 0), c.used);
    try std.testing.expect(try c.publish(lb, &.{ 30, 40 }, 2, 90, idle_ns + 22));
    c.end(lb, idle_ns + 23);
    try c.expire(&ops, 2 * idle_ns + 23);
    try std.testing.expectEqual(@as(usize, 0), c.used);
    try std.testing.expectEqual(@as(usize, 2), ops.dropped);
}

test "private cache keeps failed erasure owned and invalidates reuse before reporting failure" {
    var ops: Fake = .{};
    var c: Cache = .{ .allocator = std.testing.allocator, .budget = 100 };
    const a: Scope = @splat(1);
    const l = (try c.begin(&ops, &a, 1)).?;
    try std.testing.expect(try c.publish(l, &.{ 10, 20 }, 1, 64, 2));
    ops.fail = true;
    try std.testing.expectError(error.EraseFailed, c.forget(&ops, null));
    try std.testing.expect(c.lookup(l, &.{ 10, 20, 30 }, 2, 3) == null);
    try std.testing.expectEqual(@as(usize, 72), c.used);
    ops.fail = false;
    try std.testing.expectEqual(@as(usize, 1), try c.forget(&ops, null));
    try std.testing.expectEqual(@as(usize, 0), c.used);
}
