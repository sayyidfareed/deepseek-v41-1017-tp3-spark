// Modified for the DeepSeek 1017 TP3 integration; upstream copyrights retained. See CREDITS.md and provenance.json.
//! Smart CUDA Graph Controller 4.0.3: bounded CPU-only reuse/cost policy.
//! No CUDA calls, prompt text, timers, file I/O or allocation on a replay.
//! Admission/collective ownership contracts inspired by TensorFold graph_cache
//! ba7b2c4038011722356f475f7be633f60ef3c83a (Apache-2.0); policy is our extension.
const std = @import("std");
pub const schema: u32 = 3;
pub const max_resident = 1024;
pub const max_ghost = 512;
pub const max_victims = 8;
pub const mib: u64 = 1 << 20;
pub const Kind = enum(u8) { none = 0, round = 1, draft = 2 };
pub const Group = struct {
    kind: Kind = .none,
    rows: u32 = 0,
    bucket: u64 = 0,
    steps: u32 = 0,
    split: bool = false,
    pub fn valid(k: Group) bool {
        return switch (k.kind) {
            .none => k.rows == 0 and k.bucket == 0 and k.steps == 0 and !k.split,
            .round => k.rows > 0 and k.rows <= 48 and k.bucket > 0 and k.steps == 0 and !k.split,
            .draft => k.rows > 0 and k.rows <= 16 and k.steps > 0 and k.steps <= 8 and k.bucket == 0,
        };
    }
    pub fn order(a: Group, b: Group) bool {
        if (a.kind != b.kind) return @backingInt(a.kind) < @backingInt(b.kind);
        if (a.rows != b.rows) return a.rows < b.rows;
        if (a.bucket != b.bucket) return a.bucket < b.bucket;
        if (a.steps != b.steps) return a.steps < b.steps;
        return !a.split and b.split;
    }
    pub fn hash(k: Group) u64 {
        // Explicit scalar encoding: never hash padding or process addresses.
        var h = std.hash.Wyhash.init(0x403);
        inline for (.{ @backingInt(k.kind), k.rows, k.bucket, k.steps, @intFromBool(k.split) }) |v| {
            var n: [8]u8 = undefined;
            std.mem.writeInt(u64, &n, @intCast(v), .little);
            h.update(&n);
        }
        return h.final();
    }
};
pub const Leaf = struct {
    group: Group = .{},
    piece: u32 = 0,
    greedy: bool = false,
    pub fn valid(k: Leaf) bool {
        return k.group.valid() and k.piece < 32 and (k.group.kind != .draft or (k.piece == 0 and !k.greedy));
    }
    pub fn mask(k: Leaf) u64 {
        return @as(u64, 1) << @as(u6, @intCast(2 * k.piece + @intFromBool(k.greedy)));
    }
};
pub const Config = struct {
    enabled: bool = false,
    reserve: u64 = 1024 * mib,
    hysteresis: u64 = 256 * mib,
    transient: u64 = 64 * mib,
    unknown_peak: u64 = 64 * mib,
    limit: u32 = max_resident,
    poll_primitives: u32 = 128,
    trim: bool = true,
    pub fn valid(c: Config) bool {
        return c.reserve >= 256 * mib and c.reserve <= 8192 * mib and c.hysteresis >= 64 * mib and
            c.transient >= 16 * mib and c.unknown_peak >= mib and c.limit > 0 and c.limit <= max_resident and
            c.poll_primitives >= 16 and c.poll_primitives <= 4096;
    }
    pub fn digest(c: Config, profile: u64, world: u32) u64 {
        var h = std.hash.Wyhash.init(profile);
        inline for (.{ schema, @intFromBool(c.enabled), c.reserve, c.hysteresis, c.transient, c.unknown_peak, c.limit, c.poll_primitives, @intFromBool(c.trim), world }) |v| {
            var b: [8]u8 = undefined;
            std.mem.writeInt(u64, &b, @intCast(v), .little);
            h.update(&b);
        }
        return h.final();
    }
};
pub const Record = struct {
    last: u64 = 0,
    epoch: u64 = 0,
    frequency: u32 = 1,
    gap: u32 = 128,
    returns: u32 = 0,
    captures: u32 = 0,
    recaptures: u32 = 0,
    rebuild_ns: u64 = 1000000,
    bytes: u64 = 16 * mib,
    byte_samples: u32 = 0,
    build_per_leaf_ns: u64 = 1000000,
    build_samples: u32 = 0,
    leaves: u64 = 0,
    protected_until: u64 = 0,
    evicted_at: u64 = 0,
    pub fn aged(r: Record, clock: u64) u32 {
        const n = clock / 4096 -| r.epoch;
        return if (n >= 32) 0 else r.frequency >> @as(u5, @intCast(n));
    }
    pub fn probability(r: Record, clock: u64) u64 {
        const age = clock -| r.last;
        const gap = @max(1, r.gap);
        const frequency: u64 = r.aged(clock);
        // Bounded smoothed prior with recency and observed return interval.
        const evidence: u64 = @min(32, frequency + @min(8, r.recaptures));
        const p = (evidence + 1) * 65536 / (evidence + 4);
        return p * 512 / @max(512, age +| @as(u64, gap));
    }
    pub fn value(r: Record, clock: u64) u64 {
        // P(return)*measured reconstruction cost per MiB. No eager savings
        // are added here: capture vs eager is a DIFFERENT alternative.
        const cost = @min(r.rebuild_ns, 60 * std.time.ns_per_s);
        return r.probability(clock) * cost / @max(mib, r.bytes) * mib / 65536;
    }
};
pub const Candidate = struct { key: Group, value: u64, bytes: u64 };
pub const Plan = struct { keys: [max_victims]Group = @splat(.{}), n: u8 = 0, estimated_bytes: u64 = 0 };
pub const Calibration = struct {
    key: Leaf,
    eager_ns: u64,
    replay_ns: u64,
    samples: u32,
    pub fn valid(c: Calibration) bool {
        return c.key.valid() and c.key.group.kind != .none and c.samples >= 8 and c.eager_ns > c.replay_ns and c.eager_ns < 60 * std.time.ns_per_s;
    }
};
pub const Ledger = struct {
    const Map = std.AutoHashMapUnmanaged(Group, Record);
    gpa: std.mem.Allocator,
    resident: Map = .empty,
    ghost: Map = .empty,
    clock: u64 = 0,
    generation: u64 = 0,
    hits: u64 = 0,
    misses: u64 = 0,
    captures: u64 = 0,
    eager: u64 = 0,
    evictions: u64 = 0,
    ghost_returns: u64 = 0,
    quick_returns: u64 = 0,
    capture_ns: u64 = 0,
    control_ns: u64 = 0,
    polls: u64 = 0,
    destroy_ns: u64 = 0,
    trim_ns: u64 = 0,
    capture_peak: u64 = 0,
    recovery_checks: u64 = 0,
    recovery_bytes: u64 = 0,
    calibration: [128]Calibration = undefined,
    calibration_n: usize = 0,

    pub fn init(gpa: std.mem.Allocator) !Ledger {
        var s: Ledger = .{ .gpa = gpa };
        errdefer s.deinit();
        try s.resident.ensureTotalCapacity(gpa, max_resident);
        try s.ghost.ensureTotalCapacity(gpa, max_ghost);
        return s;
    }
    pub fn deinit(s: *Ledger) void {
        s.resident.deinit(s.gpa);
        s.ghost.deinit(s.gpa);
    }
    pub fn tick(s: *Ledger) void {
        s.clock +|= 1;
    }
    pub fn use(s: *Ledger, key: Group) void {
        if (s.resident.getPtr(key)) |r| {
            const gap = @min(std.math.maxInt(u32), s.clock -| r.last);
            if (r.last != s.clock) {
                r.gap = @intCast((@as(u64, r.gap) * 3 + gap) / 4);
                r.returns +|= 1;
                r.frequency = r.aged(s.clock) +| 1;
                r.epoch = s.clock / 4096;
                r.last = s.clock;
            }
        }
    }
    pub fn hit(s: *Ledger) void {
        s.hits +|= 1;
    }
    fn ghostRoom(s: *Ledger) void {
        if (s.ghost.count() < max_ghost) return;
        var victim: ?Group = null;
        var oldest: u64 = std.math.maxInt(u64);
        var it = s.ghost.iterator();
        while (it.next()) |e| if (e.value_ptr.last < oldest or (e.value_ptr.last == oldest and (victim == null or e.key_ptr.order(victim.?)))) {
            victim = e.key_ptr.*;
            oldest = e.value_ptr.last;
        };
        if (victim) |key| _ = s.ghost.remove(key);
    }
    pub fn miss(s: *Ledger, key: Group) void {
        s.misses +|= 1;
        if (s.resident.contains(key)) return;
        if (s.ghost.getPtr(key)) |r| {
            if (r.last != s.clock) {
                const gap = @min(std.math.maxInt(u32), s.clock -| r.last);
                r.gap = @intCast((@as(u64, r.gap) * 3 + gap) / 4);
                r.returns +|= 1;
                r.frequency = r.aged(s.clock) +| 1;
                r.epoch = s.clock / 4096;
                r.last = s.clock;
            }
        } else {
            s.ghostRoom();
            s.ghost.putAssumeCapacity(key, .{ .last = s.clock, .epoch = s.clock / 4096 });
        }
    }
    pub fn expected(s: *const Ledger, key: Group) u64 {
        const r = s.resident.get(key) orelse s.ghost.get(key) orelse return 1;
        return @min(64, @max(1, r.probability(s.clock) * 512 / 65536 / @max(1, r.gap)));
    }
    pub fn worthCapture(s: *const Ledger, key: Leaf, control_ns: u64) bool {
        // Only matched GPU calibration may reject on economics. Missing data
        // is not a fabricated profit estimate; ample-room capture stays on.
        for (s.calibration[0..s.calibration_n]) |c| if (std.meta.eql(c.key, key)) {
            const r = s.resident.get(key.group) orelse s.ghost.get(key.group) orelse Record{};
            return s.expected(key.group) * (c.eager_ns - c.replay_ns) > r.rebuild_ns +| control_ns;
        };
        return true;
    }
    pub fn peak(s: *const Ledger, c: Config) u64 {
        return @max(c.unknown_peak, @min(256 * mib, s.capture_peak +| s.capture_peak / 2));
    }
    pub fn candidate(s: *const Ledger, current: Group, selected: []const Group, emergency: bool) ?Candidate {
        var best: ?Candidate = null;
        var it = s.resident.iterator();
        outer: while (it.next()) |en| {
            const key = en.key_ptr.*;
            const r = en.value_ptr.*;
            if (std.meta.eql(key, current) or (!emergency and r.protected_until > s.clock)) continue;
            for (selected) |k| if (std.meta.eql(k, key)) continue :outer;
            const v = r.value(s.clock);
            if (best == null or v < best.?.value or (v == best.?.value and (r.last < s.resident.get(best.?.key).?.last or
                (r.last == s.resident.get(best.?.key).?.last and key.order(best.?.key))))) best = .{ .key = key, .value = v, .bytes = r.bytes };
        }
        return best;
    }
    pub fn plan(s: *const Ledger, current: Group, deficit: u64, force_slot: bool, emergency: bool) Plan {
        var p: Plan = .{};
        while (p.n < max_victims and (p.estimated_bytes < deficit or (force_slot and p.n == 0))) {
            const v = s.candidate(current, p.keys[0..p.n], emergency) orelse break;
            p.keys[p.n] = v.key;
            p.n += 1;
            p.estimated_bytes +|= v.bytes;
        }
        return p;
    }
    pub fn retire(s: *Ledger, key: Group) void {
        const kv = s.resident.fetchRemove(key) orelse return;
        s.ghostRoom();
        var r = kv.value;
        r.evicted_at = s.clock;
        r.leaves = 0;
        s.ghost.putAssumeCapacity(key, r);
        s.evictions +|= 1;
    }
    pub fn captured(s: *Ledger, key: Leaf, cost_ns: u64, observed_bytes: u64) void {
        if (!s.resident.contains(key.group)) {
            std.debug.assert(s.resident.count() < max_resident);
            var r = if (s.ghost.fetchRemove(key.group)) |g| blk: {
                var old = g.value;
                if (old.captures > 0) s.ghost_returns +|= 1;
                if (old.evicted_at != 0 and old.evicted_at <= s.clock and s.clock - old.evicted_at <= 128) {
                    s.quick_returns +|= 1;
                    old.recaptures +|= 1;
                }
                old.leaves = 0;
                old.bytes = 0;
                old.rebuild_ns = 0;
                break :blk old;
            } else Record{ .bytes = 0, .rebuild_ns = 0 };
            r.last = s.clock;
            r.epoch = s.clock / 4096;
            r.protected_until = s.clock +| @min(128, 8 + @as(u64, r.recaptures) * 8);
            s.resident.putAssumeCapacity(key.group, r);
        }
        const r = s.resident.getPtr(key.group).?;
        std.debug.assert(r.leaves & key.mask() == 0);
        r.leaves |= key.mask();
        r.captures +|= 1;
        // Host reconstruction includes capture+instantiate+upload issue only.
        const sample = @min(cost_ns, 60 * std.time.ns_per_s);
        const clipped = if (r.build_samples == 0) sample else std.math.clamp(sample, r.build_per_leaf_ns / 4, r.build_per_leaf_ns * 4);
        r.build_per_leaf_ns = if (r.build_samples == 0) clipped else (r.build_per_leaf_ns * 3 + clipped) / 4;
        r.build_samples +|= 1;
        r.rebuild_ns +|= clipped;
        const bounded = @min(128 * mib, observed_bytes);
        r.bytes +|= @max(mib, if (bounded > 0) bounded else 16 * mib);
        if (bounded > 0) r.byte_samples +|= 1;
        s.capture_peak = @max(s.capture_peak, bounded);
        s.capture_ns +|= cost_ns;
        s.captures +|= 1;
    }
    pub fn inventory(s: *const Ledger) u64 {
        var h: u64 = 0;
        var it = s.resident.iterator();
        while (it.next()) |e| h ^= e.key_ptr.hash() *% 0x9e3779b97f4a7c15 ^ std.math.rotl(u64, e.value_ptr.leaves, 17);
        return h;
    }
};

test "replays allocate nothing and one primitive is counted once across stretches" {
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{});
    var s = try Ledger.init(failing.allocator());
    defer s.deinit();
    const k: Group = .{ .kind = .round, .rows = 6, .bucket = 1024 };
    s.tick();
    s.captured(.{ .group = k }, 1000, mib);
    failing.fail_index = failing.alloc_index;
    for (0..100) |_| {
        s.tick();
        s.use(k);
        s.use(k);
        s.use(k);
        s.hit();
    }
    try std.testing.expectEqual(@as(u32, 100), s.resident.get(k).?.returns);
    try std.testing.expectEqual(@as(u64, 101), s.clock);
}
test "shared body and distinct heads retire as one owner with bounded ghost history" {
    var s = try Ledger.init(std.testing.allocator);
    defer s.deinit();
    for (0..2000) |i| {
        const k: Group = .{ .kind = .round, .rows = 6, .bucket = i + 1 };
        s.tick();
        s.captured(.{ .group = k, .piece = 0 }, 1000, mib);
        s.captured(.{ .group = k, .piece = 2 }, 1000, mib);
        s.captured(.{ .group = k, .piece = 2, .greedy = true }, 1000, mib);
        try std.testing.expectEqual(@as(u64, 49), s.resident.get(k).?.leaves);
        s.retire(k);
        try std.testing.expect(s.ghost.count() <= max_ghost);
    }
    try std.testing.expectEqual(@as(usize, 0), s.resident.count());
    try std.testing.expectEqual(@as(usize, max_ghost), s.ghost.count());
}
test "quick ghost return earns protection and old popularity ages out" {
    var s = try Ledger.init(std.testing.allocator);
    defer s.deinit();
    const a: Group = .{ .kind = .round, .rows = 1, .bucket = 100 };
    const b: Group = .{ .kind = .draft, .rows = 1, .steps = 1 };
    s.tick();
    s.captured(.{ .group = a }, 1000, mib);
    s.retire(a);
    s.tick();
    s.captured(.{ .group = a }, 1000, mib);
    s.captured(.{ .group = b }, 1000, mib);
    try std.testing.expectEqual(@as(u64, 1), s.quick_returns);
    try std.testing.expectEqual(@as(u8, 0), s.plan(.{}, mib, false, false).n);
    s.clock += 100;
    for (0..100) |_| {
        s.tick();
        s.use(a);
    }
    try std.testing.expect(std.meta.eql(b, s.candidate(.{}, &.{}, false).?.key));
    s.clock += 32 * 4096;
    s.use(b);
    try std.testing.expect(std.meta.eql(a, s.candidate(.{}, &.{}, false).?.key));
}
test "compatible value compares round and draft and retires only deficit" {
    var s = try Ledger.init(std.testing.allocator);
    defer s.deinit();
    const a: Group = .{ .kind = .round, .rows = 1, .bucket = 100 };
    const b: Group = .{ .kind = .draft, .rows = 1, .steps = 1 };
    s.tick();
    s.captured(.{ .group = a }, 1000000, 16 * mib);
    s.captured(.{ .group = b }, 1000, 16 * mib);
    s.clock += 100;
    const p = s.plan(a, mib, false, false);
    try std.testing.expectEqual(@as(u8, 1), p.n);
    try std.testing.expect(std.meta.eql(b, p.keys[0]));
    try std.testing.expectEqual(@as(u8, 0), s.plan(b, mib, false, false).n -| 1);
}
test "capture admission does not invent GPU savings and calibration is leaf exact" {
    var s = try Ledger.init(std.testing.allocator);
    defer s.deinit();
    const k: Leaf = .{ .group = .{ .kind = .round, .rows = 1, .bucket = 100 } };
    try std.testing.expect(s.worthCapture(k, 100));
    s.calibration[0] = .{ .key = k, .eager_ns = 2000, .replay_ns = 1000, .samples = 8 };
    s.calibration_n = 1;
    try std.testing.expect(!s.worthCapture(k, 100));
    var other = k;
    other.greedy = true;
    try std.testing.expect(s.worthCapture(other, 100));
}
test "config and metadata bounds are explicit and include topology" {
    const c: Config = .{};
    try std.testing.expect(c.valid());
    try std.testing.expect(c.digest(1, 3) != c.digest(1, 4));
    var d = c;
    d.enabled = true;
    try std.testing.expect(c.digest(1, 4) != d.digest(1, 4));
    // Hash maps use < 2*capacity buckets; upper bound includes keys, records,
    // control/calibration buffers and conservative hashmap bookkeeping.
    try std.testing.expect((max_resident + max_ghost) * 2 * (@sizeOf(Group) + @sizeOf(Record) + 16) + @sizeOf(Ledger) < 2 * 1024 * 1024);
}
test "repeated eager misses learn a changed workload instead of permanent capture denial" {
    var s = try Ledger.init(std.testing.allocator);
    defer s.deinit();
    const k: Group = .{ .kind = .round, .rows = 6, .bucket = 1024 };
    for (0..30) |_| {
        s.tick();
        s.miss(k);
    }
    try std.testing.expect(s.expected(k) > 1);
    try std.testing.expectEqual(@as(usize, 0), s.resident.count());
    s.captured(.{ .group = k }, 1000000, mib);
    try std.testing.expectEqual(@as(u64, 0), s.ghost_returns);
    try std.testing.expectEqual(@as(u64, 0), s.quick_returns);
}
test "capture reconstruction smoothing clips a single load spike" {
    var s = try Ledger.init(std.testing.allocator);
    defer s.deinit();
    const k: Leaf = .{ .group = .{ .kind = .round, .rows = 6, .bucket = 1024 } };
    s.tick();
    s.captured(k, 1000000, mib);
    s.retire(k.group);
    s.tick();
    s.captured(k, 1000000000, mib);
    try std.testing.expect(s.resident.get(k.group).?.build_per_leaf_ns < 2000000);
}
