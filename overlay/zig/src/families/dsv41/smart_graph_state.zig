// Modified for the DeepSeek 1017 TP3 integration; upstream copyrights retained. See CREDITS.md and provenance.json.
//! Bounded, versioned statistics only. Never stores GPU handles or pointers.
const std = @import("std");
const p = @import("smart_graph_policy.zig");
pub const capacity = 128 * 1024;
const magic: u64 = 0x5347535441544533;
fn checksum(bytes: []const u8) u32 {
    var crc = std.hash.Crc32.init();
    crc.update(bytes[0..32]);
    crc.update(bytes[36..]);
    return crc.final();
}
pub const Codec = struct {
    bytes: []u8,
    at: usize = 0,
    pub fn put(c: *Codec, comptime T: type, value: T) !void {
        if (c.at + @sizeOf(T) > c.bytes.len) return error.StateTooLarge;
        std.mem.writeInt(T, c.bytes[c.at..][0..@sizeOf(T)], value, .little);
        c.at += @sizeOf(T);
    }
    pub fn get(c: *Codec, comptime T: type) !T {
        if (c.at + @sizeOf(T) > c.bytes.len) return error.BadState;
        const v = std.mem.readInt(T, c.bytes[c.at..][0..@sizeOf(T)], .little);
        c.at += @sizeOf(T);
        return v;
    }
    pub fn key(c: *Codec, k: p.Group) !void {
        try c.put(u8, @backingInt(k.kind));
        try c.put(u32, k.rows);
        try c.put(u64, k.bucket);
        try c.put(u32, k.steps);
        try c.put(u8, @intFromBool(k.split));
    }
    pub fn readKey(c: *Codec) !p.Group {
        const kind = std.enums.fromInt(p.Kind, try c.get(u8)) orelse return error.BadState;
        const rows = try c.get(u32);
        const bucket = try c.get(u64);
        const steps = try c.get(u32);
        const split = try c.get(u8);
        const k: p.Group = .{ .kind = kind, .rows = rows, .bucket = bucket, .steps = steps, .split = split == 1 };
        if (split > 1 or !k.valid() or kind == .none) return error.BadState;
        return k;
    }
    fn record(c: *Codec, r: p.Record) !void {
        inline for (.{ r.last, r.epoch, @as(u64, r.frequency), r.gap, r.returns, r.captures, r.recaptures, r.rebuild_ns, r.bytes, r.byte_samples, r.build_per_leaf_ns, r.build_samples }) |v| try c.put(u64, v);
    }
    fn readRecord(c: *Codec, clock: u64) !p.Record {
        const last = try c.get(u64);
        const epoch = try c.get(u64);
        var n: [7]u32 = undefined;
        n[0] = std.math.cast(u32, try c.get(u64)) orelse return error.BadState;
        n[1] = std.math.cast(u32, try c.get(u64)) orelse return error.BadState;
        n[2] = std.math.cast(u32, try c.get(u64)) orelse return error.BadState;
        n[3] = std.math.cast(u32, try c.get(u64)) orelse return error.BadState;
        n[4] = std.math.cast(u32, try c.get(u64)) orelse return error.BadState;
        const cost = try c.get(u64);
        const bytes = try c.get(u64);
        n[5] = std.math.cast(u32, try c.get(u64)) orelse return error.BadState;
        const build = try c.get(u64);
        n[6] = std.math.cast(u32, try c.get(u64)) orelse return error.BadState;
        if (last > clock or epoch > clock / 4096 or cost > 64 * 60 * std.time.ns_per_s or bytes > 64 * 128 * p.mib or build > 60 * std.time.ns_per_s) return error.BadState;
        return .{ .last = last, .epoch = epoch, .frequency = n[0], .gap = n[1], .returns = n[2], .captures = n[3], .recaptures = n[4], .rebuild_ns = cost, .bytes = bytes, .byte_samples = n[5], .evicted_at = 0, .build_per_leaf_ns = build, .build_samples = n[6] };
    }
};

pub fn save(s: *const p.Ledger, digest: u64, out: []u8) ![]u8 {
    // The best 512 histories survive, not an unbounded dump of the process.
    var keys: [p.max_ghost]p.Group = undefined;
    var records: [p.max_ghost]p.Record = undefined;
    var count: usize = 0;
    inline for (.{ &s.resident, &s.ghost }) |map| {
        var it = map.iterator();
        while (it.next()) |e| {
            if (count < keys.len) {
                keys[count] = e.key_ptr.*;
                records[count] = e.value_ptr.*;
                count += 1;
            } else {
                var worst: usize = 0;
                for (records[1..], 1..) |r, i| if (r.value(s.clock) < records[worst].value(s.clock)) {
                    worst = i;
                };
                if (e.value_ptr.value(s.clock) > records[worst].value(s.clock)) {
                    keys[worst] = e.key_ptr.*;
                    records[worst] = e.value_ptr.*;
                }
            }
        }
    }
    var c: Codec = .{ .bytes = out };
    try c.put(u64, magic);
    try c.put(u32, p.schema);
    try c.put(u64, digest);
    try c.put(u64, s.clock);
    try c.put(u32, @intCast(count));
    try c.put(u32, 0);
    for (keys[0..count], records[0..count]) |k, r| {
        try c.key(k);
        try c.record(r);
    }
    std.mem.writeInt(u32, out[32..36], checksum(out[0..c.at]), .little);
    return out[0..c.at];
}
pub fn restore(s: *p.Ledger, digest: u64, bytes: []u8) !void {
    if (bytes.len > capacity or s.resident.count() != 0 or s.ghost.count() != 0) return error.BadState;
    var c: Codec = .{ .bytes = bytes };
    if (try c.get(u64) != magic or try c.get(u32) != p.schema or try c.get(u64) != digest) return error.StateMismatch;
    const clock = try c.get(u64);
    const n = try c.get(u32);
    const crc = try c.get(u32);
    if (n > p.max_ghost or checksum(bytes) != crc) return error.BadState;
    var keys: [p.max_ghost]p.Group = undefined;
    var records: [p.max_ghost]p.Record = undefined;
    for (keys[0..n], records[0..n], 0..) |*k, *r, i| {
        k.* = try c.readKey();
        r.* = try c.readRecord(clock);
        for (keys[0..i]) |old| if (std.meta.eql(k.*, old)) return error.BadState;
    }
    if (c.at != bytes.len) return error.BadState;
    // Full validation precedes every mutation. Restored entries are ghosts:
    // no executable is resident in a new process and no warm GPU is claimed.
    s.clock = clock;
    for (keys[0..n], records[0..n]) |k, r| s.ghost.putAssumeCapacity(k, r);
}
pub fn loadCalibration(s: *p.Ledger, digest: u64, bytes: []u8) !void {
    var c: Codec = .{ .bytes = bytes };
    if (try c.get(u64) != 0x534743414c494233 or try c.get(u64) != digest) return error.StateMismatch;
    const n = try c.get(u32);
    const crc = try c.get(u32);
    if (n > s.calibration.len or std.hash.Crc32.hash(bytes[c.at..]) != crc) return error.BadState;
    var records: [128]p.Calibration = undefined;
    for (records[0..n], 0..) |*r, i| {
        const group = try c.readKey();
        const piece = try c.get(u32);
        const greedy = try c.get(u8);
        r.* = .{ .key = .{ .group = group, .piece = piece, .greedy = greedy == 1 }, .eager_ns = try c.get(u64), .replay_ns = try c.get(u64), .samples = try c.get(u32) };
        if (greedy > 1 or !r.valid()) return error.BadState;
        for (records[0..i]) |old| if (std.meta.eql(r.key, old.key)) return error.BadState;
    }
    if (c.at != bytes.len) return error.BadState;
    @memcpy(s.calibration[0..n], records[0..n]);
    s.calibration_n = n;
}

test "statistics restore as ghosts and wrong build topology or corrupt file is atomic" {
    var a = try p.Ledger.init(std.testing.allocator);
    defer a.deinit();
    var b = try p.Ledger.init(std.testing.allocator);
    defer b.deinit();
    a.tick();
    a.captured(.{ .group = .{ .kind = .round, .rows = 6, .bucket = 8192 } }, 2000, p.mib);
    var buf: [capacity]u8 = undefined;
    const bytes = try save(&a, 123, &buf);
    try std.testing.expectError(error.StateMismatch, restore(&b, 124, bytes));
    try std.testing.expectEqual(@as(usize, 0), b.ghost.count());
    try restore(&b, 123, bytes);
    try std.testing.expectEqual(@as(usize, 0), b.resident.count());
    try std.testing.expectEqual(@as(usize, 1), b.ghost.count());
    b.ghost.clearRetainingCapacity();
    bytes[bytes.len - 1] ^= 1;
    try std.testing.expectError(error.BadState, restore(&b, 123, bytes));
    try std.testing.expectEqual(@as(usize, 0), b.ghost.count());
    bytes[bytes.len - 1] ^= 1;
    bytes[20] ^= 1; // header clock is checksummed too
    try std.testing.expectError(error.BadState, restore(&b, 123, bytes));
}
test "maximum history has a fixed file size and contains no resident handles" {
    var s = try p.Ledger.init(std.testing.allocator);
    defer s.deinit();
    for (0..p.max_resident) |i| {
        s.tick();
        s.captured(.{ .group = .{ .kind = .round, .rows = 1, .bucket = i + 1 } }, 1000, p.mib);
    }
    var buf: [capacity]u8 = undefined;
    const bytes = try save(&s, 55, &buf);
    try std.testing.expect(bytes.len < capacity);
    var b = try p.Ledger.init(std.testing.allocator);
    defer b.deinit();
    try restore(&b, 55, bytes);
    try std.testing.expectEqual(@as(usize, p.max_ghost), b.ghost.count());
}
