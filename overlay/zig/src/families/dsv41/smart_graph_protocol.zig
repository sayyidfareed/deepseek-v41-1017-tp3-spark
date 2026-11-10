// Modified for the DeepSeek 1017 TP3 integration; upstream copyrights retained. See CREDITS.md and provenance.json.
//! Strict, allocation-free graph-control wire schema. Not a CUDA/NCCL stream.
const std = @import("std");
const p = @import("smart_graph_policy.zig");
pub const capacity = 512;
pub const Phase = enum(u8) { boot = 1, probe = 2, offer = 3, retired = 4, captured = 5, retire_ready = 6 };
pub const Action = enum(u8) { keep = 0, capture = 1, eager = 2, retire = 3, abort = 4 };
pub const Frame = struct {
    phase: Phase,
    action: Action = .keep,
    rank: u8 = 0,
    generation: u64 = 0,
    digest: u64 = 0,
    serial: u64 = 0,
    inventory: u64 = 0,
    leaf: p.Leaf = .{},
    available: u64 = 0,
    rss: u64 = 0,
    cost_ns: u64 = 0,
    bytes: u64 = 0,
    ok: bool = true,
    trim: bool = false,
    plan: p.Plan = .{},

    pub fn encode(f: Frame, out: *[capacity]u8) ![]const u8 {
        var w: Writer = .{ .buf = out };
        try w.int(u64, 0x5446475241504833);
        try w.int(u32, p.schema);
        try w.int(u8, @backingInt(f.phase));
        try w.int(u8, @backingInt(f.action));
        try w.int(u8, f.rank);
        try w.int(u64, f.generation);
        try w.int(u64, f.digest);
        try w.int(u64, f.serial);
        try w.int(u64, f.inventory);
        try w.group(f.leaf.group);
        try w.int(u32, f.leaf.piece);
        try w.int(u8, @intFromBool(f.leaf.greedy));
        inline for (.{ f.available, f.rss, f.cost_ns, f.bytes }) |v| try w.int(u64, v);
        try w.int(u8, @intFromBool(f.ok));
        try w.int(u8, @intFromBool(f.trim));
        try w.int(u8, f.plan.n);
        if (f.plan.n > p.max_victims) return error.BadGraphPlan;
        for (f.plan.keys[0..f.plan.n]) |k| try w.group(k);
        return out[0..w.at];
    }
    pub fn decode(bytes: []const u8) !Frame {
        if (bytes.len > capacity) return error.BadGraphFrame;
        var r: Reader = .{ .buf = bytes };
        if (try r.int(u64) != 0x5446475241504833 or try r.int(u32) != p.schema) return error.BadGraphSchema;
        const phase = std.enums.fromInt(Phase, try r.int(u8)) orelse return error.BadGraphPhase;
        const action = std.enums.fromInt(Action, try r.int(u8)) orelse return error.BadGraphAction;
        var f: Frame = .{ .phase = phase, .action = action, .rank = try r.int(u8) };
        f.generation = try r.int(u64);
        f.digest = try r.int(u64);
        f.serial = try r.int(u64);
        f.inventory = try r.int(u64);
        f.leaf = .{ .group = try r.group(), .piece = try r.int(u32), .greedy = try r.boolean() };
        f.available = try r.int(u64);
        f.rss = try r.int(u64);
        f.cost_ns = try r.int(u64);
        f.bytes = try r.int(u64);
        f.ok = try r.boolean();
        f.trim = try r.boolean();
        f.plan.n = try r.int(u8);
        if (f.plan.n > p.max_victims or !f.leaf.valid() or f.rank >= 4) return error.BadGraphFrame;
        for (f.plan.keys[0..f.plan.n], 0..) |*key, i| {
            key.* = try r.group();
            if (key.kind == .none) return error.BadGraphPlan;
            for (f.plan.keys[0..i]) |old| if (std.meta.eql(old, key.*)) return error.BadGraphPlan;
        }
        if (r.at != bytes.len or (f.action == .retire) != (f.plan.n > 0) or (f.trim and f.plan.n == 0)) return error.BadGraphFrame;
        if (f.action == .retire and f.phase != .offer and f.phase != .probe) return error.BadGraphFrame;
        if ((f.action == .capture or f.action == .eager) and f.phase != .offer) return error.BadGraphFrame;
        return f;
    }
};
pub fn validate(reference: Frame, peer: Frame, expected_rank: u8) !void {
    if (peer.rank != expected_rank or peer.phase != reference.phase or peer.serial != reference.serial or
        peer.generation != reference.generation or peer.digest != reference.digest or peer.inventory != reference.inventory or
        !std.meta.eql(peer.leaf, reference.leaf)) return error.GraphRankDivergence;
}
pub fn aggregate(a: *Frame, b: Frame) void {
    a.available = @min(a.available, b.available);
    a.rss = @max(a.rss, b.rss);
    a.cost_ns = @max(a.cost_ns, b.cost_ns);
    a.bytes = @max(a.bytes, b.bytes);
    a.ok = a.ok and b.ok;
}
const Writer = struct {
    buf: []u8,
    at: usize = 0,
    fn int(w: *Writer, comptime T: type, v: T) !void {
        if (w.at + @sizeOf(T) > w.buf.len) return error.BadGraphFrame;
        std.mem.writeInt(T, w.buf[w.at..][0..@sizeOf(T)], v, .little);
        w.at += @sizeOf(T);
    }
    fn group(w: *Writer, k: p.Group) !void {
        try w.int(u8, @backingInt(k.kind));
        try w.int(u32, k.rows);
        try w.int(u64, k.bucket);
        try w.int(u32, k.steps);
        try w.int(u8, @intFromBool(k.split));
    }
};
const Reader = struct {
    buf: []const u8,
    at: usize = 0,
    fn int(r: *Reader, comptime T: type) !T {
        if (r.at + @sizeOf(T) > r.buf.len) return error.ShortGraphFrame;
        const v = std.mem.readInt(T, r.buf[r.at..][0..@sizeOf(T)], .little);
        r.at += @sizeOf(T);
        return v;
    }
    fn boolean(r: *Reader) !bool {
        const v = try r.int(u8);
        if (v > 1) return error.BadGraphBoolean;
        return v == 1;
    }
    fn group(r: *Reader) !p.Group {
        const k: p.Group = .{ .kind = std.enums.fromInt(p.Kind, try r.int(u8)) orelse return error.BadGraphKey, .rows = try r.int(u32), .bucket = try r.int(u64), .steps = try r.int(u32), .split = try r.boolean() };
        if (!k.valid()) return error.BadGraphKey;
        return k;
    }
};
test "wire roundtrip rejects truncation trailing bytes and invalid schema before mutation" {
    var buf: [capacity]u8 = undefined;
    const f: Frame = .{ .phase = .offer, .digest = 17, .leaf = .{ .group = .{ .kind = .round, .rows = 48, .bucket = 8192 } } };
    const encoded = try f.encode(&buf);
    const length = encoded.len;
    const decoded = try Frame.decode(encoded);
    try validate(f, decoded, 0);
    for (0..length) |n| try std.testing.expectError(error.ShortGraphFrame, Frame.decode(buf[0..n]));
    buf[length] = 0;
    try std.testing.expectError(error.BadGraphFrame, Frame.decode(buf[0 .. length + 1]));
    buf[0] ^= 1;
    try std.testing.expectError(error.BadGraphSchema, Frame.decode(buf[0..length]));
}
test "all four ranks must agree on configuration generation exact key and ownership" {
    const head: Frame = .{ .phase = .offer, .digest = 42, .generation = 7, .serial = 19, .inventory = 123 };
    for (1..4) |i| {
        var peer = head;
        peer.rank = @intCast(i);
        try validate(head, peer, @intCast(i));
    }
    var bad = head;
    bad.rank = 1;
    bad.digest += 1;
    try std.testing.expectError(error.GraphRankDivergence, validate(head, bad, 1));
    bad = head;
    bad.rank = 1;
    bad.generation += 1;
    try std.testing.expectError(error.GraphRankDivergence, validate(head, bad, 1));
    bad = head;
    bad.rank = 1;
    bad.inventory += 1;
    try std.testing.expectError(error.GraphRankDivergence, validate(head, bad, 1));
    bad = head;
    bad.rank = 1;
    bad.leaf.piece = 1;
    try std.testing.expectError(error.GraphRankDivergence, validate(head, bad, 1));
}
test "tightest memory slowest reconstruction and any failed capture govern the whole cluster" {
    var all: Frame = .{ .phase = .captured, .available = 9 * 1024 * p.mib, .cost_ns = 100 };
    aggregate(&all, .{ .phase = .captured, .available = 500 * p.mib, .cost_ns = 900, .ok = false });
    try std.testing.expectEqual(@as(u64, 500 * p.mib), all.available);
    try std.testing.expectEqual(@as(u64, 900), all.cost_ns);
    try std.testing.expect(!all.ok);
}
test "duplicate victims and oversized work budget are never accepted" {
    var buf: [capacity]u8 = undefined;
    const k: p.Group = .{ .kind = .draft, .rows = 1, .steps = 1 };
    var f: Frame = .{ .phase = .probe, .action = .retire, .plan = .{ .n = 2 } };
    f.plan.keys[0] = k;
    f.plan.keys[1] = k;
    try std.testing.expectError(error.BadGraphPlan, Frame.decode(try f.encode(&buf)));
    f.plan.n = 9;
    try std.testing.expectError(error.BadGraphPlan, f.encode(&buf));
}
