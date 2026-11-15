// Modified for the DeepSeek 1017 TP3 integration; upstream copyrights retained. See CREDITS.md and provenance.json.
//! Temporary graph-policy learning for mixed private batches. Executable graphs
//! contain reusable launch geometry, not token/KV contents. Preserve that actual
//! inventory while discarding private timings, counters and reuse history.
const std = @import("std");
const policy = @import("smart_graph_policy.zig");
const ZeroFree = @import("lanes").ZeroFree;

pub const Snapshot = struct {
    zero: ZeroFree,
    saved: policy.Ledger,

    pub fn begin(self: *Snapshot, ledger: *policy.Ledger) !void {
        self.zero = .{ .parent = ledger.gpa };
        var temporary = try policy.Ledger.init(self.zero.allocator());
        errdefer temporary.deinit();
        const resident = temporary.resident;
        const ghost = temporary.ghost;
        temporary = ledger.*;
        temporary.gpa = self.zero.allocator();
        temporary.resident = resident;
        temporary.ghost = ghost;
        var r = ledger.resident.iterator();
        while (r.next()) |entry| temporary.resident.putAssumeCapacity(entry.key_ptr.*, entry.value_ptr.*);
        var g = ledger.ghost.iterator();
        while (g.next()) |entry| temporary.ghost.putAssumeCapacity(entry.key_ptr.*, entry.value_ptr.*);
        self.saved = ledger.*;
        ledger.* = temporary;
    }

    pub fn end(self: *Snapshot, ledger: *policy.Ledger) void {
        // Physical evictions cannot be undone by restoring metadata. Retain
        // prior ordinary records for survivors/evictees, without private costs.
        var removed: [policy.max_resident]policy.Group = undefined;
        defer std.crypto.secureZero(u8, std.mem.asBytes(&removed));
        var n: usize = 0;
        var prior = self.saved.resident.iterator();
        while (prior.next()) |entry| if (!ledger.resident.contains(entry.key_ptr.*)) {
            removed[n] = entry.key_ptr.*;
            n += 1;
        };
        const evictions = self.saved.evictions;
        for (removed[0..n]) |key| self.saved.retire(key);
        self.saved.evictions = evictions;
        var current = ledger.resident.iterator();
        while (current.next()) |entry| {
            const key = entry.key_ptr.*;
            if (self.saved.resident.getPtr(key)) |record| {
                record.leaves = entry.value_ptr.leaves;
            } else {
                var record = if (self.saved.ghost.fetchRemove(key)) |old| old.value else policy.Record{};
                record.leaves = entry.value_ptr.leaves;
                self.saved.resident.putAssumeCapacity(key, record);
            }
        }
        self.saved.generation = ledger.generation; // All-rank structural identity.
        ledger.deinit(); // ZeroFree erases both maps, including removed cells.
        std.crypto.secureZero(u8, std.mem.asBytes(ledger));
        ledger.* = self.saved;
    }
};

test "private graph learning restores ordinary history and preserves actual executables" {
    var ledger = try policy.Ledger.init(std.testing.allocator);
    defer ledger.deinit();
    const a: policy.Leaf = .{ .group = .{ .kind = .round, .rows = 4, .bucket = 1024 } };
    const b: policy.Leaf = .{ .group = .{ .kind = .round, .rows = 8, .bucket = 2048 } };
    const c: policy.Leaf = .{ .group = .{ .kind = .round, .rows = 12, .bucket = 4096 } };
    ledger.captured(a, 2000, 10 * policy.mib);
    ledger.captured(b, 3000, 12 * policy.mib);
    ledger.tick(); ledger.use(a.group);
    const ordinary = ledger.resident.get(a.group).?;
    const old_clock = ledger.clock;
    const old_captures = ledger.captures;
    const old_cost = ledger.capture_ns;
    var scope: Snapshot = undefined;
    try scope.begin(&ledger);
    ledger.tick(); ledger.use(a.group); ledger.hit();
    ledger.retire(b.group);
    ledger.captured(c, 999999, 50 * policy.mib);
    ledger.generation += 1;
    scope.end(&ledger);
    try std.testing.expectEqualDeep(ordinary, ledger.resident.get(a.group).?);
    try std.testing.expect(!ledger.resident.contains(b.group));
    try std.testing.expect(ledger.ghost.contains(b.group));
    const newly_compiled = ledger.resident.get(c.group).?;
    try std.testing.expectEqual(c.mask(), newly_compiled.leaves);
    try std.testing.expectEqual(@as(u32, 0), newly_compiled.build_samples);
    try std.testing.expectEqual(@as(u32, 0), newly_compiled.captures);
    try std.testing.expectEqual(old_clock, ledger.clock);
    try std.testing.expectEqual(old_captures, ledger.captures);
    try std.testing.expectEqual(old_cost, ledger.capture_ns);
    try std.testing.expectEqual(@as(u64, 0), ledger.hits);
    try std.testing.expectEqual(@as(u64, 1), ledger.generation);
}
