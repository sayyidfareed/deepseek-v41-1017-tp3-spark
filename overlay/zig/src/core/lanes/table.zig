// Modified for the DeepSeek 1017 TP3 integration; upstream copyrights retained. See CREDITS.md and provenance.json.
//! Milliseconds by an integer key (rows or drafts): the int -> float dicts the depth rule reads in Python.
const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Table = struct {
    slots: []?f64 = &.{},

    pub fn deinit(t: *Table, gpa: Allocator) void {
        gpa.free(t.slots);
        t.* = .{};
    }

    pub fn get(t: Table, key: i64) ?f64 {
        if (key < 0 or key >= @as(i64, @intCast(t.slots.len))) return null;
        return t.slots[@intCast(key)];
    }

    /// Set `key` (growing the table): keys are small non-negative ints.
    pub fn put(t: *Table, gpa: Allocator, key: i64, value: f64) !void {
        std.debug.assert(key >= 0);
        const at: usize = @intCast(key);
        if (at >= t.slots.len) {
            const old = t.slots.len;
            t.slots = try gpa.realloc(t.slots, at + 1);
            @memset(t.slots[old..], null);
        }
        t.slots[at] = value;
    }

    /// Python's `not costs`: no key set.
    pub fn empty(t: Table) bool {
        for (t.slots) |slot| if (slot != null) return false;
        return true;
    }
};

test "table get and put" {
    const gpa = std.testing.allocator;
    var t: Table = .{};
    defer t.deinit(gpa);
    try std.testing.expect(t.empty());
    try t.put(gpa, 3, 1.5);
    try std.testing.expectEqual(@as(?f64, 1.5), t.get(3));
    try std.testing.expectEqual(@as(?f64, null), t.get(2));
    try std.testing.expectEqual(@as(?f64, null), t.get(-1));
}

/// A measured cost must be finite and positive; zero would make the draft allocation divide by zero.
pub fn validateMs(values: []const f64) !void {
    if (values.len == 0) return error.BadRoundMs;
    for (values) |v| if (!std.math.isFinite(v) or v <= 0) return error.BadRoundMs;
}

/// Parse the existing TF_DS_ROUND_MS format without retaining a partial allocation on refusal.
pub fn parseMs(gpa: Allocator, text: []const u8) ![]const f64 {
    var values: std.ArrayList(f64) = .empty;
    errdefer values.deinit(gpa);
    var it = std.mem.splitScalar(u8, text, ',');
    while (it.next()) |part| {
        const value = std.fmt.parseFloat(f64, std.mem.trim(u8, part, " ")) catch return error.BadRoundMs;
        try values.append(gpa, value);
    }
    try validateMs(values.items);
    return values.toOwnedSlice(gpa);
}

/// Drafter overhead may be zero in a controlled comparison, but never negative or non-finite.
pub fn parseDraftMs(text: []const u8) !f64 {
    const value = std.fmt.parseFloat(f64, text) catch return error.BadDraftMs;
    if (!std.math.isFinite(value) or value < 0) return error.BadDraftMs;
    return value;
}

test "measured costs refuse malformed, zero and nonfinite values without leaking" {
    const a = std.testing.allocator;
    for ([_][]const u8{ "", "1,", "0", "-1", "nan", "inf", "1,nan", "1,-2" }) |text| {
        try std.testing.expectError(error.BadRoundMs, parseMs(a, text));
    }
    const values = try parseMs(a, "1, 2.5,3");
    defer a.free(values);
    try std.testing.expectEqualSlices(f64, &.{ 1, 2.5, 3 }, values);
    try std.testing.expectError(error.BadDraftMs, parseDraftMs("nan"));
    try std.testing.expectError(error.BadDraftMs, parseDraftMs("-1"));
    try std.testing.expectEqual(@as(f64, 0), try parseDraftMs("0"));
}
