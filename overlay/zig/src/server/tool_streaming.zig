// Modified for the DeepSeek 1017 TP3 integration; upstream copyrights retained. See CREDITS.md and provenance.json.
//! The streaming half of tool calls: each finished call's opening delta, then its arguments (``stream_tool_call_deltas``).
const std = @import("std");
const json = @import("json.zig");
const tool_specs = @import("tool_specs.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;

/// ``stream_tool_call_deltas``: each call's opening delta, then its arguments in one more.
pub fn deltas(a: Allocator, calls: []const Value) Allocator.Error![]Value {
    var out: std.ArrayList(Value) = .empty;
    for (calls, 0..) |call, index| {
        const function = call.get("function") orelse continue;
        if (function != .object) continue;
        const head_fn = try json.newObject(a);
        try head_fn.put(a, "name", .{ .string = if (function.get("name")) |n| (if (n.truthy()) try tool_specs.pyStr(a, n) else "") else "" });
        try head_fn.put(a, "arguments", .{ .string = "" });
        const head = try json.newObject(a);
        try head.put(a, "index", try json.intValue(a, index));
        const id = call.get("id");
        try head.put(a, "id", .{ .string = if (id != null and id.?.truthy()) try tool_specs.pyStr(a, id.?) else try std.fmt.allocPrint(a, "call_{d}", .{index}) });
        const kind = call.get("type");
        try head.put(a, "type", .{ .string = if (kind != null and kind.?.truthy()) try tool_specs.pyStr(a, kind.?) else "function" });
        try head.put(a, "function", .{ .object = head_fn });
        try out.append(a, try wrapCalls(a, .{ .object = head }));
        const args = function.get("arguments");
        const text = if (args != null and args.?.truthy()) try tool_specs.pyStr(a, args.?) else "";
        if (text.len > 0) {
            const arg_fn = try json.newObject(a);
            try arg_fn.put(a, "arguments", .{ .string = text });
            const more = try json.newObject(a);
            try more.put(a, "index", try json.intValue(a, index));
            try more.put(a, "function", .{ .object = arg_fn });
            try out.append(a, try wrapCalls(a, .{ .object = more }));
        }
    }
    return out.items;
}

/// ``{"tool_calls": [one]}``: one streamed call delta.
pub fn wrapCalls(a: Allocator, one: Value) Allocator.Error!Value {
    const list = try a.alloc(Value, 1);
    list[0] = one;
    const delta = try json.newObject(a);
    try delta.put(a, "tool_calls", .{ .array = list });
    return .{ .object = delta };
}

/// The variable that picks how a stream with tools sends its calls (serve.zig reads it; capabilities lists it).
pub const env = [_][]const u8{"TENSORFOLD_TOOL_STREAM"};

/// ``live``: DSML calls go out as they decode, a value past a closer inside it once the text says where its block ends.
/// ``hold``: from the first call's opener on, the content and the calls go out when the reply ends, as the whole reply's
/// parse gives them, so a cut or misformatted call never reaches the client.
pub const Mode = enum { live, hold };

/// TENSORFOLD_TOOL_STREAM: unset, empty or blank is ``live``; else ``live`` or ``hold`` exactly (surrounding whitespace
/// ignored), Invalid for anything else.
pub fn mode(value: ?[]const u8) error{Invalid}!Mode {
    const v = std.mem.trim(u8, value orelse "", " \t\r\n");
    if (v.len == 0) return .live;
    return std.meta.stringToEnum(Mode, v) orelse error.Invalid;
}

test "TENSORFOLD_TOOL_STREAM: live by default, live or hold, anything else refused" {
    try std.testing.expectEqual(Mode.live, try mode(null));
    try std.testing.expectEqual(Mode.live, try mode(""));
    try std.testing.expectEqual(Mode.live, try mode(" live\n"));
    try std.testing.expectEqual(Mode.hold, try mode("hold"));
    try std.testing.expectError(error.Invalid, mode("Hold"));
    try std.testing.expectError(error.Invalid, mode("1"));
    try std.testing.expectEqual(Mode.live, try mode(" \t\n"));
    try std.testing.expectEqual(Mode.hold, try mode(" hold "));
    try std.testing.expectError(error.Invalid, mode("HOLD"));
    try std.testing.expectError(error.Invalid, mode("live hold"));
}
