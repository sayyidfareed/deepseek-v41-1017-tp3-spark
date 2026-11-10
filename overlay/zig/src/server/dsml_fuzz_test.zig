// Modified for the DeepSeek 1017 TP3 integration; upstream copyrights retained. See CREDITS.md and provenance.json.
//! Random DeepSeek-V4.1 replies fed to tool_stream's DSML streamer at random split points, against
//! tool_parse.parse of the same text, the reply's end settled as the route settles it. Well-formed replies stream the
//! parse's calls byte for byte. In every reply (odd markup, closers inside values, replies cut anywhere) each call a
//! client can read whole (its arguments JSON) is one of the parse's calls, in order: no whole call goes out that the
//! parse does not return. A repeated
//! parameter goes out twice, so there the check reads the arguments as JSON with the last value kept. Well-formed replies
//! whose values hold the block's own closer (a file about DSML) and no other DSML tag stream the parse's calls byte for
//! byte as well: the stream reads past such a closer as the parse's longer reading does.
const std = @import("std");
const json = @import("json.zig");
const tool_parse = @import("tool_parse.zig");
const tool_stream = @import("tool_stream.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;

const D = "\u{ff5c}DSML\u{ff5c}";
const tools_text =
    \\[{"type":"function","function":{"name":"Bash","parameters":{"type":"object","properties":{"command":{"type":"string"},"timeout":{"type":"number"}}}}},
    \\{"type":"function","function":{"name":"Write","parameters":{"type":"object","properties":{"file_path":{"type":"string"},"content":{"type":"string"}}}}}]
;

const bits = [_][]const u8{ "ls -la", "a\"b", "back\\slash", "\n", "\r\n", "\t", "\x01", "\x7f", "\u{6d77}", "\u{1f30a}", "\u{1f469}\u{200d}\u{1f4bb}", "</" ++ D ++ " par", "</" ++ D ++ " parameter", "</" ++ D, "</", "<", "\"", "}", "{", "\\u0041", " ", "string=\"true\"", "<" ++ D ++ " parameter name=\"x\" string=\"true\">", "<" ++ D ++ " invoke name=\"Bash\">", "<" ++ D ++ " calls>", "</" ++ D ++ " invo", "</" ++ D ++ " cal", "\u{ff5c}", "<|channel|", "<tool_call" };
const tie_calls = [_][]const u8{
    "</" ++ D ++ " calls>\n<tool_call>{\"name\":\"Bash\",\"arguments\":{\"command\":\"ls\"}}</tool_call>\n",
    "</" ++ D ++ " calls> <tool_call>{\"name\":\"Write\",\"arguments\":{\"file_path\":\"/a\"}}</tool_call>",
};
const closers = [_][]const u8{ "</" ++ D ++ " invoke>", "</" ++ D ++ " calls>", "</" ++ D ++ " parameter>" };
const json_values = [_][]const u8{ "5", " 120000 ", "{\"a\":[1,2]}", "true", "null", "NaN", "1e5", "-0", "1.0", "\"str\"", "not json", "", "[1, 2 ,3]", "{\"k\": \"v\\n\"}", "\"\\u00e9\"", "12345678901234567890" };
const names = [_][]const u8{ "Bash", " bash ", "Write", "Unknown", "", "  ", "WRITE" };
const params = [_][]const u8{ "command", "timeout", "content", "file_path", "x", "\u{6d77}", "a\\b" };
const flags = [_][]const u8{ " string=\"true\"", " string=\"false\"", "", " string=\"maybe\"", " string=\"true\" x" };

/// ``plain``: of the DSML tags, values hold only the block's own closer (no opener, invoke or parameter tag, no other call
/// form's), as a file about DSML does; the parse's longer reading then wins outright.
pub const Mode = struct { closers: bool = false, malformed: bool = false, cut: bool = false, plain: bool = false };

const plain_bits = [_][]const u8{ "ls -la", "a\"b", "back\\slash", "\n", "\r\n", "\t", "\x01", "\x7f", "\u{6d77}", "\u{1f30a}", "\u{1f469}\u{200d}\u{1f4bb}", "</" ++ D ++ " par", "</" ++ D ++ " parameter", "</" ++ D, "</", "<", "\"", "}", "{", "\\u0041", " ", "string=\"true\"", "</" ++ D ++ " invo", "</" ++ D ++ " cal", "\u{ff5c}", "calls>", " calls" };

pub const Gen = struct {
    r: std.Random,
    a: Allocator,
    m: Mode,
    out: std.ArrayList(u8) = .empty,
    repeated: bool = false,
    closer_in_value: bool = false, // a value holds the block's own closer
    used: [params.len]bool = @splat(false),

    fn add(g: *Gen, s: []const u8) !void {
        try g.out.appendSlice(g.a, s);
    }
    fn chance(g: *Gen, p: f32) bool {
        return g.r.float(f32) < p;
    }
    fn pick(g: *Gen, comptime T: type, list: []const T, n: usize) T {
        return list[g.r.uintLessThan(usize, @min(n, list.len))];
    }
    fn param(g: *Gen) !void {
        try g.add("<" ++ D ++ " parameter name=\"");
        const i = g.r.uintLessThan(usize, params.len);
        if (g.used[i]) g.repeated = true;
        g.used[i] = true;
        try g.add(params[i]);
        if (!(g.m.malformed and g.chance(0.05))) try g.add("\"");
        const flag = g.pick([]const u8, &flags, if (g.m.malformed) flags.len else 3);
        try g.add(flag);
        try g.add(">");
        if (std.mem.eql(u8, flag, " string=\"false\"")) {
            try g.add(g.pick([]const u8, &json_values, json_values.len));
        } else for (0..g.r.uintLessThan(usize, 6)) |_| {
            if (g.m.plain) {
                if (g.chance(0.1)) {
                    try g.add("</" ++ D ++ " calls>");
                    g.closer_in_value = true;
                } else try g.add(g.pick([]const u8, &plain_bits, plain_bits.len));
            } else if (g.m.closers and g.chance(0.02)) {
                // the block's closer and then a whole call in another form: a reading cut there ties the longer one
                try g.add(g.pick([]const u8, &tie_calls, tie_calls.len));
            } else try g.add(if (g.m.closers and g.chance(0.04)) g.pick([]const u8, &closers, closers.len) else g.pick([]const u8, &bits, bits.len));
        }
        if (!(g.m.malformed and g.chance(0.08))) try g.add("</" ++ D ++ " parameter>");
        if (g.chance(0.8)) try g.add("\n");
    }
    fn invoke(g: *Gen) !void {
        g.used = @splat(false);
        try g.add("<" ++ D ++ " invoke name=\"");
        try g.add(g.pick([]const u8, &names, if (g.m.malformed) names.len else 4));
        try g.add(if (g.m.malformed and g.chance(0.05)) "\" x=\"1\">\n" else "\">\n");
        for (0..g.r.uintLessThan(usize, 4)) |_| try g.param();
        if (!(g.m.malformed and g.chance(0.08))) try g.add("</" ++ D ++ " invoke>");
        try g.add("\n");
    }
    fn block(g: *Gen) !void {
        try g.add(if (g.chance(0.8)) "\n\n<" ++ D ++ " calls>\n" else "<" ++ D ++ " calls>");
        for (0..1 + g.r.uintLessThan(usize, 3)) |_| try g.invoke();
        if (!(g.m.malformed and g.chance(0.1))) try g.add("</" ++ D ++ " calls>");
    }
    pub fn reply(g: *Gen) ![]const u8 {
        if (g.chance(0.5)) try g.add("Running it.");
        if (g.m.malformed and g.chance(0.05)) try g.add("Use `<" ++ D ++ " calls>` like:");
        try g.block();
        if (g.chance(0.15)) {
            try g.add("\nThen.");
            try g.block();
        }
        if (g.chance(0.1)) try g.add("\ntrailing");
        var text = g.out.items;
        if (g.m.cut and g.r.boolean()) {
            var cut = g.r.uintLessThan(usize, text.len + 1);
            while (cut < text.len and text[cut] & 0xc0 == 0x80) cut += 1;
            text = text[0..cut];
        }
        return text;
    }
};

/// What a client got of the calls: each call's name and its arguments joined from the deltas.
const Wire = struct { names: []const []const u8, args: []const []const u8 };

/// The text fed as it grows by 1 to 12 bytes (kept to whole characters), then whole, then settled as the route settles
/// a reply's end (what a block read past a closer inside a value withheld goes out only when the parse takes the calls
/// that went out); the deltas' shape checked: each header once and in order, pieces for the newest call (a held call's
/// brace for the one before it), valid UTF-8, joined to the streamer's record of what the client has.
fn feed(a: Allocator, tools: []const Value, text: []const u8, parsed: []const Value, r: std.Random) !Wire {
    var s = try tool_stream.Streamer.init(a, tools);
    var out: std.ArrayList(Value) = .empty;
    var end: usize = 0;
    while (end < text.len) {
        end = @min(text.len, end + 1 + r.uintLessThan(usize, 12));
        while (end < text.len and text[end] & 0xc0 == 0x80) end += 1;
        try s.feedDsml(text[0..end], &out);
    }
    try s.feedDsml(text, &out);
    const takes = tool_stream.parseTakes(s.dsml.streamed(), parsed);
    if (s.dsml.deferred and takes) try out.appendSlice(a, s.dsml.withheld.items);
    var called: std.ArrayList([]const u8) = .empty;
    var joined: std.ArrayList(std.ArrayList(u8)) = .empty;
    for (out.items) |d| for (d.get("tool_calls").?.array) |one| {
        const index: usize = @intCast(one.get("index").?.int64().?);
        const function = one.get("function").?;
        const piece = function.strField("arguments").?;
        try std.testing.expect(std.unicode.utf8ValidateSlice(piece));
        if (one.get("id") != null) {
            try std.testing.expectEqual(joined.items.len, index);
            try called.append(a, function.strField("name").?);
            try joined.append(a, .empty);
            continue;
        }
        try std.testing.expect(index + 1 == joined.items.len or index + 2 == joined.items.len);
        try joined.items[index].appendSlice(a, piece);
    };
    const client = if (takes) s.dsml.streamed() else try s.dsml.wire(a);
    try std.testing.expectEqual(client.len, joined.items.len);
    for (client, joined.items) |c, j| try std.testing.expectEqualStrings(c.arguments.items, j.items);
    var args: std.ArrayList([]const u8) = .empty;
    for (joined.items) |j| try args.append(a, j.items);
    // the route's settle: when the parse takes what went out, its later calls follow whole
    if (takes) for (parsed[client.len..]) |p| {
        try called.append(a, p.get("function").?.strField("name").?);
        try args.append(a, arguments(p));
    };
    return .{ .names = called.items, .args = args.items };
}

fn arguments(v: Value) []const u8 {
    return v.get("function").?.strField("arguments").?;
}

/// The same arguments for a client: the same bytes, or the same JSON with a repeated key's last value kept (both
/// holding NaN, which the parse writes bare and a JSON reader refuses, counts as the same).
pub fn sameArguments(a: Allocator, parsed: []const u8, sent: []const u8) Allocator.Error!bool {
    if (std.mem.eql(u8, parsed, sent)) return true;
    if (std.mem.indexOf(u8, parsed, "NaN") != null and std.mem.indexOf(u8, sent, "NaN") != null) return true;
    const x = try json.parseText(a, parsed);
    const y = try json.parseText(a, sent);
    return x == .ok and y == .ok and json.equal(x.ok, y.ok);
}

fn run(seed: u64, count: usize, m: Mode) !void {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    for (0..count) |_| {
        var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const tools = (try json.parse(a, tools_text)).ok.array;
        var g: Gen = .{ .r = r, .a = a, .m = m };
        const text = try g.reply();
        const parsed = (try tool_parse.parse(a, text, tools, null)).calls orelse &.{};
        const got = try feed(a, tools, text, parsed, r);
        // every call the client can read whole (its arguments JSON) is the parse's, in order
        var at: usize = 0;
        for (got.names, got.args) |n, args| {
            if (!try readable(a, args)) continue;
            while (at < parsed.len) : (at += 1) {
                const p = parsed[at];
                if (std.mem.eql(u8, p.get("function").?.strField("name").?, n) and try sameArguments(a, arguments(p), args)) break;
            } else {
                std.debug.print("a whole streamed call the parse does not return: {s} {s} in {f}\n", .{ n, args, std.zig.fmtString(text) });
                return error.WholeCallNotParsed;
            }
            at += 1;
        }
        if ((!m.closers or m.plain) and !m.malformed and !m.cut and !g.repeated) {
            errdefer std.debug.print("the stream is not the parse: {f}\n", .{std.zig.fmtString(text)});
            try std.testing.expectEqual(parsed.len, got.names.len);
            for (parsed, got.names, got.args) |p, n, args| {
                try std.testing.expectEqualStrings(p.get("function").?.strField("name").?, n);
                try std.testing.expectEqualStrings(arguments(p), args);
            }
        }
    }
}

/// A client can read ``args`` as a whole call's arguments: JSON, or JSON but for a bare NaN (which the parse writes).
fn readable(a: Allocator, args: []const u8) Allocator.Error!bool {
    if (try json.parseText(a, args) == .ok) return true;
    if (std.mem.indexOf(u8, args, "NaN") == null) return false;
    return try json.parseText(a, try std.mem.replaceOwned(u8, a, args, "NaN", "0")) == .ok;
}

test "random well-formed DSML replies stream the parse's calls byte for byte at random splits" {
    try run(1, 4000, .{});
}

test "random well-formed replies whose values hold the block's own closer stream the parse's calls byte for byte at random splits" {
    try run(5, 4000, .{ .closers = true, .plain = true });
    try run(6, 2000, .{ .closers = true, .plain = true });
}

test "random replies with odd markup, closers inside values, or a cut anywhere send no whole call the parse does not return" {
    try run(2, 3000, .{ .malformed = true });
    try run(3, 3000, .{ .closers = true });
    try run(4, 4000, .{ .closers = true, .malformed = true, .cut = true });
}

const reply_text = @import("reply_text.zig");

test "the stream filter, going on from the round before, hides what it hides reading each round's text whole" {
    var prng = std.Random.DefaultPrng.init(7);
    const r = prng.random();
    const modes = [_]Mode{ .{}, .{ .closers = true, .plain = true }, .{ .malformed = true }, .{ .closers = true }, .{ .closers = true, .malformed = true, .cut = true } };
    const others = [_][]const u8{ "", "<tool_call>{\"name\":\"Bash\"}</tool_call>", "<|tool_call>call:Bash{}<tool_call|>", "<" ++ D ++ "tool_calls>\n</" ++ D ++ "tool_calls>", "<tool_call>open", "prose " };
    for (0..2000) |i| {
        var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        var g: Gen = .{ .r = r, .a = a, .m = modes[i % modes.len] };
        const body = try g.reply();
        const text = try std.mem.concat(a, u8, &.{ others[r.uintLessThan(usize, others.len)], body, others[r.uintLessThan(usize, others.len)] });
        var h: reply_text.Hide = .{};
        var whole: std.ArrayList(u8) = .empty;
        var resumed: std.ArrayList(u8) = .empty;
        var end: usize = 0;
        while (end < text.len) {
            end = @min(text.len, end + 1 + r.uintLessThan(usize, 12));
            while (end < text.len and text[end] & 0xc0 == 0x80) end += 1;
            const want = try reply_text.hideInto(a, &whole, text[0..end], false);
            const got = try reply_text.hideIntoResume(a, &resumed, text[0..end], &h);
            std.testing.expectEqualStrings(want, got) catch |e| {
                std.debug.print("at {d} of {f}\n", .{ end, std.zig.fmtString(text) });
                return e;
            };
        }
    }
}
