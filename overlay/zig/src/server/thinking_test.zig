// Modified for the DeepSeek 1017 TP3 integration; upstream copyrights retained. See CREDITS.md and provenance.json.
//! Synthetic reasoning boundaries through the actual OpenAI handler, both bytewise SSE and whole replies.
const std = @import("std");
const api = @import("engine_api");
const server = @import("root.zig");
const json = server.json;
const model_text = @import("model_text.zig");
const openai = @import("openai.zig");
const Conn = @import("http_conn.zig").Conn;
const Allocator = std.mem.Allocator;

const ByteText = struct {
    fn text(t: *@This()) model_text.Text {
        return .{ .ctx = t, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource } };
    }
    fn encode(_: *anyopaque, a: Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        const ids = try a.alloc(u32, input.len);
        for (input, ids) |byte, *id| id.* = byte;
        return ids;
    }
    fn decode(_: *anyopaque, a: Allocator, ids: []const u32) Allocator.Error![]u8 {
        const out = try a.alloc(u8, ids.len);
        for (ids, out) |id, *byte| byte.* = @intCast(id);
        if (std.unicode.utf8ValidateSlice(out)) return out;
        // Fixture replies are valid UTF-8; a bytewise event can only leave the final character incomplete.
        // Match the real decoder's U+FFFD tail so Incremental waits instead of counting broken bytes as characters.
        var end = out.len;
        while (end > 0 and !std.unicode.utf8ValidateSlice(out[0..end])) end -= 1;
        return std.mem.concat(a, u8, &.{ out[0..end], "\u{fffd}" });
    }
    fn tokenId(_: *anyopaque, _: []const u8) ?u32 { return null; }
    fn tokenString(_: *anyopaque, a: Allocator, id: u32) Allocator.Error![]u8 { return std.fmt.allocPrint(a, "{d}", .{id}); }
    fn vocabSize(_: *anyopaque) u32 { return 256; }
    fn eosIds(_: *anyopaque) []const u32 { return &.{}; }
    fn render(_: *anyopaque, a: Allocator, _: json.Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error![]u8 { return a.dupe(u8, "prompt"); }
    fn templateSource(_: *anyopaque) []const u8 { return ""; }
};

/// A synthetic engine emits one byte per token event, preserving every parser boundary for SSE checks.
const Capture = struct {
    raw: []const u8,
    fn engine(e: *@This()) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) api.Info { return .{ .context_window = 4096 }; }
    fn submit(ctx: *anyopaque, id: api.Id, _: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e: *Capture = @ptrCast(@alignCast(ctx));
        std.debug.assert(std.unicode.utf8ValidateSlice(e.raw));
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        for (e.raw) |byte| sink.event(sink.ctx, id, &.{ .tokens = &.{@as(u32, byte)} });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void { out.* = .{}; }
    fn memory(_: *anyopaque, _: bool) ?api.Memory { return null; }
};

const Seen = struct {
    status: ?u16 = null,
    opened: bool = false,
    done: bool = false,
    error_type: []const u8 = "",
    error_message: []const u8 = "",
    content: [4096]u8 = @splat(0),
    content_len: usize = 0,
    content_events: usize = 0,
    reasoning: [4096]u8 = @splat(0),
    reasoning_len: usize = 0,
    calls: usize = 0,
    arguments: [4096]u8 = @splat(0),
    arguments_len: usize = 0,
    preparing: i64 = -1,
    fn out(c: *Seen) openai.Out { return .{ .ctx = c, .vt = &.{ .open = open, .event = event, .reply = reply } }; }
    fn open(ctx: *anyopaque) error{Closed}!void { const c: *Seen = @ptrCast(@alignCast(ctx)); c.opened = true; }
    fn capture(c: *Seen, value: json.Value) void {
        if (value.get("content")) |v| if (v == .string) {
            @memcpy(c.content[c.content_len..][0..v.string.len], v.string);
            c.content_len += v.string.len;
            if (v.string.len > 0) c.content_events += 1;
        };
        if (value.get("reasoning_content")) |v| if (v == .string) {
            @memcpy(c.reasoning[c.reasoning_len..][0..v.string.len], v.string);
            c.reasoning_len += v.string.len;
        };
        if (value.get("tool_calls")) |list| for (list.array) |tool| {
            const f = tool.get("function") orelse continue;
            if (f.get("name") != null) c.calls += 1;
            if (f.get("arguments")) |args| if (args == .string) {
                @memcpy(c.arguments[c.arguments_len..][0..args.string.len], args.string);
                c.arguments_len += args.string.len;
            };
        };
    }
    fn errorBody(c: *Seen, p: json.Value) bool {
        const err = p.get("error") orelse return false;
        c.error_type = err.strField("type") orelse "";
        c.error_message = err.strField("message") orelse "";
        return true;
    }
    fn event(ctx: *anyopaque, payload: ?json.Value) error{Closed}!void {
        const c: *Seen = @ptrCast(@alignCast(ctx));
        const p = payload orelse { c.done = true; return; };
        if (c.errorBody(p)) return;
        const choices = p.get("choices") orelse return;
        if (choices.array.len == 0) return;
        c.capture(choices.array[0].get("delta") orelse return);
    }
    fn reply(ctx: *anyopaque, status: u16, p: json.Value) void {
        const c: *Seen = @ptrCast(@alignCast(ctx));
        c.status = status;
        if (c.errorBody(p)) return;
        const choices = p.get("choices") orelse return;
        if (choices.array.len > 0) c.capture(choices.array[0].get("message") orelse return);
    }
};

fn send(a: Allocator, engine: *Capture, stream: bool, tools: bool, thinking: bool) !Seen {
    var text: ByteText = .{};
    var srv = try server.Server.init(std.testing.allocator, std.testing.io, engine.engine(), text.text(), .{
        .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = true, .use_drafts = false,
    }, null);
    defer srv.deinit();
    const body = try std.fmt.allocPrint(a,
        "{{\"model\":\"m\",\"messages\":[{{\"role\":\"user\",\"content\":\"synthetic\"}}],\"max_tokens\":3000,\"stream\":{},\"chat_template_kwargs\":{{\"thinking\":{}}}{s}}}",
        .{ stream, thinking, if (tools) ",\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"type\":\"object\",\"properties\":{\"q\":{\"type\":\"string\"}}}}}]" else "" });
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    var seen: Seen = .{};
    openai.run(srv, a, seen.out(), .{ .conn = &conn }, true, (try json.parse(a, body)).ok);
    seen.error_message = try a.dupe(u8, seen.error_message);
    seen.error_type = try a.dupe(u8, seen.error_type);
    seen.preparing = srv.preparing.load(.acquire);
    return seen;
}

const D = "\u{ff5c}DSML\u{ff5c}";
const call_head = "<" ++ D ++ " calls>\n<" ++ D ++ " invoke name=\"lookup\">\n<" ++ D ++ " parameter name=\"q\" string=\"true\">";
const call_tail = "</" ++ D ++ " parameter>\n</" ++ D ++ " invoke>\n</" ++ D ++ " calls>";
const call = call_head ++ "7" ++ call_tail;

test "explicit reopened blocks are structured reasoning before prose or DSML in SSE and whole replies" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    for ([_]bool{ false, true }) |stream| for ([_]bool{ false, true }) |thinking| {
        var prose_engine: Capture = .{ .raw = "<think>First.</think><think>Second.</think>Answer." };
        const prose = try send(arena.allocator(), &prose_engine, stream, false, thinking);
        try std.testing.expectEqualStrings("Answer.", prose.content[0..prose.content_len]);
        try std.testing.expectEqualStrings("First.\n\nSecond.", prose.reasoning[0..prose.reasoning_len]);
        var tool_engine: Capture = .{ .raw = "<think>First.</think><think>Second.</think>" ++ call };
        const tool = try send(arena.allocator(), &tool_engine, stream, true, thinking);
        try std.testing.expectEqual(@as(usize, 0), tool.content_len);
        try std.testing.expectEqual(@as(usize, 1), tool.calls);
        try std.testing.expectEqualStrings("First.\n\nSecond.", tool.reasoning[0..tool.reasoning_len]);
    };
}

test "empty-thinking answers and tools preserve quoted markers, code, balanced literal tags and argument strings" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const answers = [_][]const u8{
        "Answer.", "", "Use `</think>` literally.", "The literal is \"</think>\".", "Use '</think>' literally.",
        "```text\n</think>\n```", "~~~text\n</think>\n~~~", "Example <think>literal</think>.", "I'll answer normally.",
        "~~~text\ninside~~~ is not a closing fence </think>\n~~~",
        "```text\ninside``` is not a closing fence </think>\n```",
    };
    for ([_]bool{ false, true }) |stream| {
        for (answers) |answer| {
            var engine: Capture = .{ .raw = try std.mem.concat(arena.allocator(), u8, &.{ "</think>", answer }) };
            const seen = try send(arena.allocator(), &engine, stream, false, true);
            try std.testing.expectEqualStrings(answer, seen.content[0..seen.content_len]);
            try std.testing.expectEqual(@as(usize, 0), seen.reasoning_len);
            try std.testing.expectEqualStrings("", seen.error_message);
            if (stream and std.mem.eql(u8, answer, "Answer.")) try std.testing.expect(seen.content_events > 1);
        }
        var direct: Capture = .{ .raw = "</think>" ++ call };
        const normal = try send(arena.allocator(), &direct, stream, true, true);
        try std.testing.expectEqual(@as(usize, 1), normal.calls);
        try std.testing.expectEqual(@as(usize, 0), normal.content_len);
        var args: Capture = .{ .raw = "</think>" ++ call_head ++ "literal </think> <think> <tool_call>" ++ call_tail };
        const arg_seen = try send(arena.allocator(), &args, stream, true, true);
        try std.testing.expectEqual(@as(usize, 1), arg_seen.calls);
        try std.testing.expectEqualStrings("{\"q\":\"literal </think> <think> <tool_call>\"}", arg_seen.arguments[0..arg_seen.arguments_len]);
        try std.testing.expectEqual(@as(usize, 0), arg_seen.content_len);
    }
}

test "quoted tool markup inside reasoning never becomes a call, and later literal answer markers stay literal" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const thought = "The example `" ++ call ++ "` is quoted, not a real call.";
    const answer = "A literal <think>example</think> is code-like text.";
    for ([_]bool{ false, true }) |stream| {
        var engine: Capture = .{ .raw = thought ++ "</think>" ++ answer };
        const seen = try send(arena.allocator(), &engine, stream, true, true);
        try std.testing.expectEqualStrings(thought, seen.reasoning[0..seen.reasoning_len]);
        try std.testing.expectEqualStrings(answer, seen.content[0..seen.content_len]);
        try std.testing.expectEqual(@as(usize, 0), seen.calls);
    }
}

test "empty-reasoning text keeps incremental SSE instead of waiting for the final answer" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var engine: Capture = .{ .raw = "</think>Incremental answer with several words." };
    const seen = try send(arena.allocator(), &engine, true, false, true);
    try std.testing.expectEqualStrings("Incremental answer with several words.", seen.content[0..seen.content_len]);
    try std.testing.expect(seen.content_events > 10);
    try std.testing.expect(seen.opened and seen.done and seen.status == null);
    try std.testing.expectEqualStrings("", seen.error_message);
    try std.testing.expectEqual(@as(i64, 0), seen.preparing);
}

test "the pinned real DeepSeek template distinguishes structured reasoning from contaminated content" {
    const template = @import("template");
    const source = @embedFile("testdata/deepseek1017-chat-template.jinja");
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try std.json.parseFromSlice(std.json.Value, a,
        "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"type\":\"object\",\"properties\":{\"q\":{\"type\":\"string\"}}}}}]", .{})).value;
    for ([_]bool{ false, true }) |thinking| for ([_]bool{ false, true }) |contaminated| {
        const messages = try std.fmt.allocPrint(a,
            "[{{\"role\":\"user\",\"content\":\"Read synthetic note\"}},{{\"role\":\"assistant\",\"content\":\"{s}\",\"reasoning_content\":{s},\"tool_calls\":[{{\"id\":\"call_synthetic\",\"type\":\"function\",\"function\":{{\"name\":\"lookup\",\"arguments\":{{\"q\":\"7\"}}}}}}]}},{{\"role\":\"tool\",\"tool_call_id\":\"call_synthetic\",\"content\":\"7\"}},{{\"role\":\"user\",\"content\":\"Continue briefly\"}}]",
            .{ if (contaminated) "SYNTHETIC_PLAN</think>" else "", if (contaminated) "null" else "\"SYNTHETIC_PLAN\"" });
        const values = (try std.json.parseFromSlice(std.json.Value, a, messages, .{})).value;
        const context_text = try std.fmt.allocPrint(a, "{{\"enable_thinking\":{},\"reasoning_effort\":100}}", .{thinking});
        const context = (try std.json.parseFromSlice(std.json.Value, a, context_text, .{})).value;
        const rendered = try template.renderChat(a, source, values, tools, context, true, null);
        const suffix = if (thinking) "<｜Assistant｜><think>" else "<｜Assistant｜></think>";
        try std.testing.expect(std.mem.endsWith(u8, rendered, suffix));
        const malformed = "<think></think>SYNTHETIC_PLAN</think>\n\n<" ++ D ++ " calls>";
        try std.testing.expectEqual(thinking and contaminated, std.mem.indexOf(u8, rendered, malformed) != null);
        if (thinking and !contaminated) try std.testing.expect(std.mem.indexOf(u8, rendered, "<think>SYNTHETIC_PLAN</think>\n\n<" ++ D ++ " calls>") != null);
    };
}
