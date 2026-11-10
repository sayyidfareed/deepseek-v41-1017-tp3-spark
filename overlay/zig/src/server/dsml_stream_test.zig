// Modified for the DeepSeek 1017 TP3 integration; upstream copyrights retained. See CREDITS.md and provenance.json.
//! DeepSeek-V4.1's DSML tool calls stream as they decode. Real token sequences (the DeepSeek-V4.1
//! tokenizer, testdata/dsml_stream.json) run through the chat route and the Messages route with a CPU fake engine. Each
//! stream's argument pieces join to the whole reply's arguments byte for byte, in order, sent once; a stream without
//! tools, or with the one-call policy, is the same as before the patch event for event (testdata/dsml_stream_golden.json,
//! recorded without the patch). A value that holds its block's own closer (a file about DSML) streams up to the closer,
//! then waits until the text says where the block ends, and the call completes to the whole reply's; where the whole
//! reply's parse rejects a call whose header went out, the stream ends with the reply's own finish reason.
const std = @import("std");
const builtin = @import("builtin");
const api = @import("engine_api");
const tokenizer = @import("tokenizer");
const server = @import("root.zig");
const json = server.json;
const model_text = @import("model_text.zig");
const openai = @import("openai.zig");
const anthropic = @import("anthropic.zig");
const tool_streaming = @import("tool_streaming.zig");
const Conn = @import("http_conn.zig").Conn;
const Allocator = std.mem.Allocator;

const fixture_text = @embedFile("testdata/dsml_stream.json");
const golden_text = @embedFile("testdata/dsml_stream_golden.json");

/// The fixture's tokenizer (only the cases' tokens, under their real ids) behind the server's text seam; the
/// rendered prompt is a placeholder the fake engine does not read.
const FixtureText = struct {
    tok: tokenizer.Tokenizer,
    eos: [1]u32,

    fn text(t: *FixtureText) model_text.Text {
        return .{ .ctx = t, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource } };
    }

    fn self(ctx: *anyopaque) *FixtureText {
        return @ptrCast(@alignCast(ctx));
    }

    fn encode(_: *anyopaque, a: Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        const ids = try a.alloc(u32, input.len);
        for (input, ids) |byte, *id| id.* = byte;
        return ids;
    }

    fn decode(ctx: *anyopaque, a: Allocator, ids: []const u32) Allocator.Error![]u8 {
        return self(ctx).tok.decode(a, ids, false) catch |e| if (e == error.OutOfMemory) error.OutOfMemory else a.dupe(u8, "\u{fffd}");
    }

    fn tokenId(ctx: *anyopaque, piece: []const u8) ?u32 {
        return self(ctx).tok.specialTokenId(piece);
    }

    fn tokenString(ctx: *anyopaque, a: Allocator, id: u32) Allocator.Error![]u8 {
        return a.dupe(u8, self(ctx).tok.id_to_token.get(id) orelse "");
    }

    fn vocabSize(_: *anyopaque) u32 {
        return 129280;
    }

    fn eosIds(ctx: *anyopaque) []const u32 {
        return &self(ctx).eos;
    }

    fn render(_: *anyopaque, a: Allocator, _: json.Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error![]u8 {
        return a.dupe(u8, "prompt");
    }

    fn templateSource(_: *anyopaque) []const u8 {
        return "";
    }
};

/// An engine that answers with a case's ids in rounds of `pattern` tokens (cycled), as a drafting engine commits them,
/// and ends as the ids end: with their end token, or at the length (a reply cut by max_tokens).
const ScriptEngine = struct {
    ids: []const u32,
    pattern: []const usize,
    cut: bool,

    fn engine(e: *ScriptEngine) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }

    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 1 << 20 };
    }

    fn submit(ctx: *anyopaque, id: api.Id, _: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e: *ScriptEngine = @ptrCast(@alignCast(ctx));
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        var pos: usize = 0;
        var round: usize = 0;
        while (pos < e.ids.len) : (round += 1) {
            const end = @min(e.ids.len, pos + e.pattern[round % e.pattern.len]);
            sink.event(sink.ctx, id, &.{ .tokens = e.ids[pos..end] });
            pos = end;
        }
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = if (e.cut) .length else .stop } });
    }

    fn cancel(_: *anyopaque, _: api.Id) void {}

    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }

    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

/// What the route sent: each stream event's JSON, or the whole reply.
const Seen = struct {
    a: Allocator,
    events: std.ArrayList([]const u8) = .empty,
    done: bool = false,
    whole: ?json.Value = null,
    pings: usize = 0,

    fn out(s: *Seen) openai.Out {
        return .{ .ctx = s, .vt = &.{ .open = open, .event = event, .reply = reply, .ping = ping } };
    }

    /// A keep-alive, kept among the events (as `{"ping":true}`, which readers of choices skip) to see what it came between.
    fn ping(ctx: *anyopaque) error{Closed}!void {
        const s: *Seen = @ptrCast(@alignCast(ctx));
        s.pings += 1;
        s.events.append(s.a, "{\"ping\":true}") catch return error.Closed;
    }

    fn open(_: *anyopaque) error{Closed}!void {}

    fn event(ctx: *anyopaque, payload: ?json.Value) error{Closed}!void {
        const s: *Seen = @ptrCast(@alignCast(ctx));
        const p = payload orelse {
            s.done = true;
            return;
        };
        s.events.append(s.a, json.stringify(s.a, p, .{}) catch return error.Closed) catch return error.Closed;
    }

    fn reply(ctx: *anyopaque, _: u16, payload: json.Value) void {
        const s: *Seen = @ptrCast(@alignCast(ctx));
        s.whole = payload;
    }
};

const Fixture = struct {
    tok: tokenizer.Tokenizer,
    eos: u32,
    cases: json.Value,

    fn load(a: Allocator) !Fixture {
        const root = (try json.parse(a, fixture_text)).ok;
        const tok = try tokenizer.parse(std.testing.allocator, try json.stringify(a, root.get("tokenizer").?, .{ .ascii = false }));
        return .{ .tok = tok, .eos = @intCast(root.get("eos").?.int64().?), .cases = root.get("cases").? };
    }

    fn ids(f: *const Fixture, a: Allocator, name: []const u8) ![]u32 {
        for (f.cases.array) |c| if (std.mem.eql(u8, c.get("name").?.string, name)) {
            const list = c.get("ids").?.array;
            const out = try a.alloc(u32, list.len);
            for (list, out) |v, *o| o.* = @intCast(v.int64().?);
            return out;
        };
        return error.NoSuchCase;
    }
};

/// A coding agent's tools as an OpenAI client offers them (WebFetch is not offered).
const tools_json =
    \\[{"type":"function","function":{"name":"Write","parameters":{"type":"object","properties":{"file_path":{"type":"string"},"content":{"type":"string"}},"required":["file_path","content"]}}},
    \\{"type":"function","function":{"name":"Edit","parameters":{"type":"object","properties":{"file_path":{"type":"string"},"old_string":{"type":"string"},"new_string":{"type":"string"},"replace_all":{"type":"boolean"}}}}},
    \\{"type":"function","function":{"name":"Bash","parameters":{"type":"object","properties":{"command":{"type":"string"},"description":{"type":"string"},"timeout":{"type":"number"}}}}},
    \\{"type":"function","function":{"name":"TodoWrite","parameters":{"type":"object","properties":{"todos":{"type":"array"}}}}},
    \\{"type":"function","function":{"name":"Read","parameters":{"type":"object","properties":{"file_path":{"type":"string"}}}}}]
;

const Ask = struct {
    case: []const u8,
    stream: bool = true,
    tools: bool = true,
    single: bool = false, // parallel_tool_calls: false
    pattern: []const usize = &.{ 3, 1, 4, 1, 5, 2 },
};

/// One request of `ask` through the chat route (or `messages`'s raw Messages body through the Messages route).
fn chatRun(a: Allocator, f: *Fixture, ask: Ask) !Seen {
    const ids = try f.ids(a, ask.case);
    const cut = std.mem.eql(u8, ask.case, "cut");
    var text: FixtureText = .{ .tok = f.tok, .eos = .{f.eos} };
    var backend: ScriptEngine = .{ .ids = ids, .pattern = ask.pattern, .cut = cut };
    var srv = try server.Server.init(std.testing.allocator, std.testing.io, backend.engine(), text.text(), .{
        .served_name = "test-model",
        .model_ids = &.{"test-model"},
        .enable_thinking = true,
        .use_drafts = false,
    }, null);
    defer srv.deinit();
    const max = if (cut) ids.len else 4096;
    const body = try std.fmt.allocPrint(a, "{{\"model\":\"test-model\",\"messages\":[{{\"role\":\"user\",\"content\":\"x\"}}],\"stream\":{s},\"max_tokens\":{d}{s}{s}}}", .{
        if (ask.stream) "true" else "false",
        max,
        if (ask.tools) ",\"tools\":" ++ tools_json else "",
        if (ask.single) ",\"parallel_tool_calls\":false" else "",
    });
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    var seen: Seen = .{ .a = a };
    openai.run(srv, a, seen.out(), .{ .conn = &conn }, true, (try json.parse(a, body)).ok);
    return seen;
}

const Call = struct { id: []const u8, name: []const u8, arguments: std.ArrayList(u8) = .empty, pieces: usize = 0 };

/// A chat stream read as a client reads it, its order checked on the way: the role first, each call's header (index,
/// id, type, name, empty arguments) once and in index order, its pieces after it and before the next header, no text
/// between a call's header and its last piece (LiteLLM's Messages stream ends a tool_use block at a text delta), valid
/// UTF-8 each, and nothing after the chunk that carries the finish reason.
const Read = struct {
    content: std.ArrayList(u8) = .empty,
    reasoning: std.ArrayList(u8) = .empty,
    calls: std.ArrayList(Call) = .empty,
    finish: ?[]const u8 = null,

    fn of(a: Allocator, seen: *const Seen) !Read {
        var r: Read = .{};
        try std.testing.expect(seen.done);
        var role = false;
        var text_after_header = false; // text came after the newest call's header
        for (seen.events.items) |text| {
            const ev = (try json.parse(a, text)).ok;
            const choices = ev.get("choices") orelse continue;
            if (choices.array.len == 0) continue; // a usage chunk
            try std.testing.expect(r.finish == null); // nothing after the finish
            const choice = choices.array[0];
            if (choice.get("finish_reason")) |fr| if (fr == .string) {
                r.finish = fr.string;
            };
            const delta = choice.get("delta") orelse continue;
            if (delta.get("role")) |_| role = true;
            const said = delta.strField("content") orelse "";
            const thought = delta.strField("reasoning_content") orelse "";
            if (said.len > 0 or thought.len > 0 or delta.get("tool_calls") != null) try std.testing.expect(role);
            try r.content.appendSlice(a, said);
            try r.reasoning.appendSlice(a, thought);
            if (r.calls.items.len > 0 and (said.len > 0 or thought.len > 0)) text_after_header = true;
            const tc = delta.get("tool_calls") orelse continue;
            if (r.finish != null) return error.CallsWithTheFinish;
            for (tc.array) |one| {
                const index: usize = @intCast(one.get("index").?.int64().?);
                const function = one.get("function").?;
                const piece = function.strField("arguments").?;
                try std.testing.expect(std.unicode.utf8ValidateSlice(piece));
                if (one.get("id")) |id| {
                    try std.testing.expectEqual(r.calls.items.len, index); // a new call, in order, once
                    try std.testing.expectEqualStrings("function", one.strField("type").?);
                    try std.testing.expectEqualStrings("", piece);
                    try r.calls.append(a, .{ .id = id.string, .name = function.strField("name").? });
                    text_after_header = false;
                    continue;
                }
                try std.testing.expectEqual(r.calls.items.len -| 1, index); // pieces only for the call being written
                if (text_after_header) return error.PieceAfterText; // a client ended this call at that text
                const c = &r.calls.items[index];
                try c.arguments.appendSlice(a, piece);
                c.pieces += 1;
            }
        }
        return r;
    }
};

/// The whole reply's message: content (null as ""), reasoning, calls and finish reason.
fn wholeMessage(seen: *const Seen) struct { content: []const u8, reasoning: []const u8, calls: []const json.Value, finish: []const u8 } {
    const choice = seen.whole.?.get("choices").?.array[0];
    const message = choice.get("message").?;
    return .{
        .content = message.strField("content") orelse "",
        .reasoning = message.strField("reasoning_content") orelse "",
        .calls = if (message.get("tool_calls")) |c| c.array else &.{},
        .finish = choice.strField("finish_reason").?,
    };
}

/// The stream of `case` at each round pattern against the whole reply: the same calls, names and arguments byte for byte,
/// call i's arguments in at least `min_pieces[i]` pieces; the same content, reasoning and finish reason.
fn expectStreamed(case: []const u8, min_pieces: []const usize) !void {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try Fixture.load(a);
    defer f.tok.deinit();
    const whole_seen = try chatRun(a, &f, .{ .case = case, .stream = false });
    const whole = wholeMessage(&whole_seen);
    try std.testing.expect(whole.calls.len > 0);
    try std.testing.expectEqualStrings("tool_calls", whole.finish);
    for ([_][]const usize{ &.{1}, &.{ 3, 1, 4, 1, 5, 2 }, &.{64} }) |pattern| {
        const seen = try chatRun(a, &f, .{ .case = case, .pattern = pattern });
        const r = try Read.of(a, &seen);
        try std.testing.expectEqualStrings(whole.finish, r.finish.?);
        try std.testing.expectEqualStrings(whole.content, r.content.items);
        try std.testing.expectEqualStrings(whole.reasoning, r.reasoning.items);
        try std.testing.expectEqual(whole.calls.len, r.calls.items.len);
        for (whole.calls, r.calls.items, min_pieces) |w, s, least| {
            const function = w.get("function").?;
            try std.testing.expectEqualStrings(function.strField("name").?, s.name);
            try std.testing.expectEqualStrings(function.strField("arguments").?, s.arguments.items);
            if (pattern.len > 1 or pattern[0] == 1) try std.testing.expect(s.pieces >= least);
        }
    }
}

test "a long Write streams its arguments as they decode, the parse's bytes joined (quotes, backslashes, CRLF, split characters)" {
    try expectStreamed("write", &.{40});
}

test "an Edit streams its strings and sends its boolean when the parameter closes, as the parse writes it" {
    try expectStreamed("edit", &.{6});
}

test "two calls stream one after the other: a Bash command with a number, then a TodoWrite whose array the parse re-writes" {
    // the TodoWrite array goes out whole as its parameter closes, then the call's brace: two pieces
    try expectStreamed("two_calls", &.{ 6, 2 });
}

test "a call to a tool the request did not offer streams as the parse returns it, under its own name" {
    try expectStreamed("unoffered", &.{ 4, 4 });
}

test "DSML written in the reasoning, or quoted in the answer, is no call: no tool delta, the same content" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try Fixture.load(a);
    defer f.tok.deinit();
    const whole = wholeMessage(&try chatRun(a, &f, .{ .case = "quoted", .stream = false }));
    try std.testing.expectEqual(@as(usize, 0), whole.calls.len);
    try std.testing.expectEqualStrings("stop", whole.finish);
    const r = try Read.of(a, &try chatRun(a, &f, .{ .case = "quoted", .pattern = &.{1} }));
    try std.testing.expectEqual(@as(usize, 0), r.calls.items.len);
    try std.testing.expectEqualStrings(whole.finish, r.finish.?);
    try std.testing.expectEqualStrings(whole.content, r.content.items);
    try std.testing.expectEqualStrings(whole.reasoning, r.reasoning.items);
}

test "a call cut by max_tokens ends as the whole reply reports it: length, its content, no brace for the open call" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try Fixture.load(a);
    defer f.tok.deinit();
    const whole = wholeMessage(&try chatRun(a, &f, .{ .case = "cut", .stream = false }));
    try std.testing.expectEqual(@as(usize, 0), whole.calls.len); // the parse rejects a block without its closer
    try std.testing.expectEqualStrings("length", whole.finish);
    const full = wholeMessage(&try chatRun(a, &f, .{ .case = "write", .stream = false })).calls[0].get("function").?.strField("arguments").?;
    for ([_][]const usize{ &.{1}, &.{ 3, 1, 4, 1, 5, 2 } }) |pattern| {
        const r = try Read.of(a, &try chatRun(a, &f, .{ .case = "cut", .pattern = pattern }));
        try std.testing.expectEqualStrings("length", r.finish.?);
        try std.testing.expectEqualStrings(whole.content, r.content.items);
        // the call the reply began went out as it decoded and stays open: a prefix of the whole call's arguments
        try std.testing.expectEqual(@as(usize, 1), r.calls.items.len);
        const sent = r.calls.items[0].arguments.items;
        try std.testing.expect(sent.len > 100 and sent.len < full.len and std.mem.startsWith(u8, full, sent));
        try std.testing.expect((try json.parse(a, sent)) == .err);
    }
}

/// A stream event without what changes between runs: the reply's id and time, its runtime block, the call ids.
fn normal(a: Allocator, text: []const u8) ![]const u8 {
    const v = (try json.parse(a, text)).ok;
    for ([_][]const u8{ "id", "created", "tensorfold", "speculative" }) |k| _ = v.object.orderedRemove(k);
    if (v.get("choices")) |cs| for (cs.array) |c| if (c.get("delta")) |d| if (d.get("tool_calls")) |tc| for (tc.array) |one| {
        if (one.get("id") != null) try one.object.put(a, "id", .{ .string = "call_X" });
    };
    return json.stringify(a, v, .{});
}

/// The streams the patch must leave as they were: no tools (a reply with DSML in it, and one with DSML only quoted),
/// tools without a call, and the one-call policy (its calls go out when the reply ends).
pub const golden_asks = [_]struct { key: []const u8, ask: Ask }{
    .{ .key = "no_tools_write", .ask = .{ .case = "write", .tools = false, .pattern = &.{ 29, 61 } } },
    .{ .key = "no_tools_two_calls", .ask = .{ .case = "two_calls", .tools = false, .pattern = &.{ 7, 13 } } },
    .{ .key = "no_tools_quoted", .ask = .{ .case = "quoted", .tools = false } },
    .{ .key = "tools_quoted", .ask = .{ .case = "quoted" } },
    .{ .key = "single_unoffered", .ask = .{ .case = "unoffered", .single = true } },
    .{ .key = "single_two_calls_by_token", .ask = .{ .case = "two_calls", .single = true, .pattern = &.{1} } },
};

/// The golden file's text: each golden ask's normalized events (how testdata/dsml_stream_golden.json was recorded).
pub fn goldenJson(a: Allocator) ![]const u8 {
    var f = try Fixture.load(a);
    defer f.tok.deinit();
    const o = try json.newObject(a);
    for (golden_asks) |g| {
        const seen = try chatRun(a, &f, g.ask);
        const list = try a.alloc(json.Value, seen.events.items.len);
        for (seen.events.items, list) |e, *slot| slot.* = .{ .string = try normal(a, e) };
        try o.put(a, g.key, .{ .array = list });
    }
    return json.stringify(a, .{ .object = o }, .{ .ascii = true });
}

test "without tools, without a call, or with the one-call policy, a stream is event for event what it was before the patch" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try Fixture.load(a);
    defer f.tok.deinit();
    const golden = (try json.parse(a, golden_text)).ok;
    for (golden_asks) |g| {
        const want = golden.get(g.key).?.array;
        const seen = try chatRun(a, &f, g.ask);
        try std.testing.expect(seen.done);
        try std.testing.expectEqual(want.len, seen.events.items.len);
        for (want, seen.events.items) |w, e| try std.testing.expectEqualStrings(w.string, try normal(a, e));
    }
}

/// Drains a socket into `out` until its other end closes.
fn drain(fd: std.posix.fd_t, out: *std.ArrayList(u8)) void {
    var buf: [65536]u8 = undefined;
    while (true) {
        const n = std.posix.read(fd, &buf) catch return;
        if (n == 0) return;
        out.appendSlice(std.heap.page_allocator, buf[0..n]) catch return;
    }
}

/// `body` (a Messages request) through the engine's /v1/messages route, as a client reads its bytes.
fn messagesRun(a: Allocator, f: *Fixture, case: []const u8, body: []const u8) ![]const u8 {
    const ids = try f.ids(a, case);
    var text: FixtureText = .{ .tok = f.tok, .eos = .{f.eos} };
    var backend: ScriptEngine = .{ .ids = ids, .pattern = &.{ 3, 1, 4, 1, 5, 2 }, .cut = false };
    var srv = try server.Server.init(std.testing.allocator, std.testing.io, backend.engine(), text.text(), .{
        .served_name = "test-model",
        .model_ids = &.{"test-model"},
        .enable_thinking = true,
        .use_drafts = false,
    }, null);
    defer srv.deinit();
    const linux = std.os.linux;
    var fds: [2]i32 = undefined;
    if (linux.errno(linux.socketpair(linux.AF.UNIX, linux.SOCK.STREAM, 0, &fds)) != .SUCCESS) return error.SocketPair;
    var got: std.ArrayList(u8) = .empty;
    defer got.deinit(std.heap.page_allocator);
    const reader = try std.Thread.spawn(.{}, drain, .{ fds[1], &got });
    var headers = [_]@import("http_conn.zig").Header{.{ .name = "Content-Length", .value = try std.fmt.allocPrint(a, "{d}", .{body.len}) }};
    var conn: Conn = .{ .fd = fds[0], .peer = "", .buf = try a.dupe(u8, body), .end = body.len, .gpa = a, .headers = &headers, .path = "/v1/messages", .method = "POST", .version = "HTTP/1.1" };
    anthropic.post(srv, &conn, a);
    _ = std.posix.system.close(fds[0]);
    reader.join();
    _ = std.posix.system.close(fds[1]);
    return a.dupe(u8, got.items);
}

test "the engine's Messages route streams a DSML call as input_json_delta pieces that join to the chat reply's arguments" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try Fixture.load(a);
    defer f.tok.deinit();
    const want = wholeMessage(&try chatRun(a, &f, .{ .case = "two_calls", .stream = false })).calls;
    const tools =
        \\"tools":[{"name":"Bash","input_schema":{"type":"object","properties":{"command":{"type":"string"},"description":{"type":"string"},"timeout":{"type":"number"}}}},
        \\{"name":"TodoWrite","input_schema":{"type":"object","properties":{"todos":{"type":"array"}}}}]
    ;
    const body = "{\"model\":\"test-model\",\"max_tokens\":4096,\"stream\":true,\"thinking\":{\"type\":\"enabled\",\"budget_tokens\":1024}," ++ tools ++ ",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}";
    const bytes = try messagesRun(a, &f, "two_calls", body);
    const head = std.mem.indexOf(u8, bytes, "\r\n\r\n") orelse return error.NoHead;
    try std.testing.expect(std.mem.startsWith(u8, bytes, "HTTP/1.1 200"));
    // the tool_use blocks in order, each one's partial_json pieces joined
    var names: std.ArrayList([]const u8) = .empty;
    var joined: std.ArrayList(std.ArrayList(u8)) = .empty;
    var pieces: std.ArrayList(usize) = .empty;
    var blocks: std.AutoHashMapUnmanaged(i64, usize) = .empty;
    var stop: []const u8 = "";
    var frames = std.mem.splitSequence(u8, bytes[head + 4 ..], "\n\n");
    while (frames.next()) |frame| {
        const at = std.mem.indexOf(u8, frame, "data: ") orelse continue;
        const ev = (try json.parse(a, frame[at + 6 ..])).ok;
        const kind = ev.strField("type") orelse continue;
        if (std.mem.eql(u8, kind, "content_block_start")) {
            const cb = ev.get("content_block").?;
            if (!std.mem.eql(u8, cb.strField("type").?, "tool_use")) continue;
            try blocks.put(a, ev.get("index").?.int64().?, names.items.len);
            try names.append(a, cb.strField("name").?);
            try joined.append(a, .empty);
            try pieces.append(a, 0);
        } else if (std.mem.eql(u8, kind, "content_block_delta")) {
            const d = ev.get("delta").?;
            if (!std.mem.eql(u8, d.strField("type").?, "input_json_delta")) continue;
            const i = blocks.get(ev.get("index").?.int64().?).?;
            try joined.items[i].appendSlice(a, d.strField("partial_json").?);
            pieces.items[i] += 1;
        } else if (std.mem.eql(u8, kind, "message_delta")) stop = ev.get("delta").?.strField("stop_reason").?;
    }
    try std.testing.expectEqualStrings("tool_use", stop);
    try std.testing.expectEqual(want.len, names.items.len);
    for (want, names.items, joined.items, pieces.items, [_]usize{ 6, 2 }) |w, name, j, n, least| {
        try std.testing.expectEqualStrings(w.get("function").?.strField("name").?, name);
        try std.testing.expectEqualStrings(w.get("function").?.strField("arguments").?, j.items);
        try std.testing.expect(n >= least);
    }
}

/// Bytes as tokens, for replies the fixture does not hold; with `eos`, id 256 is the end token.
const ByteText = struct {
    eos: bool = false,

    fn text(t: *ByteText) model_text.Text {
        return .{ .ctx = t, .vtable = &.{ .encode = FixtureText.encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = FixtureText.render, .template_source = FixtureText.templateSource } };
    }

    /// The bytes, a character cut short at the end as U+FFFD (the text seam's contract, as the real decoder keeps it).
    fn decode(_: *anyopaque, a: Allocator, ids: []const u32) Allocator.Error![]u8 {
        const out = try a.alloc(u8, ids.len);
        for (ids, out) |id, *b| b.* = @intCast(id);
        var back: usize = 0;
        while (back < @min(4, out.len)) {
            back += 1;
            const lead = out[out.len - back];
            if (lead & 0xc0 == 0x80) continue;
            const n = std.unicode.utf8ByteSequenceLength(lead) catch break;
            if (n > back) return std.mem.concat(a, u8, &.{ out[0 .. out.len - back], "\u{fffd}" });
            break;
        }
        return out;
    }

    fn tokenId(_: *anyopaque, _: []const u8) ?u32 {
        return null;
    }

    fn tokenString(_: *anyopaque, a: Allocator, id: u32) Allocator.Error![]u8 {
        return std.fmt.allocPrint(a, "{d}", .{id});
    }

    fn vocabSize(_: *anyopaque) u32 {
        return 257;
    }

    fn eosIds(ctx: *anyopaque) []const u32 {
        const t: *ByteText = @ptrCast(@alignCast(ctx));
        return if (t.eos) &.{256} else &.{};
    }
};

/// `answer` (no thinking) through the chat route with the tools above, whole or streamed in rounds of 5 bytes.
fn bytesRun(a: Allocator, answer: []const u8, stream: bool) !Seen {
    const ids = try a.alloc(u32, answer.len);
    for (answer, ids) |b, *id| id.* = b;
    var text: ByteText = .{};
    var backend: ScriptEngine = .{ .ids = ids, .pattern = &.{5}, .cut = false };
    var srv = try server.Server.init(std.testing.allocator, std.testing.io, backend.engine(), text.text(), .{
        .served_name = "test-model",
        .model_ids = &.{"test-model"},
        .enable_thinking = false,
        .use_drafts = false,
    }, null);
    defer srv.deinit();
    const body = try std.mem.concat(a, u8, &.{ "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"stream\":", if (stream) "true" else "false", ",\"max_tokens\":4096,\"tools\":" ++ tools_json ++ "}" });
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    var seen: Seen = .{ .a = a };
    openai.run(srv, a, seen.out(), .{ .conn = &conn }, true, (try json.parse(a, body)).ok);
    return seen;
}

test "a call in another form after the DSML block goes out when the reply ends, under the next index" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const D = "\u{ff5c}DSML\u{ff5c}";
    const answer = "Listing it.\n\n<" ++ D ++ " calls>\n<" ++ D ++ " invoke name=\"Bash\">\n<" ++ D ++ " parameter name=\"command\" string=\"true\">ls -la \"/tmp\"</" ++ D ++
        " parameter>\n</" ++ D ++ " invoke>\n</" ++ D ++ " calls>\n<tool_call>{\"name\":\"Read\",\"arguments\":{\"file_path\":\"/etc/hosts\"}}</tool_call>";
    const whole = wholeMessage(&try bytesRun(a, answer, false));
    try std.testing.expectEqual(@as(usize, 2), whole.calls.len);
    const r = try Read.of(a, &try bytesRun(a, answer, true));
    try std.testing.expectEqualStrings(whole.finish, r.finish.?);
    // the stream shows the newline between the two forms as prose, the whole reply strips it (as before the patch)
    try std.testing.expectEqualStrings(whole.content, @import("reply_text.zig").pyStrip(r.content.items));
    try std.testing.expectEqual(@as(usize, 2), r.calls.items.len);
    for (whole.calls, r.calls.items) |w, s| {
        try std.testing.expectEqualStrings(w.get("function").?.strField("name").?, s.name);
        try std.testing.expectEqualStrings(w.get("function").?.strField("arguments").?, s.arguments.items);
    }
    try std.testing.expect(r.calls.items[0].pieces >= 3); // the DSML call as it decoded
    try std.testing.expectEqual(@as(usize, 1), r.calls.items[1].pieces); // the other one whole, at the end
}

/// How `bytesAsk` runs a reply: its rounds, thinking off, cut by max_tokens at its end, its end token, a stop string.
const Bytes = struct { pattern: []const usize = &.{5}, stream: bool = true, cut: bool = false, eos: bool = false, stop: ?[]const u8 = null, mode: tool_streaming.Mode = .live };

/// `answer` (no thinking) through the chat route with the tools above, as `o` says.
fn bytesAsk(a: Allocator, answer: []const u8, o: Bytes) !Seen {
    const ids = try a.alloc(u32, answer.len + @intFromBool(o.eos));
    for (answer, ids[0..answer.len]) |b, *id| id.* = b;
    if (o.eos) ids[answer.len] = 256;
    var text: ByteText = .{ .eos = o.eos };
    var backend: ScriptEngine = .{ .ids = ids, .pattern = o.pattern, .cut = o.cut };
    var srv = try server.Server.init(std.testing.allocator, std.testing.io, backend.engine(), text.text(), .{
        .served_name = "test-model",
        .model_ids = &.{"test-model"},
        .enable_thinking = false,
        .use_drafts = false,
        .tool_stream = o.mode,
    }, null);
    defer srv.deinit();
    const stop = if (o.stop) |st| try std.fmt.allocPrint(a, ",\"stop\":[{s}]", .{try json.quote(a, st, .{})}) else "";
    const body = try std.fmt.allocPrint(a, "{{\"model\":\"test-model\",\"messages\":[{{\"role\":\"user\",\"content\":\"x\"}}],\"stream\":{s},\"max_tokens\":{d}{s},\"tools\":{s}}}", .{
        if (o.stream) "true" else "false", if (o.cut) answer.len else 4096, stop, tools_json,
    });
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    var seen: Seen = .{ .a = a };
    openai.run(srv, a, seen.out(), .{ .conn = &conn }, true, (try json.parse(a, body)).ok);
    return seen;
}

const BD = "\u{ff5c}DSML\u{ff5c}";
const b_open = "\n\n<" ++ BD ++ " calls>\n";
const b_close = "</" ++ BD ++ " calls>";

fn bInvoke(comptime name: []const u8, comptime params: []const u8) []const u8 {
    return "<" ++ BD ++ " invoke name=\"" ++ name ++ "\">\n" ++ params ++ "</" ++ BD ++ " invoke>\n";
}

fn bParam(comptime name: []const u8, comptime value: []const u8) []const u8 {
    return "<" ++ BD ++ " parameter name=\"" ++ name ++ "\" string=\"true\">" ++ value ++ "</" ++ BD ++ " parameter>\n";
}

/// A stream that ends inside a block (the cases below) against the whole reply: the same finish reason and content,
/// no call in the whole reply, and in the stream `open` headers whose arguments are not JSON (a client cannot read
/// them as a whole call). Returns the stream as read.
fn expectCutBlock(a: Allocator, answer: []const u8, o: Bytes, open: usize) !Read {
    var whole_o = o;
    whole_o.stream = false;
    const whole = wholeMessage(&try bytesAsk(a, answer, whole_o));
    try std.testing.expectEqual(@as(usize, 0), whole.calls.len);
    const r = try Read.of(a, &try bytesAsk(a, answer, o));
    try std.testing.expectEqualStrings(whole.finish, r.finish.?);
    try std.testing.expectEqualStrings(whole.content, r.content.items);
    try std.testing.expectEqual(open, r.calls.items.len);
    for (r.calls.items) |c| try std.testing.expect((try json.parse(a, c.arguments.items)) == .err);
    return r;
}

test "TENSORFOLD_TOOL_STREAM: one #14 reply and one reply cut by max_tokens, live and hold" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const closer_in_value = comptime ("Writing." ++ b_open ++ bInvoke("Write", bParam("file_path", "/a") ++ bParam("content", "end: " ++ b_close ++ " and more")) ++ b_close);
    const whole = wholeMessage(&try bytesAsk(a, closer_in_value, .{ .stream = false, .eos = true }));
    try std.testing.expectEqual(@as(usize, 1), whole.calls.len);
    for ([_]tool_streaming.Mode{ .live, .hold }) |mode| for ([_][]const usize{ &.{1}, &.{5}, &.{closer_in_value.len} }) |p| {
        const r = try Read.of(a, &try bytesAsk(a, closer_in_value, .{ .eos = true, .pattern = p, .mode = mode }));
        try std.testing.expectEqualStrings(whole.finish, r.finish.?);
        try std.testing.expectEqualStrings(whole.content, r.content.items);
        try std.testing.expectEqual(@as(usize, 1), r.calls.items.len);
        try std.testing.expectEqualStrings(whole.calls[0].get("function").?.strField("arguments").?, r.calls.items[0].arguments.items);
        // hold: the arguments in one piece when the reply ended; live: in pieces as they decoded
        if (mode == .hold) try std.testing.expectEqual(@as(usize, 1), r.calls.items[0].pieces) else if (p[0] < closer_in_value.len) try std.testing.expect(r.calls.items[0].pieces > 1);
    };
    const cut = comptime ("Writing." ++ b_open ++ "<" ++ BD ++ " invoke name=\"Write\">\n" ++ bParam("file_path", "/a") ++ "<" ++ BD ++ " parameter name=\"content\" string=\"true\"># History of the trans");
    for ([_][]const usize{ &.{1}, &.{5}, &.{ 3, 4 } }) |p| {
        // live: the open call's header and its pieces so far, not JSON, and length; hold: no call, the whole reply's content
        const live = try expectCutBlock(a, cut, .{ .cut = true, .pattern = p }, 1);
        try std.testing.expectEqualStrings("length", live.finish.?);
        const hold = try expectCutBlock(a, cut, .{ .cut = true, .pattern = p, .mode = .hold }, 0);
        try std.testing.expectEqualStrings("length", hold.finish.?);
    }
}

test "thinking off, a Harmony marker in a string value and the reply cut inside it: no read past the text, the call stays open" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // the finish reads Harmony channels out of this text (content ""), so its parse no longer reads the streamed answer
    const answer = "Searching." ++ b_open ++ "<" ++ BD ++ " invoke name=\"Bash\">\n<" ++ BD ++ " parameter name=\"command\" string=\"true\">grep -n '<|channel|>' zig/src/server/reply_text.zig and some more words that the cut ends";
    const whole = wholeMessage(&try bytesAsk(a, answer, .{ .cut = true, .stream = false }));
    try std.testing.expectEqualStrings("length", whole.finish);
    try std.testing.expectEqualStrings("", whole.content);
    for ([_][]const usize{ &.{1}, &.{5}, &.{ 3, 4 } }) |p| {
        const r = try Read.of(a, &try bytesAsk(a, answer, .{ .cut = true, .pattern = p }));
        try std.testing.expectEqualStrings("length", r.finish.?);
        // the prose before the block streamed before the marker came, as without the patch
        try std.testing.expectEqualStrings("Searching.", r.content.items);
        try std.testing.expectEqual(@as(usize, 1), r.calls.items.len);
        try std.testing.expect(std.mem.startsWith(u8, "{\"command\":\"grep -n '<|channel|>'", r.calls.items[0].arguments.items));
        try std.testing.expect((try json.parse(a, r.calls.items[0].arguments.items)) == .err);
    }
    // a stop string and an end token inside the value end the same way
    const stopped = try Read.of(a, &try bytesAsk(a, answer ++ " STOPHERE more", .{ .stop = "STOPHERE" }));
    try std.testing.expectEqualStrings("stop", stopped.finish.?);
    try std.testing.expect(stopped.calls.items.len == 1 and (try json.parse(a, stopped.calls.items[0].arguments.items)) == .err);
    const ended = try Read.of(a, &try bytesAsk(a, answer, .{ .eos = true }));
    try std.testing.expectEqualStrings("stop", ended.finish.?);
    try std.testing.expect(ended.calls.items.len == 1 and (try json.parse(a, ended.calls.items[0].arguments.items)) == .err);
}

test "thinking off, a Harmony marker in a whole block: the parse takes no call, and the stream stops at the marker" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const answer = comptime ("Searching." ++ b_open ++ bInvoke("Bash", bParam("command", "grep -n '<|channel|>' zig/src/server/reply_text.zig")) ++ b_close);
    const whole = wholeMessage(&try bytesAsk(a, answer, .{ .stream = false }));
    try std.testing.expectEqual(@as(usize, 0), whole.calls.len);
    try std.testing.expectEqualStrings("stop", whole.finish);
    for ([_][]const usize{ &.{1}, &.{5}, &.{1 << 20} }) |p| {
        const r = try Read.of(a, &try bytesAsk(a, answer, .{ .pattern = p }));
        try std.testing.expectEqualStrings("stop", r.finish.?);
        // a round that holds the marker turns the streamer off before its call closes: no brace, no whole call
        try std.testing.expect(r.calls.items.len <= 1);
        for (r.calls.items) |c| try std.testing.expect((try json.parse(a, c.arguments.items)) == .err);
    }
}

test "prose after a block, in the round that closes it, goes out after the call's last piece" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const one = comptime ("Listing." ++ b_open ++ bInvoke("Bash", bParam("command", "ls -la")) ++ b_close ++ " Then I read it.");
    // two blocks with prose between them and after them, in rounds that hold a closer, prose and the next opener
    const two = comptime (one ++ b_open ++ bInvoke("Read", bParam("file_path", "/etc/hosts")) ++ b_close ++ " Done.");
    for ([_][]const u8{ one, two }, [_]usize{ 1, 2 }) |answer, n| {
        const whole = wholeMessage(&try bytesAsk(a, answer, .{ .stream = false }));
        try std.testing.expectEqual(n, whole.calls.len);
        try std.testing.expectEqualStrings("tool_calls", whole.finish);
        for ([_][]const usize{ &.{1}, &.{5}, &.{16}, &.{ 3, 4 }, &.{ 40, 7 }, &.{1 << 20} }) |p| {
            const r = try Read.of(a, &try bytesAsk(a, answer, .{ .pattern = p }));
            try std.testing.expectEqualStrings(whole.finish, r.finish.?);
            // the stream hides a block's leading blank line with the block (reply_text.calls), as without the patch
            try std.testing.expectEqualStrings(try std.mem.replaceOwned(u8, a, whole.content, "\n\n", ""), r.content.items);
            try std.testing.expectEqual(n, r.calls.items.len);
            for (whole.calls, r.calls.items) |w, s| {
                try std.testing.expectEqualStrings(w.get("function").?.strField("name").?, s.name);
                try std.testing.expectEqualStrings(w.get("function").?.strField("arguments").?, s.arguments.items);
            }
        }
    }
}

test "a block cut in its second invoke sends no whole call: the first call's brace and the second call wait for the block's closer" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const head = comptime ("Two reads." ++ b_open ++ bInvoke("Read", bParam("file_path", "/etc/hosts")) ++ "<" ++ BD ++ " invoke name=\"Write\">\n" ++
        bParam("file_path", "/tmp/a") ++ "<" ++ BD ++ " parameter name=\"content\" string=\"true\">line one\nline two");
    // cut by max_tokens, by a stop string, and by the end token: the parse takes no call from the block
    for ([_]Bytes{ .{ .cut = true }, .{ .cut = true, .pattern = &.{1} }, .{ .cut = true, .pattern = &.{1 << 20} }, .{ .stop = "STOPHERE" }, .{ .eos = true } }) |o| {
        const answer = if (o.stop != null) head ++ " STOPHERE and more" else head;
        const r = try expectCutBlock(a, answer, o, 1);
        // the first call went out as it decoded and stays open; the second never went out
        try std.testing.expectEqualStrings("Read", r.calls.items[0].name);
        try std.testing.expectEqualStrings("{\"file_path\":\"/etc/hosts\"", r.calls.items[0].arguments.items);
    }
    // the block whole: both calls, the second as the block closes
    const whole_text = comptime (head ++ "</" ++ BD ++ " parameter>\n</" ++ BD ++ " invoke>\n" ++ b_close);
    const whole = wholeMessage(&try bytesAsk(a, whole_text, .{ .stream = false }));
    try std.testing.expectEqual(@as(usize, 2), whole.calls.len);
    for ([_][]const usize{ &.{1}, &.{5}, &.{ 3, 4 } }) |p| {
        const r = try Read.of(a, &try bytesAsk(a, whole_text, .{ .pattern = p }));
        try std.testing.expectEqualStrings("tool_calls", r.finish.?);
        try std.testing.expectEqual(@as(usize, 2), r.calls.items.len);
        for (whole.calls, r.calls.items) |w, s| try std.testing.expectEqualStrings(w.get("function").?.strField("arguments").?, s.arguments.items);
        if (p[0] == 1) try std.testing.expect(r.calls.items[0].pieces >= 5); // the first call streamed
    }
    // a whole block, then a second block cut in its second invoke: the first block's call is whole and is the parse's
    const after = comptime ("First." ++ b_open ++ bInvoke("Bash", bParam("command", "ls")) ++ b_close ++ head["Two reads.".len..]);
    const w2 = wholeMessage(&try bytesAsk(a, after, .{ .cut = true, .stream = false }));
    try std.testing.expectEqual(@as(usize, 1), w2.calls.len);
    const r2 = try Read.of(a, &try bytesAsk(a, after, .{ .cut = true }));
    // the whole reply reports its one call; the stream, holding a call the parse drops, ends as cut
    try std.testing.expectEqualStrings("tool_calls", w2.finish);
    try std.testing.expectEqualStrings("length", r2.finish.?);
    try std.testing.expectEqualStrings(w2.content, @import("reply_text.zig").pyStrip(r2.content.items));
    try std.testing.expectEqual(@as(usize, 2), r2.calls.items.len);
    try std.testing.expectEqualStrings(w2.calls[0].get("function").?.strField("arguments").?, r2.calls.items[0].arguments.items);
    try std.testing.expect((try json.parse(a, r2.calls.items[1].arguments.items)) == .err);
}

test "an XML call after a DSML block that the end token leaves open goes out once, under the next index" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const answer = comptime ("Listing it." ++ b_open ++ bInvoke("Bash", bParam("command", "ls")) ++ b_close ++ "\n<tool_call>\n<function=Read>\n<parameter=file_path>\n/etc/hosts\n</parameter>\n");
    const whole = wholeMessage(&try bytesAsk(a, answer, .{ .eos = true, .stream = false }));
    try std.testing.expectEqual(@as(usize, 2), whole.calls.len);
    for ([_][]const usize{ &.{1}, &.{5}, &.{1 << 20} }) |p| {
        const r = try Read.of(a, &try bytesAsk(a, answer, .{ .eos = true, .pattern = p }));
        try std.testing.expectEqualStrings(whole.finish, r.finish.?);
        try std.testing.expectEqual(@as(usize, 2), r.calls.items.len);
        for (whole.calls, r.calls.items) |w, s| {
            try std.testing.expectEqualStrings(w.get("function").?.strField("name").?, s.name);
            try std.testing.expectEqualStrings(w.get("function").?.strField("arguments").?, s.arguments.items);
        }
    }
}

/// One server for many requests, its replies scripted byte by byte (`ByteText`), each request through the chat route.
const ByteServer = struct {
    text: ByteText = .{},
    backend: ScriptEngine = .{ .ids = &.{}, .pattern = &.{5}, .cut = false },
    srv: *server.Server = undefined,
    mode: tool_streaming.Mode = .live,
    max_tokens: usize = 4096,

    fn init(b: *ByteServer, thinking: bool) !void {
        b.srv = try server.Server.init(std.testing.allocator, std.testing.io, b.backend.engine(), b.text.text(), .{
            .served_name = "test-model",
            .model_ids = &.{"test-model"},
            .enable_thinking = thinking,
            .use_drafts = false,
            .tool_stream = b.mode,
        }, null);
    }

    fn deinit(b: *ByteServer) void {
        b.srv.deinit();
    }

    const Got = struct {
        content: []const u8,
        reasoning: []const u8,
        calls: usize,
        finish: []const u8,
        done: bool,
        errors: usize,
        names: []const []const u8 = &.{},
        args: []const []const u8 = &.{}, // each call's arguments, its pieces joined
        after_call: bool = false, // prose came after a call's header
        content_events: usize = 0,
    };

    /// `answer` through the chat route with the tools above: whole, or streamed in rounds of `pattern` bytes; `single` is
    /// `parallel_tool_calls: false`. What a client gets: the content and reasoning joined, the calls (their names and
    /// arguments), the finish reason.
    fn chat(b: *ByteServer, a: Allocator, answer: []const u8, stream: bool, single: bool, pattern: []const usize) !Got {
        const ids = try a.alloc(u32, answer.len);
        for (answer, ids) |c, *id| id.* = c;
        b.backend.ids = ids;
        b.backend.pattern = pattern;
        const body = try std.mem.concat(a, u8, &.{ "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"stream\":", if (stream) "true" else "false", if (single) ",\"parallel_tool_calls\":false" else "", try std.fmt.allocPrint(a, ",\"max_tokens\":{d},\"tools\":", .{b.max_tokens}), tools_json ++ "}" });
        var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
        var seen: Seen = .{ .a = a };
        openai.run(b.srv, a, seen.out(), .{ .conn = &conn }, true, (try json.parse(a, body)).ok);
        var content: std.ArrayList(u8) = .empty;
        var reasoning: std.ArrayList(u8) = .empty;
        var got: Got = .{ .content = "", .reasoning = "", .calls = 0, .finish = "", .done = seen.done, .errors = 0 };
        var names: std.ArrayList([]const u8) = .empty;
        var args: std.ArrayList(std.ArrayList(u8)) = .empty;
        if (!stream) {
            const whole = seen.whole orelse return error.NoReply;
            if (whole.has("error")) got.errors += 1;
            const c = whole.get("choices").?.array[0];
            const m = c.get("message").?;
            try content.appendSlice(a, m.strField("content") orelse "");
            try reasoning.appendSlice(a, m.strField("reasoning_content") orelse "");
            if (m.get("tool_calls")) |t| for (t.array) |one| {
                try names.append(a, one.get("function").?.strField("name").?);
                try args.append(a, .empty);
                try args.items[args.items.len - 1].appendSlice(a, one.get("function").?.strField("arguments").?);
            };
            got.finish = c.strField("finish_reason") orelse "";
        } else for (seen.events.items) |e| {
            const v = (try json.parse(a, e)).ok;
            if (v.has("error")) got.errors += 1;
            const cs = v.get("choices") orelse continue;
            if (cs.array.len == 0) continue;
            const d = cs.array[0].get("delta") orelse continue;
            const said = d.strField("content") orelse "";
            if (said.len > 0) {
                got.content_events += 1;
                if (names.items.len > 0) got.after_call = true;
            }
            try content.appendSlice(a, said);
            try reasoning.appendSlice(a, d.strField("reasoning_content") orelse "");
            if (d.get("tool_calls")) |t| for (t.array) |one| {
                const function = one.get("function").?;
                if (one.get("id") != null) {
                    try names.append(a, function.strField("name").?);
                    try args.append(a, .empty);
                }
                const index: usize = @intCast(one.get("index").?.int64().?);
                if (index >= args.items.len) return error.PieceBeforeHeader;
                try args.items[index].appendSlice(a, function.strField("arguments") orelse "");
            };
            if (cs.array[0].strField("finish_reason")) |f| got.finish = f;
        }
        got.calls = names.items.len;
        got.names = names.items;
        const joined = try a.alloc([]const u8, args.items.len);
        for (args.items, joined) |arg, *slot| slot.* = arg.items;
        got.args = joined;
        got.content = content.items;
        got.reasoning = reasoning.items;
        return got;
    }
};

// A tool-call block too short to hold the DSML block it opens with (patch 0064): before the length guard the final parse
// sliced past the payload's start. Through the chat route, whole and streamed in rounds of 1, 5, 3-and-4 bytes and in
// one round, with and without the one-call policy and with and without reasoning first: no call, no error, finish "stop",
// a stream that ends with [DONE], and the content each kind of client got before (the stream filter hides an unclosed
// DSML block from its opener on; when it showed nothing the route sends the whole content at the end, unless reasoning
// went out first; a one-call reply drops the block). The same server then answers the next request.
test "a reply holding a too-short DSML tool-call block: no call, no error, the content each client got before, and the server answers the next request" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const D = "\u{ff5c}DSML\u{ff5c}";
    const short = [_][]const u8{
        "<tool_call><" ++ D ++ " calls></tool_call>",
        "<tool_call><" ++ D ++ "tool_calls></tool_call>",
        "Sure: <tool_call> <" ++ D ++ " calls> x </tool_call>",
        "<TOOL_CALL><" ++ D ++ " calls></TOOL_CALL>",
        "<|tool_call><" ++ D ++ " calls><tool_call|>",
        "<ns:tool_call><" ++ D ++ "tool_calls></ns:tool_call>",
        "<tool_call><" ++ D ++ " calls></" ++ D ++ " calls</tool_call>", // one byte short
        "<tool_call><" ++ D ++ "tool_calls></" ++ D ++ "tool_calls</tool_call>", // one byte short, V4
        "P<tool_call><" ++ D ++ " calls></tool_call>",
    };
    // streamed with no limit, by case: no reasoning first, then reasoning first
    const streamed = [2][9][]const u8{
        .{ short[0], short[1], "Sure: ", "<TOOL_CALL>", short[4], "<ns:tool_call>", short[6], short[7], "P" },
        .{ "", "", "Sure: ", "<TOOL_CALL>", "", "<ns:tool_call>", "", "", "P" },
    };
    const single_whole = [9][]const u8{ "", "", "Sure:", "", "", "", "", "", "P" };
    const single_stream = [9][]const u8{ "", "", "Sure: ", "", "", "", "", "", "P" };
    for ([_]bool{ false, true }, 0..) |thinking, ti| {
        var bs: ByteServer = .{};
        try bs.init(thinking);
        defer bs.deinit();
        for (short, 0..) |t, i| for ([_]bool{ false, true }) |single| for ([_]bool{ false, true }) |stream| {
            const answer = if (thinking) try std.mem.concat(a, u8, &.{ "r</think>", t }) else t;
            const want = if (single) (if (stream) single_stream[i] else single_whole[i]) else if (stream) streamed[ti][i] else t;
            for ([_][]const usize{ &.{1}, &.{5}, &.{ 3, 4 }, &.{answer.len} }, 0..) |pattern, pi| {
                if (!stream and pi > 0) continue;
                const g = try bs.chat(a, answer, stream, single, pattern);
                try std.testing.expectEqual(@as(usize, 0), g.calls);
                try std.testing.expectEqual(@as(usize, 0), g.errors);
                try std.testing.expectEqualStrings("stop", g.finish);
                if (stream) try std.testing.expect(g.done);
                try std.testing.expectEqualStrings(want, g.content);
                try std.testing.expectEqualStrings(if (thinking) "r" else "", g.reasoning);
            }
        };
        const next = try bs.chat(a, if (thinking) "r</think>OK" else "OK", false, false, &.{5});
        try std.testing.expectEqualStrings("OK", next.content);
        try std.testing.expectEqualStrings("stop", next.finish);
    }
}

/// `body` POSTed to `path` through the server's route dispatch over a socket, as a client reads the bytes.
fn wirePost(a: Allocator, srv: *server.Server, path: []const u8, body: []const u8) ![]const u8 {
    const linux = std.os.linux;
    var fds: [2]i32 = undefined;
    if (linux.errno(linux.socketpair(linux.AF.UNIX, linux.SOCK.STREAM, 0, &fds)) != .SUCCESS) return error.SocketPair;
    var got: std.ArrayList(u8) = .empty;
    defer got.deinit(std.heap.page_allocator);
    const reader = try std.Thread.spawn(.{}, drain, .{ fds[1], &got });
    var headers = [_]@import("http_conn.zig").Header{.{ .name = "Content-Length", .value = try std.fmt.allocPrint(a, "{d}", .{body.len}) }};
    var conn: Conn = .{ .fd = fds[0], .peer = "", .buf = try a.dupe(u8, body), .end = body.len, .gpa = a, .headers = &headers, .path = path, .method = "POST", .version = "HTTP/1.1" };
    @import("routes.zig").dispatch(srv, &conn, a);
    _ = std.posix.system.close(fds[0]);
    reader.join();
    _ = std.posix.system.close(fds[1]);
    return a.dupe(u8, got.items);
}

test "the same too-short block over HTTP through the chat, Responses and Messages routes: 200, no call, a normal end, and the next request answered" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const D = "\u{ff5c}DSML\u{ff5c}";
    var bs: ByteServer = .{};
    try bs.init(false);
    defer bs.deinit();
    const schema = "{\"type\":\"object\",\"properties\":{\"content\":{\"type\":\"string\"}}}";
    // end_whole: the field of the whole reply that says it ended normally, and its value
    const Route = struct { path: []const u8, body: []const u8, end_whole: [2][]const u8, end_stream: []const u8, call: []const u8 };
    for ([_][]const u8{ "<tool_call><" ++ D ++ " calls></tool_call>", "P<tool_call><" ++ D ++ "tool_calls></tool_call>", "OK" }) |answer| {
        for ([_]bool{ false, true }) |stream| {
            const st = if (stream) "true" else "false";
            const routes = [_]Route{
                .{ .path = "/v1/chat/completions", .body = try std.mem.concat(a, u8, &.{ "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"max_tokens\":4096,\"stream\":", st, ",\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"write\",\"parameters\":", schema, "}}]}" }), .end_whole = .{ "finish_reason", "stop" }, .end_stream = "data: [DONE]", .call = "\"tool_calls\"" },
                .{ .path = "/v1/responses", .body = try std.mem.concat(a, u8, &.{ "{\"model\":\"test-model\",\"input\":\"x\",\"max_output_tokens\":4096,\"stream\":", st, ",\"tools\":[{\"type\":\"function\",\"name\":\"write\",\"parameters\":", schema, "}]}" }), .end_whole = .{ "status", "completed" }, .end_stream = "response.completed", .call = "\"function_call\"" },
                .{ .path = "/v1/messages", .body = try std.mem.concat(a, u8, &.{ "{\"model\":\"test-model\",\"max_tokens\":4096,\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"stream\":", st, ",\"tools\":[{\"name\":\"write\",\"input_schema\":", schema, "}]}" }), .end_whole = .{ "stop_reason", "end_turn" }, .end_stream = "message_stop", .call = "\"tool_use\"" },
            };
            for (routes) |r| {
                const ids = try a.alloc(u32, answer.len);
                for (answer, ids) |c, *id| id.* = c;
                bs.backend.ids = ids;
                bs.backend.pattern = &.{3};
                const wire = try wirePost(a, bs.srv, r.path, r.body);
                std.testing.expect(std.mem.startsWith(u8, wire, "HTTP/1.1 200 ")) catch |e| {
                    std.debug.print("{s} stream={}: {s}\n", .{ r.path, stream, wire[0..@min(wire.len, 300)] });
                    return e;
                };
                try std.testing.expect(std.mem.indexOf(u8, wire, r.call) == null);
                if (stream) {
                    try std.testing.expect(std.mem.indexOf(u8, wire, r.end_stream) != null);
                } else {
                    const body = wire[(std.mem.indexOf(u8, wire, "\r\n\r\n") orelse return error.NoBody) + 4 ..];
                    const v = (try json.parse(a, body)).ok;
                    const holder = if (std.mem.eql(u8, r.end_whole[0], "finish_reason")) v.get("choices").?.array[0] else v;
                    try std.testing.expectEqualStrings(r.end_whole[1], holder.strField(r.end_whole[0]) orelse "");
                }
            }
        }
    }
}

/// `s` without its whitespace. The stream drops the two newlines the template writes before the V4.1 opener along with
/// the opener and sends a trailing space it held; the whole reply's content keeps the newlines and ends trimmed (as it
/// did before patch 0064), so the two are compared without whitespace.
fn withoutBlanks(a: Allocator, s: []const u8) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    for (s) |c| if (!std.ascii.isWhitespace(c)) try out.append(a, c);
    return out.items;
}

/// A DSML block of one spelling: V4.1's, its opener with the two newlines the template writes before it or without them,
/// or V4's; `tag` is the opener alone.
const Spelling = struct { open: []const u8, tag: []const u8, close: []const u8, invoke: []const u8, invoke_close: []const u8, param: []const u8, param_close: []const u8 };
const spellings = [_]Spelling{
    .{ .open = "\n\n<" ++ BD ++ " calls>\n", .tag = "<" ++ BD ++ " calls>", .close = "</" ++ BD ++ " calls>", .invoke = "<" ++ BD ++ " invoke name=\"", .invoke_close = "</" ++ BD ++ " invoke>\n", .param = "<" ++ BD ++ " parameter name=\"", .param_close = "</" ++ BD ++ " parameter>\n" },
    .{ .open = "<" ++ BD ++ " calls>\n", .tag = "<" ++ BD ++ " calls>", .close = "</" ++ BD ++ " calls>", .invoke = "<" ++ BD ++ " invoke name=\"", .invoke_close = "</" ++ BD ++ " invoke>\n", .param = "<" ++ BD ++ " parameter name=\"", .param_close = "</" ++ BD ++ " parameter>\n" },
    .{ .open = "<" ++ BD ++ "tool_calls>\n", .tag = "<" ++ BD ++ "tool_calls>", .close = "</" ++ BD ++ "tool_calls>", .invoke = "<" ++ BD ++ "invoke name=\"", .invoke_close = "</" ++ BD ++ "invoke>\n", .param = "<" ++ BD ++ "parameter name=\"", .param_close = "</" ++ BD ++ "parameter>\n" },
};

/// One Write call (or `name`'s) in a block of spelling `sp`, its content value as given.
fn spWrite(a: Allocator, sp: Spelling, name: []const u8, value: []const u8) ![]const u8 {
    return std.mem.concat(a, u8, &.{ sp.open, sp.invoke, name, "\">\n", sp.param, "file_path\" string=\"true\">/tmp/dsml.md", sp.param_close, sp.param, "content\" string=\"true\">", value, sp.param_close, sp.invoke_close, sp.close });
}

// DSML values are not escaped, so a file about DSML can hold its block's own tags (#14): read to its first closer, a block
// whose value holds that closer ended inside the value, the call was lost and the tags after the value reached the text.
// Both spellings, V4.1's block with and without the two newlines before it, a value holding the closer, the opener,
// both, or another call form's markup, prose before and after the block: the whole reply's one call holds the whole value and no tag reaches its
// content; a stream in rounds of 1, 5, 3-and-4 bytes, in one round and at every two-round split gets the same content
// and call (its arguments joined from the pieces that went out as they decoded), and the prose before the block as it decoded.
test "a DSML value holding its block's own closer: the call keeps its whole value, whole and streamed alike, at every split" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var bs: ByteServer = .{};
    try bs.init(false);
    defer bs.deinit();
    for (spellings) |sp| {
        const values = [_][]const u8{
            try std.mem.concat(a, u8, &.{ "end: ", sp.close }),
            try std.mem.concat(a, u8, &.{ "start: ", sp.tag }),
            try std.mem.concat(a, u8, &.{ sp.tag, " inner ", sp.close }),
            try std.mem.concat(a, u8, &.{ sp.close, " then ", sp.tag }),
            "q: <tool_call><function=Read></function></tool_call>", // another call form, no closer: no call of its own
        };
        for (values) |value| {
            const answer = try std.mem.concat(a, u8, &.{ "Writing.", try spWrite(a, sp, "Write", value), "\nDone." });
            const whole = try bs.chat(a, answer, false, false, &.{5});
            try std.testing.expectEqualStrings("tool_calls", whole.finish);
            try std.testing.expectEqual(@as(usize, 1), whole.calls);
            try std.testing.expectEqualStrings("Write", whole.names[0]);
            try std.testing.expectEqualStrings(value, (try json.parse(a, whole.args[0])).ok.strField("content").?);
            try std.testing.expect(std.mem.indexOf(u8, whole.content, "\u{ff5c}") == null);
            var patterns: std.ArrayList([]const usize) = .empty;
            try patterns.appendSlice(a, &.{ &.{1}, &.{5}, &.{ 3, 4 }, &.{answer.len} });
            for (1..answer.len) |k| try patterns.append(a, try a.dupe(usize, &.{ k, answer.len }));
            for (patterns.items) |p| {
                const r = try bs.chat(a, answer, true, false, p);
                try std.testing.expect(r.done);
                try std.testing.expectEqualStrings(whole.finish, r.finish);
                try std.testing.expectEqualStrings(try withoutBlanks(a, whole.content), try withoutBlanks(a, r.content));
                try std.testing.expectEqual(@as(usize, 1), r.calls);
                try std.testing.expectEqualStrings("Write", r.names[0]);
                try std.testing.expectEqualStrings(whole.args[0], r.args[0]);
            }
            // the prose before the block, a round a byte, went out as it decoded
            try std.testing.expect((try bs.chat(a, answer, true, false, &.{1})).content_events > "Writing".len);
        }
    }
}

// A DSML value that holds its block's closer and, after it, a call in another form: read to its first closer the block
// gives no call and the other form's call follows it; read past it the block gives its own call. Each reading has one
// call, and on a tie the parse keeps the reading cut at the first closer, so the whole reply's call is the other form's
// (Read). A V4.1 stream reads the block as reply_text.blockEnd does, past the closer, and sends the Write's header and
// value as it decodes; what would complete it waits for the reply's end, and the parse does not take it: the Write stays
// open (its arguments not JSON), no complete call the parse does not return goes out, and the stream ends as the reply
// did (stop), the settle warning logged. With the one-call policy the calls go out at the end, the parse's; a V4 block is
// another call form to the stream, so its calls go out at the end too: Read, as the whole reply has it.
test "a DSML value holding its closer and then a call in another form: the parse keeps that call on a tie, and no stream completes a call the parse does not return" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var bs: ByteServer = .{};
    try bs.init(false);
    defer bs.deinit();
    for (spellings, 0..) |sp, si| {
        const value = try std.mem.concat(a, u8, &.{ "x", sp.close, "\n<tool_call>{\"name\":\"Read\",\"arguments\":{\"file_path\":\"/etc/hosts\"}}</tool_call>\n" });
        const answer = try spWrite(a, sp, "Write", value);
        for ([_]bool{ false, true }) |single| {
            const whole = try bs.chat(a, answer, false, single, &.{5});
            try std.testing.expectEqual(@as(usize, 1), whole.calls);
            try std.testing.expectEqualStrings("Read", whole.names[0]);
            try std.testing.expectEqualStrings("tool_calls", whole.finish);
            for ([_][]const usize{ &.{1}, &.{5}, &.{ 3, 4 }, &.{answer.len} }) |p| {
                const r = try bs.chat(a, answer, true, single, p);
                try std.testing.expect(r.done);
                try std.testing.expectEqual(@as(usize, 0), r.errors);
                try std.testing.expectEqual(@as(usize, 1), r.calls);
                try expectCompleteCallsParsed(a, r.names, r.args, whole.names, whole.args);
                if (single or si == 2) {
                    try std.testing.expectEqualStrings("tool_calls", r.finish);
                    try std.testing.expectEqualStrings("Read", r.names[0]);
                    try std.testing.expectEqualStrings(whole.args[0], r.args[0]);
                } else {
                    try std.testing.expectEqualStrings("stop", r.finish);
                    try std.testing.expectEqualStrings("Write", r.names[0]);
                    try std.testing.expect((try json.parseText(a, r.args[0])) != .ok);
                }
            }
        }
    }
}

/// Every call a client can read whole (its arguments JSON) is one of the whole reply's calls, in order.
fn expectCompleteCallsParsed(a: Allocator, names: []const []const u8, args: []const []const u8, whole_names: []const []const u8, whole_args: []const []const u8) !void {
    var at: usize = 0;
    for (names, args) |n, g| {
        const got = switch (try json.parseText(a, g)) {
            .ok => |v| v,
            .err => continue,
        };
        while (at < whole_names.len) : (at += 1) {
            const want = switch (try json.parseText(a, whole_args[at])) {
                .ok => |v| v,
                .err => continue,
            };
            if (std.mem.eql(u8, n, whole_names[at]) and json.equal(got, want)) break;
        } else {
            std.debug.print("a complete streamed call the whole reply does not have: {s} {s}\n", .{ n, g });
            return error.CompleteCallNotParsed;
        }
        at += 1;
    }
}

// The tie above over HTTP, on each streaming route, a byte a round and the whole reply in one: no route gives its client a
// complete call the whole reply does not have; the Write the stream began stays incomplete (chat completions: arguments
// that are not JSON; Messages: input that is not JSON; Responses: a function_call item that is not completed).
test "over HTTP, the tie: no route's stream completes a call the whole reply does not return" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var bs: ByteServer = .{};
    try bs.init(false);
    defer bs.deinit();
    const schema = "{\"type\":\"object\",\"properties\":{\"file_path\":{\"type\":\"string\"},\"content\":{\"type\":\"string\"}}}";
    const user = "\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],";
    const chat_body = "{\"model\":\"test-model\",\"max_tokens\":4096," ++ user ++ "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"Read\",\"parameters\":" ++ schema ++ "}},{\"type\":\"function\",\"function\":{\"name\":\"Write\",\"parameters\":" ++ schema ++ "}}]}";
    const msg_body = "{\"model\":\"test-model\",\"max_tokens\":4096," ++ user ++ "\"tools\":[{\"name\":\"Read\",\"input_schema\":" ++ schema ++ "},{\"name\":\"Write\",\"input_schema\":" ++ schema ++ "}]}";
    const resp_body = "{\"model\":\"test-model\",\"max_output_tokens\":4096,\"input\":\"x\",\"tools\":[{\"type\":\"function\",\"name\":\"Read\",\"parameters\":" ++ schema ++ "},{\"type\":\"function\",\"name\":\"Write\",\"parameters\":" ++ schema ++ "}]}";
    for (spellings[0..2]) |sp| {
        const value = try std.mem.concat(a, u8, &.{ "x", sp.close, "\n<tool_call>{\"name\":\"Read\",\"arguments\":{\"file_path\":\"/etc/hosts\"}}</tool_call>\n" });
        const answer = try std.mem.concat(a, u8, &.{ "Writing.", try spWrite(a, sp, "Write", value) });
        const ids = try a.alloc(u32, answer.len);
        for (answer, ids) |c, *id| id.* = c;
        bs.backend.ids = ids;
        for ([_][]const u8{ "/v1/chat/completions", "/v1/messages", "/v1/responses" }, [_][]const u8{ chat_body, msg_body, resp_body }) |path, body| {
            bs.backend.pattern = &.{answer.len};
            const want = try wireReply(a, try wirePost(a, bs.srv, path, try withStream(a, body, false)), path, false);
            try std.testing.expectEqual(@as(usize, 1), want.names.len);
            try std.testing.expectEqualStrings("Read", want.names[0]);
            for ([_][]const usize{ &.{1}, &.{answer.len} }) |pattern| {
                bs.backend.pattern = pattern;
                const wire = try wirePost(a, bs.srv, path, try withStream(a, body, true));
                const got = try wireReply(a, wire, path, true);
                try expectCompleteCallsParsed(a, got.names, got.args, want.names, want.args);
                try std.testing.expectEqual(@as(usize, 1), got.names.len);
                try std.testing.expectEqualStrings("Write", got.names[0]);
                try std.testing.expect((try json.parseText(a, got.args[0])) != .ok);
                if (std.mem.eql(u8, path, "/v1/responses")) {
                    for ((try finalResponse(a, wire)).get("output").?.array) |item| {
                        if (std.mem.eql(u8, item.strField("type").?, "function_call")) try std.testing.expect(!std.mem.eql(u8, item.strField("status").?, "completed"));
                    }
                }
            }
        }
    }
}

/// What a client reads from a route's reply on the wire: the calls' names and arguments (Messages' input as JSON) and the
/// text. Chat completions and Messages streams are read event by event, a Responses stream by its response.completed
/// event (the whole reply).
fn wireReply(a: Allocator, wire: []const u8, path: []const u8, stream: bool) !struct { names: []const []const u8, args: []const []const u8, text: []const u8 } {
    const body = wire[(std.mem.indexOf(u8, wire, "\r\n\r\n") orelse return error.NoBody) + 4 ..];
    const chat_route = std.mem.eql(u8, path, "/v1/chat/completions");
    const responses = std.mem.eql(u8, path, "/v1/responses");
    var names: std.ArrayList([]const u8) = .empty;
    var args: std.ArrayList(std.ArrayList(u8)) = .empty;
    var text: std.ArrayList(u8) = .empty;
    var wholes: std.ArrayList(json.Value) = .empty;
    if (!stream) try wholes.append(a, (try json.parse(a, body)).ok) else {
        var frames = std.mem.splitSequence(u8, body, "\n\n");
        while (frames.next()) |frame| {
            const at = std.mem.indexOf(u8, frame, "data: ") orelse continue;
            if (std.mem.startsWith(u8, frame[at + 6 ..], "[DONE]")) continue;
            const ev = (try json.parse(a, frame[at + 6 ..])).ok;
            if (chat_route) {
                const cs = ev.get("choices") orelse continue;
                if (cs.array.len == 0) continue;
                const d = cs.array[0].get("delta") orelse continue;
                try text.appendSlice(a, d.strField("content") orelse "");
                if (d.get("tool_calls")) |t| for (t.array) |one| {
                    const function = one.get("function").?;
                    if (one.get("id") != null) {
                        try names.append(a, function.strField("name").?);
                        try args.append(a, .empty);
                    }
                    try args.items[@intCast(one.get("index").?.int64().?)].appendSlice(a, function.strField("arguments") orelse "");
                };
            } else if (responses) {
                const kind = ev.strField("type") orelse "";
                if (std.mem.eql(u8, kind, "response.completed") or std.mem.eql(u8, kind, "response.incomplete")) try wholes.append(a, ev.get("response").?);
            } else {
                const kind = ev.strField("type") orelse continue;
                if (std.mem.eql(u8, kind, "content_block_start")) {
                    const cb = ev.get("content_block").?;
                    if (!std.mem.eql(u8, cb.strField("type").?, "tool_use")) continue;
                    try names.append(a, cb.strField("name").?);
                    try args.append(a, .empty);
                } else if (std.mem.eql(u8, kind, "content_block_delta")) {
                    const d = ev.get("delta").?;
                    if (std.mem.eql(u8, d.strField("type").?, "input_json_delta")) try args.items[args.items.len - 1].appendSlice(a, d.strField("partial_json").?);
                    if (std.mem.eql(u8, d.strField("type").?, "text_delta")) try text.appendSlice(a, d.strField("text").?);
                }
            }
        }
    }
    for (wholes.items) |v| {
        if (chat_route) {
            const m = v.get("choices").?.array[0].get("message").?;
            try text.appendSlice(a, m.strField("content") orelse "");
            if (m.get("tool_calls")) |t| for (t.array) |one| {
                try names.append(a, one.get("function").?.strField("name").?);
                try args.append(a, .empty);
                try args.items[args.items.len - 1].appendSlice(a, one.get("function").?.strField("arguments").?);
            };
        } else if (responses) {
            for (v.get("output").?.array) |item| {
                const kind = item.strField("type").?;
                if (std.mem.eql(u8, kind, "function_call")) {
                    try names.append(a, item.strField("name").?);
                    try args.append(a, .empty);
                    try args.items[args.items.len - 1].appendSlice(a, item.strField("arguments").?);
                } else if (std.mem.eql(u8, kind, "message")) {
                    for (item.get("content").?.array) |part| try text.appendSlice(a, part.strField("text") orelse "");
                }
            }
        } else {
            for (v.get("content").?.array) |block| {
                const kind = block.strField("type").?;
                if (std.mem.eql(u8, kind, "tool_use")) {
                    try names.append(a, block.strField("name").?);
                    try args.append(a, .empty);
                    try args.items[args.items.len - 1].appendSlice(a, try json.stringify(a, block.get("input").?, .{}));
                } else if (std.mem.eql(u8, kind, "text")) try text.appendSlice(a, block.strField("text").?);
            }
        }
    }
    const joined = try a.alloc([]const u8, args.items.len);
    for (args.items, joined) |arg, *slot| slot.* = arg.items;
    return .{ .names = names.items, .args = joined, .text = text.items };
}

test "a DSML value holding its block's own closer over HTTP through the chat, Responses and Messages routes: the call with its whole value, whole and streamed" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var bs: ByteServer = .{};
    try bs.init(false);
    defer bs.deinit();
    const schema = "{\"type\":\"object\",\"properties\":{\"file_path\":{\"type\":\"string\"},\"content\":{\"type\":\"string\"}}}";
    for ([_]Spelling{ spellings[0], spellings[2] }) |sp| {
        const value = try std.mem.concat(a, u8, &.{ "const end = \"", sp.close, "\";" });
        const answer = try std.mem.concat(a, u8, &.{ "Writing it.", try spWrite(a, sp, "write", value) });
        for ([_]bool{ false, true }) |stream| {
            const st = if (stream) "true" else "false";
            const routes = [_][2][]const u8{
                .{ "/v1/chat/completions", try std.mem.concat(a, u8, &.{ "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"max_tokens\":4096,\"stream\":", st, ",\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"write\",\"parameters\":", schema, "}}]}" }) },
                .{ "/v1/responses", try std.mem.concat(a, u8, &.{ "{\"model\":\"test-model\",\"input\":\"x\",\"max_output_tokens\":4096,\"stream\":", st, ",\"tools\":[{\"type\":\"function\",\"name\":\"write\",\"parameters\":", schema, "}]}" }) },
                .{ "/v1/messages", try std.mem.concat(a, u8, &.{ "{\"model\":\"test-model\",\"max_tokens\":4096,\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"stream\":", st, ",\"tools\":[{\"name\":\"write\",\"input_schema\":", schema, "}]}" }) },
            };
            for (routes) |r| {
                const ids = try a.alloc(u32, answer.len);
                for (answer, ids) |c, *id| id.* = c;
                bs.backend.ids = ids;
                bs.backend.pattern = &.{3};
                const wire = try wirePost(a, bs.srv, r[0], r[1]);
                try std.testing.expect(std.mem.startsWith(u8, wire, "HTTP/1.1 200 "));
                const got = try wireReply(a, wire, r[0], stream);
                std.testing.expectEqual(@as(usize, 1), got.names.len) catch |e| {
                    std.debug.print("{s} stream={}: {s}\n", .{ r[0], stream, wire[0..@min(wire.len, 600)] });
                    return e;
                };
                try std.testing.expectEqualStrings("write", got.names[0]);
                try std.testing.expectEqualStrings(value, (try json.parse(a, got.args[0])).ok.strField("content").?);
                try std.testing.expectEqualStrings("Writing it.", std.mem.trim(u8, got.text, " \n"));
            }
        }
    }
}

// Random DeepSeek-V4.1 replies (dsml_fuzz_test's: well-formed, odd markup, closers inside values, cut anywhere, and the
// plain closers of a file about DSML), with and without the one-call policy, through the chat route: whole, and streamed
// in random rounds of 1 to 12 bytes. A well-formed reply with unique parameters whose values hold at most the block's own
// closer: the stream is the whole reply, its calls byte for byte, its content and its finish. Every other reply: each
// call the stream got whole is one of the reply's, in order (the same bytes, or the same JSON with a repeated key's last
// value kept); with no call cut short the stream has them all and, with well-formed markup, the content; a call cut
// short (patch 0048's, when the parse reads odd markup or an invoke's closer inside a value otherwise than the stream
// did) leaves a stream with no whole call the reply's own finish reason, never tool_calls, and its content is the
// reply's with at most some of it left out.
test "random DSML replies through the chat route: a stream gets the whole reply's calls and content, at random rounds" {
    const fuzz = @import("dsml_fuzz_test.zig");
    var bs: ByteServer = .{};
    try bs.init(false);
    defer bs.deinit();
    var prng = std.Random.DefaultPrng.init(14);
    const r = prng.random();
    const modes = [_]fuzz.Mode{ .{}, .{ .malformed = true }, .{ .closers = true }, .{ .closers = true, .malformed = true, .cut = true }, .{ .closers = true, .plain = true } };
    var with_calls: usize = 0;
    var plain_closers: usize = 0;
    for (0..2500) |i| {
        var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const mode = modes[i % modes.len];
        var g: fuzz.Gen = .{ .r = r, .a = a, .m = mode };
        const text = try g.reply();
        const single = r.uintLessThan(u8, 4) == 0;
        var pattern: [8]usize = undefined;
        for (&pattern) |*p| p.* = 1 + r.uintLessThan(usize, 12);
        const whole = try bs.chat(a, text, false, single, &.{5});
        const s = try bs.chat(a, text, true, single, &pattern);
        errdefer std.debug.print("reply {d} (single {}): {f}\n  whole {s} {d} calls, content {f}\n  stream {s} {d} calls, content {f}\n", .{ i, single, std.zig.fmtString(text), whole.finish, whole.calls, std.zig.fmtString(whole.content), s.finish, s.calls, std.zig.fmtString(s.content) });
        try std.testing.expect(s.done);
        try std.testing.expectEqual(@as(usize, 0), s.errors);
        with_calls += @intFromBool(whole.calls > 0);
        if ((mode.plain or (!mode.closers and !mode.malformed and !mode.cut)) and !g.repeated) {
            try std.testing.expectEqualStrings(whole.finish, s.finish);
            try std.testing.expectEqual(whole.calls, s.calls);
            for (whole.names, s.names, whole.args, s.args) |wn, sn, wa, sa| {
                try std.testing.expectEqualStrings(wn, sn);
                try std.testing.expectEqualStrings(wa, sa);
            }
            // under the one-call policy the parse reads a later block, cut at a closer in its value, as content past
            // its first call (the two readings tie at one call); the stream hides that block whole
            const said = try withoutBlanks(a, s.content);
            const full = try withoutBlanks(a, whole.content);
            if (!single) try std.testing.expectEqualStrings(full, said) else {
                var k: usize = 0;
                for (said) |c| k = (std.mem.indexOfScalarPos(u8, full, k, c) orelse return error.StreamSaidMoreThanTheReply) + 1;
            }
            plain_closers += @intFromBool(mode.plain and g.closer_in_value and whole.calls > 0 and !single);
            continue;
        }
        var cut_calls: usize = 0;
        var whole_calls: usize = 0;
        for (s.args) |sa| {
            if ((try json.parseText(a, sa)) == .ok) whole_calls += 1 else cut_calls += 1;
        }
        if (cut_calls == 0) try std.testing.expectEqualStrings(whole.finish, s.finish) else if (whole_calls == 0) try std.testing.expect(!std.mem.eql(u8, s.finish, "tool_calls"));
        // every call the client got whole is the reply's, in order
        var at: usize = 0;
        for (s.names, s.args) |sn, sa| {
            if ((try json.parseText(a, sa)) != .ok) continue;
            while (at < whole.calls) : (at += 1) {
                if (std.mem.eql(u8, whole.names[at], sn) and try fuzz.sameArguments(a, whole.args[at], sa)) break;
            } else return error.WholeCallNotParsed;
            at += 1;
        }
        // a repeated parameter changes the bytes of a call that went out, and the finish sends no call after a changed one:
        // the parse's later calls that had not streamed are left out (EXPOSURE.md, a repeated parameter)
        if (cut_calls == 0 and !g.repeated) try std.testing.expectEqual(whole.calls, s.calls);
        // the same content when the markup is well formed (odd markup the stream hides as a block, as it decodes, the
        // parse leaves in the content; under the one-call policy the parse reads a later block, cut at a closer in its
        // value, as content): the stream's is then the reply's with some of it left out, never more
        const said = try withoutBlanks(a, s.content);
        const full = try withoutBlanks(a, whole.content);
        if (!mode.malformed and cut_calls == 0 and !(single and mode.closers)) try std.testing.expectEqualStrings(full, said) else {
            var k: usize = 0;
            for (said) |c| k = (std.mem.indexOfScalarPos(u8, full, k, c) orelse return error.StreamSaidMoreThanTheReply) + 1;
        }
    }
    try std.testing.expect(with_calls > 600);
    errdefer std.debug.print("{d} plain replies with a closer in a value\n", .{plain_closers});
    try std.testing.expect(plain_closers > 60);
}

// Thinking off, the finish reads Harmony channels out of the text, and a stop string's start held a <tool_call> back
// while the reply decoded. The whole reply's content is the final channel (none here), so it has no call; the finish's
// streamer does not read the text outside the channels either, and sends none.
test "thinking off, a call only the text before the Harmony split holds, its opener held back by a stop string's start: no call in either" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const answer = "<|channel|>analysis<|message|><tool_call><function=Write></function>";
    const stop = "<tool_call><function=Write></function>X";
    const whole = wholeMessage(&try bytesAsk(a, answer, .{ .eos = true, .stop = stop, .stream = false }));
    try std.testing.expectEqual(@as(usize, 0), whole.calls.len);
    for ([_][]const usize{ &.{1}, &.{5}, &.{1 << 20} }) |p| {
        const r = try Read.of(a, &try bytesAsk(a, answer, .{ .eos = true, .stop = stop, .pattern = p }));
        try std.testing.expectEqualStrings(whole.finish, r.finish.?);
        // what streamed before the Harmony marker was whole stays (as before this patch); no call
        try std.testing.expect(std.mem.startsWith(u8, "<|channel|>", r.content.items));
        try std.testing.expectEqual(@as(usize, 0), r.calls.items.len);
    }
}

// A block in another form that is no call: the stream filter hides its markup as it decodes, and the text after it still
// goes out; the whole reply keeps the markup as content (no call limit).
test "leading whitespace, a block that is no call, then text: the stream shows the text around the block, the whole reply the block as well" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const answer = " hello<tool_call>bad</tool_call>tail";
    const whole = wholeMessage(&try bytesAsk(a, answer, .{ .stream = false }));
    try std.testing.expectEqual(@as(usize, 0), whole.calls.len);
    try std.testing.expectEqualStrings(answer, whole.content);
    for ([_][]const usize{ &.{1}, &.{5}, &.{1 << 20} }) |p| {
        const r = try Read.of(a, &try bytesAsk(a, answer, .{ .pattern = p }));
        try std.testing.expectEqualStrings(" hellotail", r.content.items);
        try std.testing.expectEqual(@as(usize, 0), r.calls.items.len);
    }
}

/// `said` is `full` with at most some of it left out: the stream filter hides a cut block's markup that the whole reply,
/// with no call to parse out of it, keeps as content.
fn withinText(a: Allocator, said: []const u8, full: []const u8) !void {
    const s = try withoutBlanks(a, said);
    const f = try withoutBlanks(a, full);
    var k: usize = 0;
    for (s) |c| k = (std.mem.indexOfScalarPos(u8, f, k, c) orelse return error.StreamSaidMoreThanTheReply) + 1;
}

/// The longest run of keep-alive pings between two other events of `seen`.
fn longestPingRun(seen: *const Seen) usize {
    var run: usize = 0;
    var best: usize = 0;
    for (seen.events.items) |e| {
        if (std.mem.eql(u8, e, "{\"ping\":true}")) {
            run += 1;
            best = @max(best, run);
        } else run = 0;
    }
    return best;
}

// The stream's keep-alive pings after its bound of silence. A value that holds its block's closer streams up to the
// closer, then waits until the text says where the block ends; the rounds that come in meanwhile send nothing. With the
// bound at zero each of those rounds gets a ping, in one unbroken run as long as the text held back, and the call is
// still whole; with the bound as served, a reply this short gets none.
test "the keep-alive pings while a value waits past a closer inside it, and leaves the call as it was" {
    const chat = @import("chat.zig");
    const saved = chat.ping_after_ns;
    defer chat.ping_after_ns = saved;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tail: [300]u8 = @splat('x');
    const value = "end: " ++ b_close ++ &tail;
    const answer = comptime ("Writing." ++ b_open ++ bInvoke("Write", bParam("file_path", "/tmp/dsml.md") ++ bParam("content", value)) ++ b_close);
    const quiet = try bytesAsk(a, answer, .{ .pattern = &.{1} });
    try std.testing.expectEqual(@as(usize, 0), quiet.pings);
    chat.ping_after_ns = 0;
    const pinged = try bytesAsk(a, answer, .{ .pattern = &.{1} });
    try std.testing.expect(longestPingRun(&pinged) >= tail.len);
    const x = try Read.of(a, &quiet);
    const y = try Read.of(a, &pinged);
    try std.testing.expectEqualStrings("tool_calls", y.finish.?);
    try std.testing.expectEqualStrings(x.content.items, y.content.items);
    try std.testing.expectEqual(@as(usize, 1), y.calls.items.len);
    try std.testing.expectEqualStrings(x.calls.items[0].arguments.items, y.calls.items[0].arguments.items);
    try std.testing.expectEqualStrings(value, (try json.parse(a, y.calls.items[0].arguments.items)).ok.strField("content").?);
    // without the closer in the value, no round waits: the pings never run as long
    const plain = comptime ("Writing." ++ b_open ++ bInvoke("Write", bParam("file_path", "/tmp/dsml.md") ++ bParam("content", "end: " ++ &tail)) ++ b_close);
    try std.testing.expect(longestPingRun(&try bytesAsk(a, plain, .{ .pattern = &.{1} })) < tail.len / 2);
}

// The stream filter hides DSML blocks, <tool_call> and <|tool_call> (reply_text.calls); a namespaced <x:tool_call> is
// prose to it, though the parse reads it as a call, and to the DSML streamer it is another call form before the block:
// the block's call goes out when the reply ends, with the other, as the whole reply has them.
test "a call form the stream filter does not hide, before a DSML block: its markup streamed as prose, the text after the block still goes out, and the calls are the whole reply's" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const x_call = "<x:tool_call>{\"name\":\"Read\",\"arguments\":{\"file_path\":\"/etc/hosts\"}}</x:tool_call>";
    const answer = comptime (x_call ++ " then" ++ b_open ++ bInvoke("Bash", bParam("command", "ls")) ++ b_close ++ "\nTAIL");
    const whole = wholeMessage(&try bytesAsk(a, answer, .{ .stream = false }));
    try std.testing.expectEqual(@as(usize, 2), whole.calls.len);
    try std.testing.expectEqualStrings("then\n\n\nTAIL", whole.content);
    for ([_][]const usize{ &.{1}, &.{5} }) |p| {
        const r = try Read.of(a, &try bytesAsk(a, answer, .{ .pattern = p }));
        try std.testing.expectEqualStrings(whole.finish, r.finish.?);
        try std.testing.expectEqualStrings(x_call ++ " then\nTAIL", r.content.items);
        try std.testing.expectEqual(whole.calls.len, r.calls.items.len);
        for (whole.calls, r.calls.items) |w, c| {
            try std.testing.expectEqualStrings(w.get("function").?.strField("name").?, c.name);
            try std.testing.expectEqualStrings(w.get("function").?.strField("arguments").?, c.arguments.items);
        }
    }
}

/// A Messages or Responses stream's events in the order a client reads them: blocks (items) opened one at a time with
/// the next index, each delta on the open one, each closed before the next opens, none after the stream's end; and the
/// kinds of block in order (text or message, then tool_use or function_call).
fn streamOrder(a: Allocator, wire: []const u8, messages: bool) ![]const []const u8 {
    const body = wire[(std.mem.indexOf(u8, wire, "\r\n\r\n") orelse return error.NoBody) + 4 ..];
    var kinds: std.ArrayList([]const u8) = .empty;
    var open: ?i64 = null;
    var ended = false;
    var frames = std.mem.splitSequence(u8, body, "\n\n");
    while (frames.next()) |frame| {
        const at = std.mem.indexOf(u8, frame, "data: ") orelse continue;
        const ev = (try json.parse(a, frame[at + 6 ..])).ok;
        const kind = ev.strField("type") orelse continue;
        if (std.mem.eql(u8, kind, "ping")) continue;
        try std.testing.expect(!ended);
        const index = (if (messages) ev.get("index") else ev.get("output_index")) orelse {
            ended = std.mem.eql(u8, kind, if (messages) "message_stop" else "response.completed") or std.mem.eql(u8, kind, "response.incomplete");
            if (messages and std.mem.eql(u8, kind, "message_delta")) try std.testing.expect(open == null);
            continue;
        };
        const i = index.int64().?;
        if (std.mem.eql(u8, kind, if (messages) "content_block_start" else "response.output_item.added")) {
            try std.testing.expect(open == null);
            try std.testing.expectEqual(@as(i64, @intCast(kinds.items.len)), i);
            open = i;
            const block = (if (messages) ev.get("content_block") else ev.get("item")).?;
            try kinds.append(a, block.strField("type").?);
        } else if (std.mem.eql(u8, kind, if (messages) "content_block_stop" else "response.output_item.done")) {
            try std.testing.expectEqual(open, i);
            open = null;
        } else try std.testing.expectEqual(open, i);
    }
    try std.testing.expect(ended and open == null);
    return kinds.items;
}

/// The response a Responses stream ends with (response.completed or response.incomplete).
fn finalResponse(a: Allocator, wire: []const u8) !json.Value {
    const body = wire[(std.mem.indexOf(u8, wire, "\r\n\r\n") orelse return error.NoBody) + 4 ..];
    var frames = std.mem.splitSequence(u8, body, "\n\n");
    while (frames.next()) |frame| {
        const at = std.mem.indexOf(u8, frame, "data: ") orelse continue;
        const ev = (try json.parse(a, frame[at + 6 ..])).ok;
        const kind = ev.strField("type") orelse continue;
        if (std.mem.eql(u8, kind, "response.completed") or std.mem.eql(u8, kind, "response.incomplete")) return ev.get("response").?;
    }
    return error.NoFinalResponse;
}

/// The finish a stream on the wire reports: chat completions' last finish_reason, Messages' stop_reason, Responses' status.
fn wireFinish(a: Allocator, wire: []const u8, path: []const u8) ![]const u8 {
    const body = wire[(std.mem.indexOf(u8, wire, "\r\n\r\n") orelse return error.NoBody) + 4 ..];
    var finish: []const u8 = "";
    var frames = std.mem.splitSequence(u8, body, "\n\n");
    while (frames.next()) |frame| {
        const at = std.mem.indexOf(u8, frame, "data: ") orelse continue;
        if (std.mem.startsWith(u8, frame[at + 6 ..], "[DONE]")) continue;
        const ev = (try json.parse(a, frame[at + 6 ..])).ok;
        if (std.mem.eql(u8, path, "/v1/chat/completions")) {
            const cs = ev.get("choices") orelse continue;
            if (cs.array.len == 0) continue;
            if (cs.array[0].strField("finish_reason")) |f| finish = f;
        } else if (std.mem.eql(u8, path, "/v1/responses")) {
            if (ev.get("response")) |r| if (r.strField("status")) |st| {
                finish = st;
            };
        } else if (std.mem.eql(u8, ev.strField("type") orelse "", "message_delta")) {
            if (ev.get("delta").?.strField("stop_reason")) |st| finish = st;
        }
    }
    return finish;
}

// A value holding its block's closer streams through each route as a well-formed call does: Messages and Responses open
// the text, then the call, each closed before the next. The same reply cut by max_tokens right after the closer in the
// value: the call's header went out before the cut, as patch 0048 sends it, and the whole reply has no call. Each route
// then ends truthfully: chat completions with finish_reason length and the call's arguments left unclosed (not JSON),
// Messages with the tool_use block closed and stop_reason max_tokens, Responses with the function_call item closed as
// incomplete in an incomplete response; the text each got is the whole reply's.
test "over HTTP, a value holding its closer: events in order and the whole call; cut by max_tokens after the closer, the header went out and each route ends as cut" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var bs: ByteServer = .{};
    try bs.init(false);
    defer bs.deinit();
    const schema = "{\"type\":\"object\",\"properties\":{\"file_path\":{\"type\":\"string\"},\"content\":{\"type\":\"string\"}}}";
    const value = "end: </" ++ BD ++ " calls>";
    const answer = try std.mem.concat(a, u8, &.{ "Writing it.", try spWrite(a, spellings[0], "write", value) });
    const chat_body = try std.mem.concat(a, u8, &.{ "\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"stream\":true,\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"write\",\"parameters\":", schema, "}}]}" });
    const messages = try std.mem.concat(a, u8, &.{ "\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"stream\":true,\"tools\":[{\"name\":\"write\",\"input_schema\":", schema, "}]}" });
    const responses = try std.mem.concat(a, u8, &.{ "\"input\":\"x\",\"stream\":true,\"tools\":[{\"type\":\"function\",\"name\":\"write\",\"parameters\":", schema, "}]}" });
    for ([_][]const usize{ &.{1}, &.{3}, &.{answer.len} }) |p| {
        const ids = try a.alloc(u32, answer.len);
        for (answer, ids) |c, *id| id.* = c;
        bs.backend.ids = ids;
        bs.backend.pattern = p;
        const m = try wirePost(a, bs.srv, "/v1/messages", try std.mem.concat(a, u8, &.{ "{\"model\":\"test-model\",\"max_tokens\":4096,", messages }));
        const mk = try streamOrder(a, m, true);
        try std.testing.expectEqual(@as(usize, 2), mk.len);
        try std.testing.expectEqualStrings("text", mk[0]);
        try std.testing.expectEqualStrings("tool_use", mk[1]);
        try std.testing.expectEqualStrings("tool_use", try wireFinish(a, m, "/v1/messages"));
        const mg = try wireReply(a, m, "/v1/messages", true);
        try std.testing.expectEqualStrings(value, (try json.parse(a, mg.args[0])).ok.strField("content").?);
        const rw = try wirePost(a, bs.srv, "/v1/responses", try std.mem.concat(a, u8, &.{ "{\"model\":\"test-model\",\"max_output_tokens\":4096,", responses }));
        const rk = try streamOrder(a, rw, false);
        try std.testing.expectEqual(@as(usize, 2), rk.len);
        try std.testing.expectEqualStrings("message", rk[0]);
        try std.testing.expectEqualStrings("function_call", rk[1]);
        try std.testing.expectEqualStrings("completed", (try finalResponse(a, rw)).get("output").?.array[1].strField("status").?);
    }
    // cut by max_tokens 20 bytes before the block's end: after the closer in the value, the header out
    const cut = answer[0 .. answer.len - 20];
    const ids = try a.alloc(u32, cut.len);
    for (cut, ids) |c, *id| id.* = c;
    bs.backend.ids = ids;
    bs.backend.pattern = &.{3};
    bs.backend.cut = true;
    defer bs.backend.cut = false;
    const max = try std.fmt.allocPrint(a, "{{\"model\":\"test-model\",\"max_tokens\":{d},", .{cut.len});
    const whole_body = try std.mem.concat(a, u8, &.{ max, messages[0 .. messages.len - 1], "}" });
    const w = try wirePost(a, bs.srv, "/v1/messages", try std.mem.replaceOwned(u8, a, whole_body, "\"stream\":true", "\"stream\":false"));
    const wv = (try json.parse(a, w[(std.mem.indexOf(u8, w, "\r\n\r\n") orelse return error.NoBody) + 4 ..])).ok;
    try std.testing.expectEqualStrings("max_tokens", wv.strField("stop_reason").?);
    const want = try wireReply(a, w, "/v1/messages", false);
    try std.testing.expectEqual(@as(usize, 0), want.names.len);
    // Messages: the tool_use block opened by the header closes, then the rest of the whole reply's text, max_tokens
    const s = try wirePost(a, bs.srv, "/v1/messages", try std.mem.concat(a, u8, &.{ max, messages }));
    const sk = try streamOrder(a, s, true);
    try std.testing.expect(sk.len >= 2);
    try std.testing.expectEqualStrings("text", sk[0]);
    try std.testing.expectEqualStrings("tool_use", sk[1]);
    for (sk[2..]) |k| try std.testing.expectEqualStrings("text", k);
    try std.testing.expectEqualStrings("max_tokens", try wireFinish(a, s, "/v1/messages"));
    const got = try wireReply(a, s, "/v1/messages", true);
    try std.testing.expectEqual(@as(usize, 1), got.names.len);
    try std.testing.expectEqualStrings("write", got.names[0]);
    try std.testing.expect((try json.parseText(a, got.args[0])) != .ok);
    // the whole reply, with no call to parse out of the cut block, keeps the block as text; the stream hid it as it
    // decoded and shows what follows its first closer (as patch 0048 did)
    try std.testing.expect(std.mem.indexOf(u8, want.text, "<" ++ BD ++ " calls>") != null);
    try std.testing.expect(std.mem.startsWith(u8, got.text, "Writing it.") and std.mem.indexOf(u8, got.text, "<" ++ BD ++ " calls>") == null);
    try withinText(a, got.text, want.text);
    // chat completions: the call's arguments stay unclosed, finish_reason length
    const c = try wirePost(a, bs.srv, "/v1/chat/completions", try std.mem.concat(a, u8, &.{ max, chat_body }));
    const cg = try wireReply(a, c, "/v1/chat/completions", true);
    try std.testing.expectEqual(@as(usize, 1), cg.names.len);
    try std.testing.expect((try json.parseText(a, cg.args[0])) != .ok);
    try std.testing.expectEqualStrings("length", try wireFinish(a, c, "/v1/chat/completions"));
    try std.testing.expectEqualStrings(got.text, cg.text);
    // Responses: the function_call item closes as incomplete, in an incomplete response
    const rmax = try std.fmt.allocPrint(a, "{{\"model\":\"test-model\",\"max_output_tokens\":{d},", .{cut.len});
    const rs = try wirePost(a, bs.srv, "/v1/responses", try std.mem.concat(a, u8, &.{ rmax, responses }));
    const rsk = try streamOrder(a, rs, false);
    try std.testing.expect(rsk.len >= 2);
    try std.testing.expectEqualStrings("function_call", rsk[1]);
    for (rsk[2..]) |k| try std.testing.expectEqualStrings("message", k);
    const final = try finalResponse(a, rs);
    try std.testing.expectEqualStrings("incomplete", final.strField("status").?);
    const item = final.get("output").?.array[1];
    try std.testing.expectEqualStrings("incomplete", item.strField("status").?);
    try std.testing.expect((try json.parseText(a, item.strField("arguments").?)) != .ok);
}

// With the silence bound at zero, the rounds a value waits past a closer inside it get pings on the wire of each
// streaming route (an SSE comment on chat completions and Responses, a ping event on Messages), and the call still
// arrives whole.
test "keep-alive on the wire while a value waits past a closer inside it: chat completions, Responses and Messages each get pings" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;
    const chat = @import("chat.zig");
    const saved = chat.ping_after_ns;
    defer chat.ping_after_ns = saved;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var bs: ByteServer = .{};
    try bs.init(false);
    defer bs.deinit();
    const schema = "{\"type\":\"object\",\"properties\":{\"file_path\":{\"type\":\"string\"},\"content\":{\"type\":\"string\"}}}";
    const ys: [200]u8 = @splat('y');
    const value = "end: </" ++ BD ++ " calls>" ++ &ys;
    const answer = try std.mem.concat(a, u8, &.{ "Writing it.", try spWrite(a, spellings[0], "write", value) });
    const routes = [_][3][]const u8{
        .{ "/v1/chat/completions", try std.mem.concat(a, u8, &.{ "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"max_tokens\":4096,\"stream\":true,\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"write\",\"parameters\":", schema, "}}]}" }), "\n: ping\n\n" },
        .{ "/v1/responses", try std.mem.concat(a, u8, &.{ "{\"model\":\"test-model\",\"input\":\"x\",\"max_output_tokens\":4096,\"stream\":true,\"tools\":[{\"type\":\"function\",\"name\":\"write\",\"parameters\":", schema, "}]}" }), "\n: ping\n\n" },
        .{ "/v1/messages", try std.mem.concat(a, u8, &.{ "{\"model\":\"test-model\",\"max_tokens\":4096,\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"stream\":true,\"tools\":[{\"name\":\"write\",\"input_schema\":", schema, "}]}" }), "event: ping" },
    };
    for (routes) |r| {
        const ids = try a.alloc(u32, answer.len);
        for (answer, ids) |c, *id| id.* = c;
        bs.backend.ids = ids;
        bs.backend.pattern = &.{1};
        chat.ping_after_ns = 0;
        const wire = try wirePost(a, bs.srv, r[0], r[1]);
        chat.ping_after_ns = saved;
        const pings = std.mem.count(u8, wire, r[2]);
        std.testing.expect(pings > 100) catch |e| {
            std.debug.print("{s}: {d} pings in {s}\n", .{ r[0], pings, wire[0..@min(wire.len, 400)] });
            return e;
        };
        const got = try wireReply(a, wire, r[0], true);
        try std.testing.expectEqual(@as(usize, 1), got.names.len);
        try std.testing.expectEqualStrings(value, (try json.parse(a, got.args[0])).ok.strField("content").?);
    }
}

// Under the one-call policy the parse's two readings tie at one call when a later block holds its closer inside a value:
// the reading cut at that closer wins, and the block's text past the closer stays in the whole reply's content. The
// stream hides the block whole as it decodes (reply_text.blockEnd reads past the closer), so its content is the whole
// reply's with that text left out; the one call is the same in both, the first block's.
test "one-call policy, a later block holding its closer inside a value: the same call, the stream's content without the block's tail that the parse keeps" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var bs: ByteServer = .{};
    try bs.init(false);
    defer bs.deinit();
    const first = comptime ("Listing." ++ b_open ++ bInvoke("Bash", bParam("command", "ls")) ++ b_close);
    const answer = try std.mem.concat(a, u8, &.{ first, "\nThen.", try spWrite(a, spellings[0], "Write", "end: " ++ b_close ++ " more"), "\nDone." });
    const whole = try bs.chat(a, answer, false, true, &.{5});
    try std.testing.expectEqual(@as(usize, 1), whole.calls);
    try std.testing.expectEqualStrings("Bash", whole.names[0]);
    try std.testing.expect(std.mem.indexOf(u8, whole.content, "</" ++ BD ++ " invoke>") != null);
    for ([_][]const usize{ &.{1}, &.{5}, &.{ 3, 4 }, &.{answer.len} }) |p| {
        const r = try bs.chat(a, answer, true, true, p);
        try std.testing.expectEqualStrings(whole.finish, r.finish);
        try std.testing.expectEqual(@as(usize, 1), r.calls);
        try std.testing.expectEqualStrings("Bash", r.names[0]);
        try std.testing.expectEqualStrings(whole.args[0], r.args[0]);
        try std.testing.expect(std.mem.indexOf(u8, r.content, "\u{ff5c}") == null);
        try std.testing.expectEqualStrings("Listing.Then.Done.", try withoutBlanks(a, r.content));
        try withinText(a, r.content, whole.content);
    }
}

test "thinking off, a closer in a value and then a Harmony marker, cut one byte at a time: the stream ends as the whole reply, no read past the text" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const answer = "<" ++ BD ++ " calls><" ++ BD ++ " invoke name=\"Write\"><" ++ BD ++ " parameter name=\"content\">x</" ++ BD ++ " calls>y<|channel|>";
    const whole = wholeMessage(&try bytesAsk(a, answer, .{ .cut = true, .stream = false }));
    try std.testing.expectEqual(@as(usize, 0), whole.calls.len);
    for ([_][]const usize{ &.{1}, &.{5}, &.{1 << 20} }) |p| {
        const r = try Read.of(a, &try bytesAsk(a, answer, .{ .cut = true, .pattern = p }));
        try std.testing.expectEqualStrings(whole.finish, r.finish.?);
        for (r.calls.items) |c| try std.testing.expect((try json.parse(a, c.arguments.items)) == .err);
    }
}

// A whole call, then a second block cut by max_tokens after its header went out: the client holds a call the parse
// drops, so each route ends as cut (chat length, Messages max_tokens, Responses incomplete), the first call whole. The
// whole reply (not streamed) returns the first call alone, as before this patch.
test "over HTTP, a whole call then a block cut by max_tokens before, at and after its header: each route's stream ends as cut, its text the whole reply's" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const schema = "{\"type\":\"object\",\"properties\":{\"file_path\":{\"type\":\"string\"},\"content\":{\"type\":\"string\"}}}";
    const prefix = comptime ("Doing it." ++ b_open ++ bInvoke("read", bParam("file_path", "/a")) ++ b_close ++ b_open);
    const cuts = comptime [_]struct { answer: []const u8, live_calls: usize }{
        .{ .answer = prefix, .live_calls = 1 },
        .{ .answer = prefix ++ "<" ++ BD ++ " invoke name=\"wri", .live_calls = 1 },
        .{ .answer = prefix ++ "<" ++ BD ++ " invoke name=\"write\">\n" ++ bParam("file_path", "/b") ++ "<" ++ BD ++ " parameter name=\"content\" string=\"true\">cut here and more", .live_calls = 2 },
    };
    const chat_tools = "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"read\",\"parameters\":" ++ schema ++ "}},{\"type\":\"function\",\"function\":{\"name\":\"write\",\"parameters\":" ++ schema ++ "}}]}";
    const msg_tools = "\"tools\":[{\"name\":\"read\",\"input_schema\":" ++ schema ++ "},{\"name\":\"write\",\"input_schema\":" ++ schema ++ "}]}";
    const resp_tools = "\"tools\":[{\"type\":\"function\",\"name\":\"read\",\"parameters\":" ++ schema ++ "},{\"type\":\"function\",\"name\":\"write\",\"parameters\":" ++ schema ++ "}]}";
    const user = "\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],";
    for ([_]tool_streaming.Mode{ .live, .hold }) |mode| {
        var bs: ByteServer = .{ .mode = mode };
        try bs.init(false);
        defer bs.deinit();
        bs.backend.cut = true;
        for (cuts) |cut| {
            const ids = try a.alloc(u32, cut.answer.len);
            for (cut.answer, ids) |c, *id| id.* = c;
            bs.backend.ids = ids;
            const max = try std.fmt.allocPrint(a, "{{\"model\":\"test-model\",\"max_tokens\":{d},", .{cut.answer.len});
            const rmax = try std.fmt.allocPrint(a, "{{\"model\":\"test-model\",\"max_output_tokens\":{d},\"input\":\"x\",", .{cut.answer.len});
            const chat_body = try std.mem.concat(a, u8, &.{ max, user, chat_tools });
            const msg_body = try std.mem.concat(a, u8, &.{ max, user, msg_tools });
            const resp_body = try std.mem.concat(a, u8, &.{ rmax, resp_tools });
            bs.backend.pattern = &.{cut.answer.len};
            const want = try wireReply(a, try wirePost(a, bs.srv, "/v1/chat/completions", try withStream(a, chat_body, false)), "/v1/chat/completions", false);
            const want_m = try wireReply(a, try wirePost(a, bs.srv, "/v1/messages", try withStream(a, msg_body, false)), "/v1/messages", false);
            const want_r = try wireReply(a, try wirePost(a, bs.srv, "/v1/responses", try withStream(a, resp_body, false)), "/v1/responses", false);
            try std.testing.expectEqual(@as(usize, 1), want.names.len);
            try std.testing.expectEqualStrings("read", want.names[0]);
            for ([_][]const usize{ &.{1}, &.{3}, &.{cut.answer.len} }) |p| {
                bs.backend.pattern = p;
                const c = try wirePost(a, bs.srv, "/v1/chat/completions", try withStream(a, chat_body, true));
                const m = try wirePost(a, bs.srv, "/v1/messages", try withStream(a, msg_body, true));
                const r = try wirePost(a, bs.srv, "/v1/responses", try withStream(a, resp_body, true));
                const cg = try wireReply(a, c, "/v1/chat/completions", true);
                // the text of each stream is the whole reply's text on that route: a cut block stays the content
                try std.testing.expectEqualStrings(want.text, cg.text);
                try std.testing.expectEqualStrings(want_m.text, (try wireReply(a, m, "/v1/messages", true)).text);
                try std.testing.expectEqualStrings(want_r.text, (try wireReply(a, r, "/v1/responses", true)).text);
                try std.testing.expectEqualStrings(want.args[0], cg.args[0]);
                if (mode == .live) {
                    try std.testing.expectEqualStrings("length", try wireFinish(a, c, "/v1/chat/completions"));
                    try std.testing.expectEqualStrings("max_tokens", try wireFinish(a, m, "/v1/messages"));
                    try std.testing.expectEqualStrings("incomplete", (try finalResponse(a, r)).strField("status").?);
                    try std.testing.expectEqual(cut.live_calls, cg.names.len);
                    if (cg.names.len > 1) try std.testing.expect((try json.parseText(a, cg.args[1])) != .ok);
                } else {
                    // hold sends the whole reply: its calls and its finish
                    try std.testing.expectEqual(@as(usize, 1), cg.names.len);
                    const whole = try wirePost(a, bs.srv, "/v1/chat/completions", try withStream(a, chat_body, false));
                    const doc = (try json.parse(a, whole[(std.mem.indexOf(u8, whole, "\r\n\r\n") orelse return error.NoBody) + 4 ..])).ok;
                    try std.testing.expectEqualStrings(doc.get("choices").?.array[0].strField("finish_reason").?, try wireFinish(a, c, "/v1/chat/completions"));
                }
            }
        }
    }
}

fn withStream(a: Allocator, body: []const u8, stream: bool) ![]const u8 {
    return std.mem.concat(a, u8, &.{ body[0 .. body.len - 1], if (stream) ",\"stream\":true}" else ",\"stream\":false}" });
}

const tool_parse = @import("tool_parse.zig");
const V4 = spellings[2];

/// Replies that end where a whole call ends, with another call form's markup or the block's own tags in a value, and
/// replies that end inside a call: each runs to max_tokens and again to the model's end.
const at_limit = [_][]const u8{
    b_open ++ bInvoke("Write", bParam("file_path", "/a") ++ bParam("content", "<tool_call>")) ++ b_close,
    b_open ++ bInvoke("Write", bParam("file_path", "/a") ++ bParam("content", "<" ++ BD ++ "tool_calls>")) ++ b_close,
    b_open ++ bInvoke("Write", bParam("file_path", "/a") ++ bParam("content", "<|tool_call> and <x:tool_call>")) ++ b_close,
    b_open ++ bInvoke("Write", bParam("file_path", "/a") ++ bParam("content", "a file ends with " ++ b_close ++ " here")) ++ b_close,
    V4.open ++ V4.invoke ++ "Write\">\n" ++ V4.param ++ "file_path\" string=\"true\">/a" ++ V4.param_close ++ V4.param ++ "content\" string=\"true\"><tool_call> and <" ++ BD ++ " calls>" ++ V4.param_close ++ V4.invoke_close ++ V4.close,
    "<tool_call>{\"name\":\"Read\",\"arguments\":{\"file_path\":\"<|tool_call>\"}}</tool_call>",
    "<x:tool_call>{\"name\":\"Read\",\"arguments\":{\"file_path\":\"<tool_call>\"}}</x:tool_call>",
    "<x:tool_call>{\"name\":\"Read\",\"arguments\":{\"file_path\":\"/a\"}}</x:tool_call><x:tool_call>",
    "<tool_call>\n\n" ++ b_open ++ bInvoke("Read", bParam("file_path", "/a")) ++ b_close ++ "\n\n<" ++ BD ++ " calls>\n</tool_call>",
    b_open ++ "<" ++ BD ++ " invoke name=\"Write\">\n" ++ bParam("file_path", "/a") ++ "<" ++ BD ++ " parameter name=\"content\" string=\"true\">half",
    b_open ++ bInvoke("Read", bParam("file_path", "/a")) ++ b_close ++ "<tool_call>{\"name\":\"Re",
    b_open ++ bInvoke("Read", bParam("file_path", "/a")) ++ b_close ++ b_open ++ "<" ++ BD ++ " invoke name=\"Write",
    V4.open ++ V4.invoke ++ "Write\">\n",
    "<tool_call>{\"name\":\"Read\",\"arguments\":{\"file_path\":\"/a\"}}</tool_call><|tool_call>call:Read{",
};

fn bodyOf(a: Allocator, wire: []const u8) !json.Value {
    return (try json.parse(a, wire[(std.mem.indexOf(u8, wire, "\r\n\r\n") orelse return error.NoBody) + 4 ..])).ok;
}

// Live mode reports a reply max_tokens ended as cut on every route, whatever calls it made or sent (as OpenAI's API
// reports length whenever the limit was reached); the same reply ended by the model reports its calls. Hold mode and
// the whole reply report what the parse makes of it, as before.
test "over HTTP, max_tokens: live mode ends every route as cut, whatever the calls; ended by the model, as a call; hold as the whole reply" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const schema = "{\"type\":\"object\",\"properties\":{\"file_path\":{\"type\":\"string\"},\"content\":{\"type\":\"string\"}}}";
    const chat_tools = "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"Read\",\"parameters\":" ++ schema ++ "}},{\"type\":\"function\",\"function\":{\"name\":\"Write\",\"parameters\":" ++ schema ++ "}}]}";
    const msg_tools = "\"tools\":[{\"name\":\"Read\",\"input_schema\":" ++ schema ++ "},{\"name\":\"Write\",\"input_schema\":" ++ schema ++ "}]}";
    const resp_tools = "\"tools\":[{\"type\":\"function\",\"name\":\"Read\",\"parameters\":" ++ schema ++ "},{\"type\":\"function\",\"name\":\"Write\",\"parameters\":" ++ schema ++ "}]}";
    const user = "\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],";
    for ([_]tool_streaming.Mode{ .live, .hold }) |mode| {
        var bs: ByteServer = .{ .mode = mode };
        try bs.init(false);
        defer bs.deinit();
        for (at_limit) |answer| for ([_]bool{ true, false }) |by_length| {
            const ids = try a.alloc(u32, answer.len);
            for (answer, ids) |c, *id| id.* = c;
            bs.backend.ids = ids;
            bs.backend.cut = by_length;
            const limit = if (by_length) answer.len else 4096;
            const max = try std.fmt.allocPrint(a, "{{\"model\":\"test-model\",\"max_tokens\":{d},", .{limit});
            const rmax = try std.fmt.allocPrint(a, "{{\"model\":\"test-model\",\"max_output_tokens\":{d},\"input\":\"x\",", .{limit});
            const chat_body = try std.mem.concat(a, u8, &.{ max, user, chat_tools });
            const msg_body = try std.mem.concat(a, u8, &.{ max, user, msg_tools });
            const resp_body = try std.mem.concat(a, u8, &.{ rmax, resp_tools });
            bs.backend.pattern = &.{answer.len};
            const whole = try wirePost(a, bs.srv, "/v1/chat/completions", try withStream(a, chat_body, false));
            const whole_finish = (try bodyOf(a, whole)).get("choices").?.array[0].strField("finish_reason").?;
            const whole_m = (try bodyOf(a, try wirePost(a, bs.srv, "/v1/messages", try withStream(a, msg_body, false)))).strField("stop_reason").?;
            const whole_r = (try bodyOf(a, try wirePost(a, bs.srv, "/v1/responses", try withStream(a, resp_body, false)))).strField("status").?;
            const made = (try wireReply(a, whole, "/v1/chat/completions", false)).names.len > 0;
            // the whole reply is the same in both modes: a call it parses wins over the limit, as before
            try std.testing.expectEqualStrings(if (made) "tool_calls" else if (by_length) "length" else "stop", whole_finish);
            for ([_][]const usize{ &.{1}, &.{answer.len} }) |p| {
                bs.backend.pattern = p;
                const c = try wirePost(a, bs.srv, "/v1/chat/completions", try withStream(a, chat_body, true));
                const m = try wirePost(a, bs.srv, "/v1/messages", try withStream(a, msg_body, true));
                const r = try wirePost(a, bs.srv, "/v1/responses", try withStream(a, resp_body, true));
                const fc = try wireFinish(a, c, "/v1/chat/completions");
                const fm = try wireFinish(a, m, "/v1/messages");
                const fr = (try finalResponse(a, r)).strField("status").?;
                if (mode == .hold) {
                    try std.testing.expectEqualStrings(whole_finish, fc);
                    try std.testing.expectEqualStrings(whole_m, fm);
                    try std.testing.expectEqualStrings(whole_r, fr);
                } else if (by_length) {
                    try std.testing.expectEqualStrings("length", fc);
                    try std.testing.expectEqualStrings("max_tokens", fm);
                    try std.testing.expectEqualStrings("incomplete", fr);
                } else {
                    try std.testing.expectEqualStrings(if (made) "tool_calls" else "stop", fc);
                    try std.testing.expectEqualStrings(if (made) "tool_use" else "end_turn", fm);
                    try std.testing.expectEqualStrings("completed", fr);
                }
            }
        };
    }
}

// Live mode, a block the parse keeps as text: the end sends the whole reply's text past the prose that streamed when that
// prose starts it and a header the parse drops went out, or the length cut the reply; ended by the model before any
// header went out, only the prose before the opener, or the whole text when nothing streamed; a value that held the
// block's closer leaves 0048's filter view.
test "over HTTP, live mode, a block the parse keeps as text: which text each route's stream ends with" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const schema = "{\"type\":\"object\",\"properties\":{\"file_path\":{\"type\":\"string\"},\"content\":{\"type\":\"string\"}}}";
    const chat_tools = "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"Read\",\"parameters\":" ++ schema ++ "}},{\"type\":\"function\",\"function\":{\"name\":\"Write\",\"parameters\":" ++ schema ++ "}}]}";
    const msg_tools = "\"tools\":[{\"name\":\"Read\",\"input_schema\":" ++ schema ++ "},{\"name\":\"Write\",\"input_schema\":" ++ schema ++ "}]}";
    const resp_tools = "\"tools\":[{\"type\":\"function\",\"name\":\"Read\",\"parameters\":" ++ schema ++ "},{\"type\":\"function\",\"name\":\"Write\",\"parameters\":" ++ schema ++ "}]}";
    const user = "\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],";
    const header = comptime ("Writing it.<" ++ BD ++ " calls><" ++ BD ++ " invoke name=\"Write\"><" ++ BD ++ " parameter name=\"content\">x");
    const no_header = comptime ("Writing it.<" ++ BD ++ " calls><" ++ BD ++ " invoke name=\"Wri");
    const closer = comptime (header ++ "</" ++ BD ++ " calls>y");
    const bare = comptime ("<" ++ BD ++ " calls><" ++ BD ++ " invoke name=\"Wri"); // nothing streamed before the end
    const cases = comptime [_]struct { answer: []const u8, by_length: bool, text: ?[]const u8 }{
        .{ .answer = header, .by_length = true, .text = null },
        .{ .answer = header, .by_length = false, .text = null },
        .{ .answer = no_header, .by_length = true, .text = null },
        .{ .answer = no_header, .by_length = false, .text = "Writing it." },
        .{ .answer = closer, .by_length = true, .text = "Writing it.y" },
        .{ .answer = closer, .by_length = false, .text = "Writing it.y" },
        .{ .answer = bare, .by_length = true, .text = null },
        .{ .answer = bare, .by_length = false, .text = null },
    };
    var bs: ByteServer = .{ .mode = .live };
    try bs.init(false);
    defer bs.deinit();
    for (cases) |cs| {
        const ids = try a.alloc(u32, cs.answer.len);
        for (cs.answer, ids) |c, *id| id.* = c;
        bs.backend.ids = ids;
        bs.backend.cut = cs.by_length;
        const limit = if (cs.by_length) cs.answer.len else 4096;
        const max = try std.fmt.allocPrint(a, "{{\"model\":\"test-model\",\"max_tokens\":{d},", .{limit});
        const rmax = try std.fmt.allocPrint(a, "{{\"model\":\"test-model\",\"max_output_tokens\":{d},\"input\":\"x\",", .{limit});
        const chat_body = try std.mem.concat(a, u8, &.{ max, user, chat_tools });
        const msg_body = try std.mem.concat(a, u8, &.{ max, user, msg_tools });
        const resp_body = try std.mem.concat(a, u8, &.{ rmax, resp_tools });
        bs.backend.pattern = &.{cs.answer.len};
        const w = try wireReply(a, try wirePost(a, bs.srv, "/v1/chat/completions", try withStream(a, chat_body, false)), "/v1/chat/completions", false);
        try std.testing.expectEqual(@as(usize, 0), w.names.len);
        try std.testing.expectEqualStrings(cs.answer, w.text);
        const want = cs.text orelse cs.answer; // null: the whole reply's text
        for ([_][]const usize{ &.{1}, &.{cs.answer.len} }) |p| {
            bs.backend.pattern = p;
            try std.testing.expectEqualStrings(want, (try wireReply(a, try wirePost(a, bs.srv, "/v1/chat/completions", try withStream(a, chat_body, true)), "/v1/chat/completions", true)).text);
            try std.testing.expectEqualStrings(want, (try wireReply(a, try wirePost(a, bs.srv, "/v1/messages", try withStream(a, msg_body, true)), "/v1/messages", true)).text);
            try std.testing.expectEqualStrings(want, (try wireReply(a, try wirePost(a, bs.srv, "/v1/responses", try withStream(a, resp_body, true)), "/v1/responses", true)).text);
        }
    }
}

// Every route's stream, in both modes, at one-byte and whole rounds, ended by max_tokens, by the model and by a stop
// string: its text is never longer than the text the model wrote, and holds the whole reply's text no more often than that.
test "over HTTP, the end of a stream sends a reply's text at most once, on every route, however the reply ends" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const schema = "{\"type\":\"object\",\"properties\":{\"file_path\":{\"type\":\"string\"},\"content\":{\"type\":\"string\"}}}";
    const chat_tools = "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"Read\",\"parameters\":" ++ schema ++ "}},{\"type\":\"function\",\"function\":{\"name\":\"Write\",\"parameters\":" ++ schema ++ "}}]}";
    const msg_tools = "\"tools\":[{\"name\":\"Read\",\"input_schema\":" ++ schema ++ "},{\"name\":\"Write\",\"input_schema\":" ++ schema ++ "}]}";
    const resp_tools = "\"tools\":[{\"type\":\"function\",\"name\":\"Read\",\"parameters\":" ++ schema ++ "},{\"type\":\"function\",\"name\":\"Write\",\"parameters\":" ++ schema ++ "}]}";
    const user = "\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],";
    const header = comptime ("Writing it.<" ++ BD ++ " calls><" ++ BD ++ " invoke name=\"Write\"><" ++ BD ++ " parameter name=\"content\">x");
    const extra = comptime [_][]const u8{
        "<tool_call>bad</tool_call>",
        "Hello there. <tool_call>bad</tool_call> and after",
        header,
        "<" ++ BD ++ " calls><" ++ BD ++ " invoke name=\"Wri",
        "Writing it.<" ++ BD ++ " calls><" ++ BD ++ " invoke name=\"Wri",
        header ++ "</" ++ BD ++ " calls>y",
        "Plain prose with no call at all, long enough to matter.",
    };
    const replies = at_limit ++ extra;
    const stop_word = "STOPHERE";
    for ([_]tool_streaming.Mode{ .live, .hold }) |mode| {
        var bs: ByteServer = .{ .mode = mode };
        try bs.init(false);
        defer bs.deinit();
        for (replies) |reply| for ([_]enum { length, eos, stop }{ .length, .eos, .stop }) |ending| {
            const answer = if (ending == .stop) try std.mem.concat(a, u8, &.{ reply, stop_word, " after the stop" }) else reply;
            const ids = try a.alloc(u32, answer.len);
            for (answer, ids) |c, *id| id.* = c;
            bs.backend.ids = ids;
            bs.backend.cut = ending == .length;
            const limit = if (ending == .length) answer.len else 4096;
            const stop_chat = if (ending == .stop) "\"stop\":[\"" ++ stop_word ++ "\"]," else "";
            const stop_msg = if (ending == .stop) "\"stop_sequences\":[\"" ++ stop_word ++ "\"]," else "";
            const max = try std.fmt.allocPrint(a, "{{\"model\":\"test-model\",\"max_tokens\":{d},", .{limit});
            const rmax = try std.fmt.allocPrint(a, "{{\"model\":\"test-model\",\"max_output_tokens\":{d},\"input\":\"x\",", .{limit});
            const chat_body = try std.mem.concat(a, u8, &.{ max, stop_chat, user, chat_tools });
            const msg_body = try std.mem.concat(a, u8, &.{ max, stop_msg, user, msg_tools });
            const resp_body = try std.mem.concat(a, u8, &.{ rmax, resp_tools });
            bs.backend.pattern = &.{answer.len};
            const whole = (try wireReply(a, try wirePost(a, bs.srv, "/v1/chat/completions", try withStream(a, chat_body, false)), "/v1/chat/completions", false)).text;
            const wrote = if (ending == .stop) reply else answer; // what the model wrote before the stop string
            for ([_][]const usize{ &.{1}, &.{answer.len} }) |p| {
                bs.backend.pattern = p;
                var routes: [3][]const u8 = undefined;
                routes[0] = (try wireReply(a, try wirePost(a, bs.srv, "/v1/chat/completions", try withStream(a, chat_body, true)), "/v1/chat/completions", true)).text;
                routes[1] = (try wireReply(a, try wirePost(a, bs.srv, "/v1/messages", try withStream(a, msg_body, true)), "/v1/messages", true)).text;
                routes[2] = if (ending == .stop) "" else (try wireReply(a, try wirePost(a, bs.srv, "/v1/responses", try withStream(a, resp_body, true)), "/v1/responses", true)).text;
                for (routes, 0..) |s, ri| {
                    if (s.len > wrote.len) {
                        std.debug.print("mode {s} ending {s} rounds {any}: streamed {d} bytes past the {d} the model wrote: {f}\n", .{ @tagName(mode), @tagName(ending), p, s.len, wrote.len, std.zig.fmtString(s) });
                        return error.TestUnexpectedResult;
                    }
                    if (whole.len > 0 and std.mem.count(u8, s, whole) > @max(1, std.mem.count(u8, wrote, whole))) {
                        std.debug.print("mode {s} ending {s} route {d} rounds {any}: the whole text more often than the model wrote it: {f} (whole {f})\n", .{ @tagName(mode), @tagName(ending), ri, p, std.zig.fmtString(s), std.zig.fmtString(whole) });
                        return error.TestUnexpectedResult;
                    }
                }
            }
        };
    }
}

// A long value with the block's closer near its start: while the block's end is undecided, each round's stream filter
// resumes where the last left off, so the reply streams in time linear in its length, not a scan from the block's
// start (or the reply's) every round.
test "over the chat route, a 256 KiB value holding its block's closer near its start streams in time linear in its length" {
    const clock = @import("clock.zig");
    var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var bs: ByteServer = .{ .max_tokens = 1 << 19 };
    try bs.init(false);
    defer bs.deinit();
    const long = try a.alloc(u8, 1 << 18);
    @memset(long, 'x');
    const value = try std.mem.concat(a, u8, &.{ "end: ", spellings[0].close, " then ", long });
    const answer = try std.mem.concat(a, u8, &.{ "Writing.", try spWrite(a, spellings[0], "Write", value) });
    const started = clock.nowNs(std.testing.io);
    const r = try bs.chat(a, answer, true, false, &.{16});
    const took = clock.seconds(clock.nowNs(std.testing.io) - started);
    try std.testing.expect(r.done);
    try std.testing.expectEqual(@as(usize, 1), r.calls);
    try std.testing.expectEqualStrings(value, (try json.parse(a, r.args[0])).ok.strField("content").?);
    // a scan from the block's start every round took over 70 s here
    std.testing.expect(took < 2.0) catch |e| {
        std.debug.print("256 KiB value, closer near its start, 16-byte rounds: {d:.2} s\n", .{took});
        return e;
    };
}

// Prose that starts with whitespace, then a DSML call the max_tokens cut leaves open: the whole reply keeps the open block
// as text after the prose, whitespace and all. The live stream sent the prose with its whitespace and hid the block; at
// the end it sends the rest of the whole reply's text after exactly what it sent, so each route's stream ends with the
// whole reply's text, the block's text not dropped and the prose not sent twice.
test "over HTTP, prose with leading whitespace, then a DSML call cut by max_tokens: each route's stream ends with the whole reply's text" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var bs: ByteServer = .{};
    try bs.init(false);
    defer bs.deinit();
    bs.backend.cut = true;
    const schema = "{\"type\":\"object\",\"properties\":{\"file_path\":{\"type\":\"string\"},\"content\":{\"type\":\"string\"}}}";
    const user = "\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],";
    for ([_][]const u8{ " hello", "  \n hello", "\thello there" }) |prose| {
        const answer = try std.mem.concat(a, u8, &.{ prose, b_open, "<" ++ BD ++ " invoke name=\"Write\">\n", bParam("file_path", "/a"), "<" ++ BD ++ " parameter name=\"content\" string=\"true\">half of it" });
        const ids = try a.alloc(u32, answer.len);
        for (answer, ids) |c, *id| id.* = c;
        bs.backend.ids = ids;
        const max = try std.fmt.allocPrint(a, "{{\"model\":\"test-model\",\"max_tokens\":{d},", .{answer.len});
        const rmax = try std.fmt.allocPrint(a, "{{\"model\":\"test-model\",\"max_output_tokens\":{d},\"input\":\"x\",", .{answer.len});
        const bodies = [_][]const u8{
            try std.mem.concat(a, u8, &.{ max, user, "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"Write\",\"parameters\":", schema, "}}]}" }),
            try std.mem.concat(a, u8, &.{ max, user, "\"tools\":[{\"name\":\"Write\",\"input_schema\":", schema, "}]}" }),
            try std.mem.concat(a, u8, &.{ rmax, "\"tools\":[{\"type\":\"function\",\"name\":\"Write\",\"parameters\":", schema, "}]}" }),
        };
        for ([_][]const u8{ "/v1/chat/completions", "/v1/messages", "/v1/responses" }, bodies) |path, body| {
            bs.backend.pattern = &.{answer.len};
            const want = try wireReply(a, try wirePost(a, bs.srv, path, try withStream(a, body, false)), path, false);
            try std.testing.expectEqual(@as(usize, 0), want.names.len);
            try std.testing.expect(std.mem.indexOf(u8, want.text, "<" ++ BD ++ " calls>") != null);
            for ([_][]const usize{ &.{1}, &.{3}, &.{answer.len} }) |pattern| {
                bs.backend.pattern = pattern;
                const got = try wireReply(a, try wirePost(a, bs.srv, path, try withStream(a, body, true)), path, true);
                std.testing.expectEqualStrings(want.text, got.text) catch |e| {
                    std.debug.print("{s} rounds of {d}\n", .{ path, pattern[0] });
                    return e;
                };
            }
        }
    }
}

