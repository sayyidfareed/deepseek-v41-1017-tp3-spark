// Modified for the DeepSeek 1017 TP3 integration; upstream copyrights retained. See CREDITS.md and provenance.json.
//! Only routes.dispatch on the authenticated secure listener reaches this path.
//! No request body is read, and losing the socket never cancels accepted erasure.
const std = @import("std");
const Conn = @import("http_conn.zig").Conn;
const Server = @import("server.zig").Server;

pub fn emptyBody(length_count: usize, length: ?[]const u8, transfer_count: usize) bool {
    if (transfer_count != 0 or length_count > 1) return false;
    if (length_count == 0) return length == null;
    return if (length) |v| std.mem.eql(u8, v, "0") else false;
}

pub fn post(srv: *Server, conn: *Conn) void {
    conn.close = true;
    if (!srv.config.secure_request or !emptyBody(conn.headerCount("Content-Length"),
        conn.header("Content-Length"), conn.headerCount("Transfer-Encoding"))) {
        conn.sendJson(400, "{\"error\":{\"message\":\"empty body required\"}}");
        return;
    }
    const forget = srv.engine.vtable.secure_forget orelse {
        conn.sendJson(503, "{\"error\":{\"message\":\"forget unavailable\"}}");
        return;
    };
    const count = forget(srv.engine.ctx, if (conn.secure_cache_scope) |*scope| scope else null) catch {
        conn.sendJson(503, "{\"error\":{\"message\":\"forget incomplete\"}}");
        return;
    };
    var buffer: [64]u8 = undefined;
    const body = std.fmt.bufPrint(&buffer, "{{\"forgotten\":{d}}}", .{count}) catch unreachable;
    conn.sendJson(200, body);
}

test "forget accepts only an absent or explicitly zero body without ambiguous framing" {
    try std.testing.expect(emptyBody(0, null, 0));
    try std.testing.expect(emptyBody(1, "0", 0));
    try std.testing.expect(!emptyBody(2, "0", 0));
    try std.testing.expect(!emptyBody(1, "0", 1));
    try std.testing.expect(!emptyBody(1, "2", 0));
    try std.testing.expect(!emptyBody(1, "", 0));
    try std.testing.expect(!emptyBody(1, "0,0", 0));
}

test "secure forget calls the checked engine path after a client has disconnected" {
    var sockets: [2]std.c.fd_t = undefined;
    if (std.c.socketpair(std.c.AF.UNIX, std.c.SOCK.STREAM, 0, &sockets) != 0) return error.SocketFailed;
    defer _ = std.c.close(sockets[1]);
    var conn = try Conn.init(std.testing.allocator, sockets[1], "synthetic");
    defer conn.deinit();
    conn.secure_request = true;
    conn.secure_cache_scope = @splat(1);
    const Counter = struct {
        calls: usize = 0,
        scoped: bool = false,
        fn forget(p: *anyopaque, scope: ?*const [32]u8) !usize {
            const c: *@This() = @ptrCast(@alignCast(p));
            c.calls += 1;
            c.scoped = if (scope) |s| std.mem.allEqual(u8, s, 1) else false;
            return 2;
        }
    };
    var counter: Counter = .{};
    var server: Server = undefined;
    server.config = .{ .secure_request = true, .served_name = "synthetic", .model_ids = &.{"synthetic"} };
    server.engine = .{ .ctx = &counter, .vtable = &.{ .secure_forget = Counter.forget,
        .info = undefined, .submit = undefined, .cancel = undefined, .status = undefined, .memory = undefined } };
    _ = std.c.close(sockets[0]);
    post(&server, &conn);
    try std.testing.expectEqual(@as(usize, 1), counter.calls);
    try std.testing.expect(counter.scoped and conn.close and conn.broken);
}
