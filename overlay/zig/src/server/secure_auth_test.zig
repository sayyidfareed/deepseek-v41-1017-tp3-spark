// Modified for the DeepSeek 1017 TP3 integration; upstream copyrights retained. See CREDITS.md and provenance.json.
const std = @import("std");
const auth = @import("auth.zig");
const http = @import("http_conn.zig");

test "secure digest file authenticates raw bearer only and rejects plaintext provisioning" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var hash: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash("synthetic-key", &hash, .{});
    const hex = std.fmt.bytesToHex(hash, .lower);
    try tmp.dir.writeFile(io, .{ .sub_path = "digest", .data = &hex });
    const path = try std.fmt.allocPrintSentinel(a, ".zig-cache/tmp/{s}/digest", .{tmp.sub_path}, 0);
    try std.testing.expectEqual(@as(c_int, 0), std.c.fchmodat(std.posix.AT.FDCWD, path, 0o600, 0));
    var problem: []const u8 = "";
    var store = try auth.Store.initDigests(a, path, &problem);
    defer store.deinit();
    try std.testing.expect(store.matchWithDiagnostics(a, "Bearer synthetic-key", null, false) != null);
    const hash_bearer = try std.fmt.allocPrint(a, "Bearer {s}", .{hex});
    try std.testing.expect(store.matchWithDiagnostics(a, hash_bearer, null, false) == null);
    try std.testing.expect(store.matchWithDiagnostics(a, "Bearer wrong", null, false) == null);
    try tmp.dir.writeFile(io, .{ .sub_path = "digest", .data = "plaintext-must-not-load" });
    store.checked_ns = null;
    try std.testing.expect(store.matchWithDiagnostics(a, "Bearer synthetic-key", null, false) == null);
    try std.testing.expectError(error.KeyFile, auth.Store.initDigests(a, path, &problem));
}

test "secure request head never reads the body even when sent together" {
    var fds: [2]std.c.fd_t = undefined;
    if (std.c.socketpair(std.c.AF.UNIX, std.c.SOCK.STREAM, 0, &fds) != 0) return error.SocketFailed;
    defer for (fds) |fd| { _ = std.c.close(fd); };
    const bytes = "POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 4\r\nAuthorization: Bearer wrong\r\n\r\nBODY";
    try std.testing.expectEqual(@as(isize, bytes.len), std.c.send(fds[0], bytes.ptr, bytes.len, 0));
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var conn = try http.Conn.init(arena.allocator(), fds[1], "synthetic");
    defer conn.deinit();
    conn.secure_request = true;
    try std.testing.expectEqual(http.Outcome.ready, try conn.readRequest(arena.allocator(), false));
    try std.testing.expectEqual(conn.start, conn.end);
    var remaining: [4]u8 = undefined;
    try std.testing.expectEqual(@as(isize, 4), std.c.recv(fds[1], &remaining, 4, std.c.MSG.DONTWAIT));
    try std.testing.expectEqualStrings("BODY", &remaining);
}
