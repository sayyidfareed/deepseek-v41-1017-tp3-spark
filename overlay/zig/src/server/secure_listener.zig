// Modified for the DeepSeek 1017 TP3 integration; upstream copyrights retained. See CREDITS.md and provenance.json.
//! Authenticated direct-tailnet listener; shares the ordinary listener's admission gate.
const std = @import("std");
const api = @import("engine_api");
const auth = @import("auth.zig");
const server = @import("server.zig");
const listener = @import("listener.zig");
const model_text = @import("model_text.zig");
pub const Secure = struct {
    gpa: std.mem.Allocator, io: std.Io,
    keys: auth.Store, srv: *server.Server, lis: listener.Listener,
    stop_requested: std.atomic.Value(bool) = .init(false),
    thread: ?std.Thread = null,
    pub fn start(gpa: std.mem.Allocator, io: std.Io, engine: api.Engine, text: model_text.Text,
        config: server.Config, host: []const u8, port: u16, key_file: []const u8) !*Secure {
        if (!tailnet(host) or port == 0) return error.SecureTailnetRequired;
        const self = try gpa.create(Secure);
        errdefer gpa.destroy(self);
        var problem: []const u8 = "";
        self.* = .{ .gpa = gpa, .io = io, .keys = try auth.Store.initDigests(gpa, key_file, &problem),
            .srv = undefined, .lis = undefined };
        errdefer self.keys.deinit();
        var cfg = config;
        cfg.secure_request = true;
        cfg.dashboard = false;
        cfg.request_log = null;
        cfg.compact_at = null;
        cfg.compact_memory = null;
        cfg.keep_warm_s = 0;
        self.srv = try server.Server.init(gpa, io, engine, text, cfg, &self.keys);
        errdefer self.srv.deinit();
        // Verify all participating rank guards and reset support before either public listener opens.
        try engine.secureReset(true);
        try engine.secureReset(false);
        self.lis = try listener.Listener.open(try std.Io.net.IpAddress.parse(host, port));
        errdefer self.lis.close();
        self.thread = try std.Thread.spawn(.{}, server.Server.serve, .{self.srv, self.lis, &self.stop_requested});
        return self;
    }
    pub fn stop(self: *Secure) void {
        self.stop_requested.store(true, .release);
        if (self.thread) |thread| thread.join();
        self.lis.close();
        var waited: u32 = 0;
        while (self.srv.open_connections.load(.acquire) > 0 and waited < 40) : (waited += 1)
            std.Io.sleep(self.io, .fromMilliseconds(50), .awake) catch {};
        if (self.srv.open_connections.load(.acquire) > 0) std.process.exit(1);
        self.srv.deinit();
        self.keys.deinit();
        self.gpa.destroy(self);
    }
};
fn tailnet(host: []const u8) bool {
    var it = std.mem.splitScalar(u8, host, '.');
    var parts: [4]u8 = undefined;
    for (&parts) |*p| p.* = std.fmt.parseInt(u8, it.next() orelse return false, 10) catch return false;
    return it.next() == null and parts[0] == 100 and parts[1] >= 64 and parts[1] < 128;
}
test "secure listener refuses wildcard public and loopback binds" {
    try std.testing.expect(tailnet("100.64.0.1"));
    for ([_][]const u8{"0.0.0.0", "127.0.0.1", "192.0.2.10", "100.63.0.1", "100.128.0.1", "100.64.0.256"}) |host|
        try std.testing.expect(!tailnet(host));
}
