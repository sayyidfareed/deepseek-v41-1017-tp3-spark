// Modified for the DeepSeek 1017 TP3 integration; upstream copyrights retained. See CREDITS.md and provenance.json.
//! One authenticated HTTP admission, alive through tokenization, queuing and
//! inference. The host registry is mutex-protected; the engine reads revocation
//! atomically. Scope bytes never leave this process or enter ordinary caching.
const std = @import("std");
pub const Ticket = struct {
    scope: [32]u8,
    revoked: std.atomic.Value(bool) = .init(false),
    next: ?*Ticket = null,
    registered: bool = false,

    pub fn clear(t: *Ticket) void {
        std.debug.assert(!t.registered);
        std.crypto.secureZero(u8, &t.scope);
        t.revoked.store(true, .release);
        t.next = null;
    }
};

pub const Registry = struct {
    head: ?*Ticket = null,
    pub fn add(r: *Registry, t: *Ticket) !void {
        if (t.registered) return error.PrivateTicketAlreadyRegistered;
        t.next = r.head;
        t.registered = true;
        r.head = t;
    }
    pub fn remove(r: *Registry, t: *Ticket) void {
        var ptr = &r.head;
        while (ptr.*) |current| {
            if (current == t) {
                ptr.* = t.next;
                t.registered = false;
                t.clear();
                return;
            }
            ptr = &current.next;
        }
        std.debug.assert(!t.registered);
    }
    pub fn revoke(r: *Registry, scope: ?*const [32]u8) void {
        var next = r.head;
        while (next) |t| : (next = t.next) {
            if (scope) |s| if (!std.crypto.timing_safe.eql([32]u8, s.*, t.scope)) continue;
            t.revoked.store(true, .release);
        }
    }
};

test "forget revokes pre-tokenization and queued admissions without affecting another scope" {
    var a: Ticket = .{ .scope = @splat(1) };
    var b: Ticket = .{ .scope = @splat(2) };
    var c: Ticket = .{ .scope = @splat(1) };
    var r: Registry = .{};
    try r.add(&a);
    try r.add(&b);
    try r.add(&c);
    try std.testing.expectError(error.PrivateTicketAlreadyRegistered, r.add(&a));
    r.revoke(&a.scope);
    try std.testing.expect(a.revoked.load(.acquire) and c.revoked.load(.acquire));
    try std.testing.expect(!b.revoked.load(.acquire));
    r.remove(&c);
    r.remove(&a);
    try std.testing.expect(std.mem.allEqual(u8, &a.scope, 0));
    r.revoke(null);
    try std.testing.expect(b.revoked.load(.acquire));
    r.remove(&b);
    try std.testing.expect(r.head == null);
}
