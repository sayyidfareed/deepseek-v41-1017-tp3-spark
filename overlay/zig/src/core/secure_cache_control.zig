// Modified for the DeepSeek 1017 TP3 integration; upstream copyrights retained. See CREDITS.md and provenance.json.
//! Forget is a command to the existing engine thread, not an inference cancel or
//! exclusive admission pause. HTTP disconnect cannot cancel an accepted command.
const std = @import("std");
pub const tickets = @import("lanes").stream.secure_cache_tickets;
pub const Control = struct {
    admissions: tickets.Registry = .{},
    first: ?*Command = null,
    last: ?*Command = null,
    const Command = struct {
        scope: ?[32]u8,
        next: ?*Command = null,
        done: bool = false,
        failed: bool = false,
        forgotten: usize = 0,
    };

    pub fn enter(c: *Control, h: anytype, ticket: *tickets.Ticket) !void {
        h.mutex.lockUncancelable(h.io);
        defer h.mutex.unlock(h.io);
        if (h.closing or h.thread == null or h.pair_state.load(.acquire) != 0) return error.SecureUnavailable;
        try c.admissions.add(ticket);
    }
    pub fn leave(c: *Control, h: anytype, ticket: *tickets.Ticket) void {
        h.mutex.lockUncancelable(h.io);
        defer h.mutex.unlock(h.io);
        c.admissions.remove(ticket);
    }
    pub fn forget(c: *Control, h: anytype, scope: ?*const [32]u8) !usize {
        var command: Command = .{ .scope = if (scope) |s| s.* else null };
        defer if (command.scope) |*s| std.crypto.secureZero(u8, s);
        h.mutex.lockUncancelable(h.io);
        defer h.mutex.unlock(h.io);
        if (h.closing or h.thread == null or h.pair_state.load(.acquire) != 0) return error.SecureUnavailable;
        c.admissions.revoke(scope);
        if (c.last) |last| last.next = &command else c.first = &command;
        c.last = &command;
        h.wake.broadcast(h.io);
        // Stack command stays owned until the engine acknowledges. Never return
        // merely because a socket disappeared or the client stopped waiting.
        while (!command.done) h.wake.waitUncancelable(h.io, &h.mutex);
        if (command.failed) return error.SecureForgetFailed;
        return command.forgotten;
    }
    pub fn poll(c: *Control, h: anytype) void {
        while (true) {
            h.mutex.lockUncancelable(h.io);
            const command = c.first orelse { h.mutex.unlock(h.io); break; };
            c.first = command.next;
            if (c.first == null) c.last = null;
            h.mutex.unlock(h.io);
            const count = h.core.backend.privateForget(if (command.scope) |*s| s else null) catch std.c._exit(70);
            h.mutex.lockUncancelable(h.io);
            command.forgotten = count;
            command.done = true;
            h.wake.broadcast(h.io);
            h.mutex.unlock(h.io);
        }
        h.core.backend.privateCachePoll() catch std.c._exit(70);
    }
    pub fn failed(c: *Control, h: anytype) void {
        h.mutex.lockUncancelable(h.io);
        defer h.mutex.unlock(h.io);
        c.admissions.revoke(null);
        var next = c.first;
        while (next) |command| {
            next = command.next;
            command.failed = true;
            command.done = true;
        }
        c.first = null;
        c.last = null;
        h.wake.broadcast(h.io);
    }
};
