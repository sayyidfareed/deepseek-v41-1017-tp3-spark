// Modified for the DeepSeek 1017 TP3 integration; upstream copyrights retained. See CREDITS.md and provenance.json.
//! Private TP3 execution on the existing ordered rank links. The ordinary lane
//! thread must be paused, every rank's process policy established, and HTTP held.
//! No serving capability is advertised by this module: policy and router are separate.
const std = @import("std");
const api = @import("engine_api");
const lanes = @import("lanes");
const lm = @import("lanes.zig");
const Model = @import("model.zig").Model;
const private = @import("secure_state.zig");
const Fan = @import("prompt2d.zig").Fan;
const Comm = @import("comm.zig").Comm;
const Guard = @import("secure_policy.zig").Guard;

// The lane progress monitor does not cover communicator construction or cleanup.
// A blocked library call in either transition must end the process, not retain a
// private state forever. This watchdog never frees buffers under GPU work.
const Transition = struct {
    done: std.atomic.Value(bool) = .init(false),
    thread: ?std.Thread = null,
    deadline: u64 = 0,
    fn start(t: *Transition) !void {
        t.done.store(false, .release);
        t.deadline = @import("link.zig").Link.monotonicNs() + 120 * std.time.ns_per_s;
        t.thread = try std.Thread.spawn(.{}, run, .{t});
    }
    fn end(t: *Transition) void {
        t.done.store(true, .release);
        if (t.thread) |thread| thread.join();
        t.thread = null;
    }
    fn run(t: *Transition) void {
        while (!t.done.load(.acquire)) {
            if (@import("link.zig").Link.monotonicNs() >= t.deadline) std.c._exit(70);
            var ts: std.c.timespec = .{ .sec = 0, .nsec = 10 * std.time.ns_per_ms };
            _ = std.c.nanosleep(&ts, null);
        }
    }
};

pub const Session = struct {
    parent: std.mem.Allocator,
    epoch: u64,
    guard: Guard,
    state: *private.State,
    comm: Comm,
    ln: lm.Lanes,
    cfg: lanes.Config,
    clock: lanes.backend.WallClock,
    core: lanes.Engine,
    host: api.LaneHost,

    /// A failed participant is never treated as ready. The controller may resume
    /// ordinary work only on SecurePreparationRefused (all ranks rolled back).
    pub fn enter(parent: std.mem.Allocator, ordinary: *lm.Lanes, o: private.Options) !*Session {
        const m = ordinary.m;
        if (m.world != 3 or m.rank != 0) return error.PrivateStateNeedsThreeRanks;
        var transition: Transition = .{};
        try transition.start();
        defer transition.end();
        if (ordinary.streams.count() != 0 or ordinary.fill != null or ordinary.paused != null) return error.SecureBackendBusy;
        for (ordinary.parked_fills) |f| if (f != null) return error.SecureBackendBusy;
        if (ordinary.spill) |sp| {
            const tier = sp.tier();
            const busy = tier.vtable.busy orelse return error.SecureBackendNotQuiescent;
            if (busy(tier.ptr)) return error.SecureBackendBusy;
        }
        m.joinAhead();
        if (m.ctx.d.api.cuCtxSynchronize() != 0) return error.SecureBackendNotQuiescent;
        const epoch = ordinary.step;
        const guard = try Guard.enter();
        try ordinary.wire.begin(ordinary.gpa, .secure_begin, ordinary.step);
        ordinary.step += 1;
        try ordinary.wire.int(ordinary.gpa, u64, @intCast(o.window));
        try ordinary.wire.int(ordinary.gpa, u64, @intCast(o.device_bytes));
        for (m.followers()) |l| try l.?.send(ordinary.wire.buf.items);
        if (!try agree(&m.fan, 0, epoch, 1, true)) {
            try guard.leave();
            if (!try agree(&m.fan, 0, epoch, 3, true)) return error.SecureCleanupFailed;
            return error.SecurePreparationRefused;
        }
        var comm = try m.fan.comm(m.rank, m.world);
        // A transport/collective failure after agreement is fatal to the private
        // session. Do not free buffers under unknown GPU work or resume ordinary.
        const s = prepare(parent, m, comm, o, epoch, guard) catch null;
        if (!try agree(&m.fan, 0, epoch, 2, s != null)) {
            if (s) |x| try x.dispose(false) else try comm.destroyPrivate();
            try guard.leave();
            if (!try agree(&m.fan, 0, epoch, 3, true)) return error.SecureCleanupFailed;
            return error.SecurePreparationRefused;
        }
        const ready = s.?;
        ready.host.start() catch {
            // Followers are in the nested command loop. End it once, then await
            // their cleanup before reporting a recoverable preparation refusal.
            try ready.finish();
            return error.SecurePreparationRefused;
        };
        if (ready.host.pair_monitor == null) {
            try ready.finish();
            return error.SecurePreparationRefused;
        }
        return ready;
    }

    fn prepare(parent: std.mem.Allocator, m: *Model, comm: Comm, o: private.Options, epoch: u64, guard: Guard) !*Session {
        const s = try parent.create(Session);
        errdefer parent.destroy(s);
        s.parent = parent;
        s.epoch = epoch;
        s.guard = guard;
        s.comm = comm;
        s.state = try private.State.prepare(parent, m, &s.comm, m.fan, o);
        errdefer { s.state.erase() catch std.process.exit(70); s.state.release() catch std.process.exit(70); }
        const gpa = s.state.m.gpa;
        s.ln = lm.Lanes.init(gpa, try s.state.begin());
        s.ln.private_request = true;
        s.cfg = try lanes.Config.init(gpa, s.ln.facts(), @intCast(s.state.m.eng.round_rows), @intCast(s.state.m.eng.round_rows - 1));
        errdefer s.cfg.deinit(gpa);
        s.clock = .{ .io = m.io };
        s.core = lanes.Engine.init(gpa, &s.cfg, s.ln.backend(), s.clock.clock());
        s.core.fill_chunks = 1;
        s.core.fill_first = 1;
        s.host = api.LaneHost.init(gpa, m.io, &s.core, .{
            .lanes = 1, .context_window = @intCast(o.window), .call_gates = true,
        });
        // No prompt store, spill tier, learner, keepalive, public memory source,
        // graph profile, vision hook or independent diagnostic monitor is attached.
        s.host.enter = .{ .ctx = s, .run = bind };
        s.host.pair = .{ .ctx = s, .abort = abortPrivate, .exit = exitPrivate, .code = 70 };
        return s;
    }

    fn abortPrivate(p: *anyopaque) void {
        const s: *Session = @ptrCast(@alignCast(p));
        s.comm.aborted.store(true, .release);
        _ = s.state.m.comm.abort();
    }

    fn exitPrivate(_: *anyopaque, _: u8) void { std.c._exit(70); }

    fn bind(p: *anyopaque) void {
        const s: *Session = @ptrCast(@alignCast(p));
        s.state.m.ctx.makeCurrent() catch std.process.exit(70);
    }

    pub fn engine(s: *Session) api.Engine { return s.host.engine(); }

    /// After the HTTP sink has finished/cancelled and drained. It is legitimate to
    /// join here; never substitute a timeout followed by forced freeing of state.
    pub fn finish(s: *Session) !void {
        var transition: Transition = .{};
        try transition.start();
        defer transition.end();
        const fan = s.state.m.fan;
        const epoch = s.epoch;
        const guard = s.guard;
        if (s.host.thread != null) s.host.stop();
        s.core.deinit();
        try s.ln.endPrivate();
        // The checked nested .done above is sent once, then host arrays are erased.
        s.ln.deinit();
        s.cfg.deinit(s.state.m.gpa);
        try s.dispose(true);
        try guard.leave();
        if (!try agree(&fan, 0, epoch, 3, true)) return error.SecureCleanupFailed;
    }

    fn dispose(s: *Session, core_released: bool) !void {
        if (!core_released) {
            s.core.deinit();
            // Preparation was refused before nested commands began: do not send
            // .done into the ordinary command stream.
            s.ln.peer = null;
            s.ln.deinit();
            s.cfg.deinit(s.state.m.gpa);
        }
        try s.comm.destroyPrivate(); // private communicator only; ordinary NCCL is untouched
        try s.state.erase();
        try s.state.release();
        const parent = s.parent;
        std.crypto.secureZero(u8, std.mem.asBytes(s));
        parent.destroy(s);
    }
};

/// Runs only at the outer follower loop's boundary, after its own ordinary spill
/// queue is drained. All nested frame buffers use the private erasing allocator.
pub fn follow(parent: std.mem.Allocator, m: *Model, o: private.Options, idle: bool, epoch: u64) anyerror!void {
    var transition: Transition = .{};
    try transition.start();
    defer transition.end();
    m.joinAhead();
    const guard = Guard.enter() catch null;
    // A dead head must also end a follower stuck inside a private collective,
    // without entering the ordinary Model.close synchronization path.
    var watch = @import("link.zig").Watch.init(m.fan.links[0..1], m, lostHead);
    try watch.start();
    defer watch.end();
    const quiet = guard != null and idle and m.ctx.d.api.cuCtxSynchronize() == 0;
    if (!try agree(&m.fan, m.rank, epoch, 1, quiet)) {
        if (guard) |g| try g.leave();
        // A missing/partially entered guard is not a verified rollback.
        if (!try agree(&m.fan, m.rank, epoch, 3, guard != null)) return error.SecureCleanupFailed;
        return;
    }
    var comm = try m.fan.comm(m.rank, m.world);
    const state = private.State.prepare(parent, m, &comm, m.fan, o) catch null;
    if (!try agree(&m.fan, m.rank, epoch, 2, state != null)) {
        try comm.destroyPrivate();
        if (state) |s| { try s.erase(); try s.release(); }
        try guard.?.leave();
        if (!try agree(&m.fan, m.rank, epoch, 3, true)) return error.SecureCleanupFailed;
        return;
    }
    const s = state.?;
    transition.end();
    // Any protocol or cleanup failure propagates out of the outer follower; it
    // must not announce a successful wipe or continue ordinary commands.
    try lm.followPrivate(s.m.gpa, try s.begin());
    try transition.start();
    try comm.destroyPrivate();
    try s.erase();
    try s.release();
    try guard.?.leave();
    if (!try agree(&m.fan, m.rank, epoch, 3, true)) return error.SecureCleanupFailed;
}

fn lostHead(_: *anyopaque, _: usize) void { std.c._exit(70); }

/// Fixed, content-free control records. Missing, malformed or duplicate phases
/// are not success. Called in the same three phases on every participant.
fn agree(fan: *const Fan, rank: u32, epoch: u64, phase: u8, ready: bool) !bool {
    const deadline = @import("link.zig").Link.monotonicNs() +| (if (@import("builtin").is_test) @as(u64, 1) else 120) * std.time.ns_per_s;
    var packet: [11]u8 = undefined;
    packet[0] = 0x53; packet[1] = phase; packet[2] = @intFromBool(ready);
    std.mem.writeInt(u64, packet[3..11], epoch, .little);
    var buf: [11]u8 = undefined;
    if (rank != 0) {
        const peer = fan.links[0] orelse return error.SecureRankMissing;
        try peer.sendDeadline(&packet, deadline);
        const got = try peer.recvDeadline(&buf, deadline);
        return parse(got, epoch, phase);
    }
    var all = ready;
    for (fan.links[0..2]) |link| {
        const peer = link orelse return error.SecureRankMissing;
        all = try parse(try peer.recvDeadline(&buf, deadline), epoch, phase) and all;
    }
    packet[2] = @intFromBool(all);
    for (fan.links[0..2]) |link| try link.?.sendDeadline(&packet, deadline);
    return all;
}

fn parse(packet: []const u8, epoch: u64, phase: u8) !bool {
    if (packet.len != 11 or packet[0] != 0x53 or packet[1] != phase or packet[2] > 1 or
        std.mem.readInt(u64, packet[3..11], .little) != epoch) return error.SecureRankAgreement;
    return packet[2] == 1;
}

test "secure rank agreement rejects missing stale wrong-phase or malformed acknowledgements" {
    var p = [_]u8{ 0x53, 1, 1, 4, 0, 0, 0, 0, 0, 0, 0 };
    try std.testing.expect(try parse(&p, 4, 1));
    try std.testing.expectError(error.SecureRankAgreement, parse(&p, 5, 1));
    try std.testing.expectError(error.SecureRankAgreement, parse(&p, 4, 2));
    try std.testing.expectError(error.SecureRankAgreement, parse(p[0..10], 4, 1));
    p[2] = 0;
    try std.testing.expect(!try parse(&p, 4, 1));
    p[2] = 2;
    try std.testing.expectError(error.SecureRankAgreement, parse(&p, 4, 1));
}

test { _ = &Session.enter; _ = &Session.finish; _ = &follow; }

test "secure cleanup never accepts a silent or disconnected rank" {
    var pair: [2]std.c.fd_t = undefined;
    if (std.c.socketpair(std.c.AF.UNIX, std.c.SOCK.STREAM, 0, &pair) != 0) return error.SocketFailed;
    defer for (pair) |fd| { _ = std.c.close(fd); };
    const fan: Fan = .{ .links = .{ .{ .fd = pair[0] }, null, null } };
    // No acknowledgement is sent. A timeout is a failure, never a successful wipe.
    try std.testing.expectError(error.StartupTimeout, agree(&fan, 0, 19, 3, true));
    _ = std.c.shutdown(pair[1], std.c.SHUT.RDWR);
    try std.testing.expectError(error.PeerDown, agree(&fan, 0, 19, 3, true));
}

test "secure three-rank preparation refusal still requires both cleanup acknowledgements" {
    const Link = @import("link.zig").Link;
    const Peer = struct {
        rank: u32, link: Link, ready: bool,
        preparation: ?bool = null, cleanup: ?bool = null, failed: bool = false,
        fn run(p: *@This()) void {
            const fan: Fan = .{ .links = .{ p.link, null, null } };
            p.preparation = agree(&fan, p.rank, 19, 2, p.ready) catch { p.failed = true; return; };
            p.cleanup = agree(&fan, p.rank, 19, 3, true) catch { p.failed = true; return; };
        }
    };
    for ([_]bool{ true, false }) |second_ready| {
        var fds: [2][2]std.c.fd_t = undefined;
        for (&fds, 0..) |*pair, i| {
            if (std.c.socketpair(std.c.AF.UNIX, std.c.SOCK.STREAM, 0, pair) != 0) {
                for (fds[0..i]) |opened| for (opened) |fd| { _ = std.c.close(fd); };
                return error.SocketFailed;
            }
        }
        defer for (fds) |pair| for (pair) |fd| { _ = std.c.close(fd); };
        var one: Peer = .{ .rank = 1, .link = .{ .fd = fds[0][1] }, .ready = true };
        var two: Peer = .{ .rank = 2, .link = .{ .fd = fds[1][1] }, .ready = second_ready };
        const t1 = try std.Thread.spawn(.{}, Peer.run, .{&one});
        var joined_one = false;
        defer if (!joined_one) { _ = std.c.shutdown(fds[0][0], std.c.SHUT.RDWR); t1.join(); };
        const t2 = try std.Thread.spawn(.{}, Peer.run, .{&two});
        var joined_two = false;
        defer if (!joined_two) { _ = std.c.shutdown(fds[1][0], std.c.SHUT.RDWR); t2.join(); };
        const fan: Fan = .{ .links = .{ .{ .fd = fds[0][0] }, .{ .fd = fds[1][0] }, null } };
        const preparation = try agree(&fan, 0, 19, 2, true);
        const cleanup = try agree(&fan, 0, 19, 3, true);
        t1.join(); joined_one = true;
        t2.join(); joined_two = true;
        try std.testing.expectEqual(second_ready, preparation);
        try std.testing.expect(cleanup);
        try std.testing.expect(!one.failed and !two.failed);
        try std.testing.expectEqual(second_ready, one.preparation.?);
        try std.testing.expectEqual(second_ready, two.preparation.?);
        try std.testing.expect(one.cleanup.? and two.cleanup.?);
    }
}
