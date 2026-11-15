// Modified for the DeepSeek 1017 TP3 integration; upstream copyrights retained. See CREDITS.md and provenance.json.
//! Shared mode admits ordinary and secure work together; the engine owns request cleanup.
//! The retained exclusive mode is only for engines explicitly using that contract.
const std = @import("std");
const api = @import("engine_api");
pub const Gate = struct {
    io: std.Io, engine: api.Engine,
    shared: bool = false,
    private_readers: usize = 0,
    mutex: std.Io.Mutex = .init, changed: std.Io.Condition = .init,
    readers: usize = 0, writers: usize = 0, ordinary_waiters: usize = 0, exclusive: bool = false, broken: bool = false,
    pub fn enter(g: *Gate, secure: bool) !void {
        g.mutex.lockUncancelable(g.io);
        if (g.shared) {
            defer g.mutex.unlock(g.io);
            if (g.broken) return error.SecureUnavailable;
            if (secure) g.private_readers += 1 else g.readers += 1;
            return;
        }
        if (secure) g.writers += 1 else g.ordinary_waiters += 1;
        const deadline = std.Io.Clock.awake.now(g.io).toNanoseconds() + 120 * std.time.ns_per_s;
        while (!g.broken and (g.exclusive or (secure and (g.readers > 0 or g.ordinary_waiters > 0)))) {
            g.changed.waitTimeout(g.io, &g.mutex, .{ .duration = .{ .raw = .fromMilliseconds(100), .clock = .awake } }) catch {};
            if (std.Io.Clock.awake.now(g.io).toNanoseconds() > deadline) {
                if (secure) g.writers -= 1 else g.ordinary_waiters -= 1;
                g.changed.broadcast(g.io);
                g.mutex.unlock(g.io);
                return error.SecureAdmissionTimeout;
            }
        }
        if (secure) g.writers -= 1 else g.ordinary_waiters -= 1;
        if (g.broken) { g.mutex.unlock(g.io); return error.SecureUnavailable; }
        if (secure) g.exclusive = true else g.readers += 1;
        g.mutex.unlock(g.io);
        if (secure) g.engine.secureReset(true) catch |err| {
            if (err == error.SecurePreparationRefused) {
                // The backend promises that every participating rank rolled back
                // before returning this one error. No private request was submitted.
                g.mutex.lockUncancelable(g.io);
                g.exclusive = false;
                g.changed.broadcast(g.io);
                g.mutex.unlock(g.io);
                return error.SecurePreparationRefused;
            }
            g.poison();
            return error.SecureUnavailable;
        };
    }
    fn poison(g: *Gate) void {
        g.mutex.lockUncancelable(g.io);
        g.broken = true;
        g.changed.broadcast(g.io);
        g.mutex.unlock(g.io);
    }
    pub fn leave(g: *Gate, secure: bool) void {
        if (g.shared) {
            g.mutex.lockUncancelable(g.io);
            defer g.mutex.unlock(g.io);
            if (secure) g.private_readers -= 1 else g.readers -= 1;
            return;
        }
        if (secure) g.engine.secureReset(false) catch { g.poison(); return; };
        g.mutex.lockUncancelable(g.io);
        if (secure) g.exclusive = false else g.readers -= 1;
        g.changed.broadcast(g.io);
        g.mutex.unlock(g.io);
    }
};

const Fixture = struct {
    calls: usize = 0, phases: [8]bool = @splat(false), fail_at: ?usize = null, refuse: bool = false,
    fn engine(f: *Fixture) api.Engine { return .{ .ctx = f, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory, .secure_reset = reset } }; }
    fn info(_: *anyopaque) api.Info { return .{}; }
    fn submit(_: *anyopaque, _: api.Id, _: *const api.Request, _: api.Sink) api.SubmitError!void { return error.Closed; }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void { out.* = .{}; }
    fn memory(_: *anyopaque, _: bool) ?api.Memory { return null; }
    fn reset(p: *anyopaque, entering: bool) !void {
        const f: *Fixture = @ptrCast(@alignCast(p));
        if (entering and f.refuse) return error.SecurePreparationRefused;
        if (f.fail_at == f.calls) return error.SyntheticCleanupFailure;
        f.phases[f.calls] = entering;
        f.calls += 1;
    }
};

test "secure verified preparation refusal leaves ordinary admission usable" {
    var f: Fixture = .{ .refuse = true };
    var g: Gate = .{ .io = std.testing.io, .engine = f.engine() };
    try std.testing.expectError(error.SecurePreparationRefused, g.enter(true));
    try std.testing.expect(!g.broken and !g.exclusive);
    try g.enter(false);
    g.leave(false);
    try std.testing.expectEqual(@as(usize, 0), f.calls);
}

test "secure gate closes ordinary admission after either cleanup failure" {
    for ([_]usize{0, 1}) |fail_at| {
        var f: Fixture = .{ .fail_at = fail_at };
        var g: Gate = .{ .io = std.testing.io, .engine = f.engine() };
        if (fail_at == 0) try std.testing.expectError(error.SecureUnavailable, g.enter(true)) else {
            try g.enter(true);
            g.leave(true);
        }
        try std.testing.expectError(error.SecureUnavailable, g.enter(false));
        try std.testing.expectError(error.SecureUnavailable, g.enter(true));
    }
}

test "secure gate drains ordinary work then completes both resets" {
    var f: Fixture = .{};
    var g: Gate = .{ .io = std.testing.io, .engine = f.engine() };
    try g.enter(false);
    const Worker = struct {
        gate: *Gate,
        entered: std.atomic.Value(bool) = .init(false),
        release: std.atomic.Value(bool) = .init(false),
        failed: std.atomic.Value(bool) = .init(false),
        fn run(w: *@This()) void {
            w.gate.enter(true) catch { w.failed.store(true, .release); return; };
            w.entered.store(true, .release);
            while (!w.release.load(.acquire)) std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
            w.gate.leave(true);
        }
    };
    var w: Worker = .{ .gate = &g };
    const thread = try std.Thread.spawn(.{}, Worker.run, .{&w});
    defer { w.release.store(true, .release); thread.join(); }
    var waiting = false;
    for (0..1000) |_| {
        g.mutex.lockUncancelable(g.io);
        waiting = g.writers > 0;
        g.mutex.unlock(g.io);
        if (waiting) break;
        try std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake);
    }
    // Release even if the timing assertion fails so the test cannot strand its thread.
    const premature = w.entered.load(.acquire);
    // A waiting secure request must not make a newly arriving ordinary turn wait.
    // Keep the first reader until the second is admitted: the writer cannot win a race.
    const Reader = struct {
        gate: *Gate,
        entered: std.atomic.Value(bool) = .init(false),
        fn run(reader: *@This()) void {
            reader.gate.enter(false) catch return;
            reader.entered.store(true, .release);
            reader.gate.leave(false);
        }
    };
    var reader: Reader = .{ .gate = &g };
    const reader_thread = try std.Thread.spawn(.{}, Reader.run, .{&reader});
    for (0..1000) |_| {
        if (reader.entered.load(.acquire)) break;
        try std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake);
    }
    const ordinary_passed = reader.entered.load(.acquire);
    g.leave(false); // always drain before asserting, including a regression
    if (!ordinary_passed) w.release.store(true, .release);
    reader_thread.join();
    try std.testing.expect(ordinary_passed);
    try std.testing.expect(waiting and !premature);
    for (0..1000) |_| {
        if (w.entered.load(.acquire) or w.failed.load(.acquire)) break;
        try std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake);
    }
    try std.testing.expect(w.entered.load(.acquire) and !w.failed.load(.acquire));
    w.release.store(true, .release);
    // Taking ordinary admission again must wait for the post-request reset.
    try g.enter(false);
    g.leave(false);
    try std.testing.expectEqual(@as(usize, 2), f.calls);
    try std.testing.expect(f.phases[0] and !f.phases[1]);
}


test "shared admission permits regular and secure requests together without reset" {
    var f: Fixture = .{};
    var g: Gate = .{ .io = std.testing.io, .engine = f.engine(), .shared = true };
    try g.enter(false);
    try g.enter(true);
    try g.enter(false);
    try g.enter(false);
    try std.testing.expectEqual(@as(usize, 3), g.readers);
    try std.testing.expectEqual(@as(usize, 1), g.private_readers);
    try std.testing.expect(!g.exclusive);
    try std.testing.expectEqual(@as(usize, 0), f.calls);
    g.leave(true);
    for (0..3) |_| g.leave(false);
    try std.testing.expectEqual(@as(usize, 0), g.readers + g.private_readers);
    g.poison();
    try std.testing.expectError(error.SecureUnavailable, g.enter(true));
    try std.testing.expectError(error.SecureUnavailable, g.enter(false));
}
