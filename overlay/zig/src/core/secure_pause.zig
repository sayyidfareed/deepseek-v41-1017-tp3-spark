// Modified for the DeepSeek 1017 TP3 integration; upstream copyrights retained. See CREDITS.md and provenance.json.
//! Suspend an idle lane thread without destroying its ordinary core or warm cache.
//! All fields below are protected by that host's existing mutex.
const std = @import("std");

pub const Pause = struct {
    requested: std.atomic.Value(bool) = .init(false),
    acknowledged: bool = false,

    /// Caller already owns the HTTP gate. A real engine-thread acknowledgement is
    /// required; HTTP idleness or status counters alone cannot grant ownership.
    pub fn acquire(p: *Pause, h: anytype, timeout_ms: u32) !void {
        h.mutex.lockUncancelable(h.io);
        defer h.mutex.unlock(h.io);
        if (h.closing or h.thread == null or h.pair_state.load(.acquire) != 0) return error.SecurePauseUnavailable;
        if (p.requested.load(.acquire) or p.acknowledged) return error.SecurePauseAlreadyHeld;
        p.requested.store(true, .release);
        h.wake.broadcast(h.io);
        const deadline = std.Io.Clock.awake.now(h.io).toNanoseconds() + @as(i96, timeout_ms) * std.time.ns_per_ms;
        while (!p.acknowledged) {
            if (h.closing or h.pair_state.load(.acquire) != 0 or std.Io.Clock.awake.now(h.io).toNanoseconds() >= deadline) {
                p.requested.store(false, .release);
                h.wake.broadcast(h.io);
                return error.SecurePauseUnavailable;
            }
            h.wake.waitTimeout(h.io, &h.mutex, .{ .duration = .{ .raw = .fromMilliseconds(10), .clock = .awake } }) catch {};
        }
    }

    /// Only after all ranks completed secure cleanup. No lane/cache object is reset.
    pub fn release(p: *Pause, h: anytype) void {
        h.mutex.lockUncancelable(h.io);
        defer h.mutex.unlock(h.io);
        p.requested.store(false, .release);
        h.wake.broadcast(h.io);
        while (p.acknowledged) h.wake.waitUncancelable(h.io, &h.mutex);
    }

    /// Called by the engine thread between complete iterations. Existing work and
    /// ordinary spill writes finish naturally before the pause can be acknowledged.
    pub fn poll(p: *Pause, h: anytype) void {
        if (!p.requested.load(.acquire)) return;
        h.mutex.lockUncancelable(h.io);
        defer h.mutex.unlock(h.io);
        if (!p.requested.load(.acquire) or h.closing or h.pair_state.load(.acquire) != 0) return;
        if (h.queued.items.len != 0 or h.admitted.items.len != 0 or h.ready.items.len != 0 or
            h.loading.items.len != 0 or h.deferred.items.len != 0 or h.cancels.items.len != 0 or
            h.filling != null or h.passing != null or h.lone_job != null or h.core.activeCount() != 0) return;
        if (h.cache) |store| if (store.busy()) return;
        p.acknowledged = true;
        h.wake.broadcast(h.io);
        while (p.requested.load(.acquire) and !h.closing and h.pair_state.load(.acquire) == 0)
            h.wake.waitUncancelable(h.io, &h.mutex);
        p.acknowledged = false;
        h.wake.broadcast(h.io);
    }
};
