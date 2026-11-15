// Modified for the DeepSeek 1017 TP3 integration; upstream copyrights retained. See CREDITS.md and provenance.json.
//! Both listeners submit to the same lane host. A startup-only rank check runs
//! before either listener opens; individual secure requests never pause ordinary work.
const std = @import("std");
const api = @import("engine_api");
const cuda = @import("cuda");
const ds = @import("dsv41");

pub const Router = struct {
    ordinary: *api.LaneHost,
    ctx: *const cuda.Context,
    family: *anyopaque,
    ready: std.atomic.Value(bool) = .init(false),
    cache_enabled: std.atomic.Value(bool) = .init(false),

    pub fn close(r: *Router) !void {
        // The serving layer has already drained both listeners.
        r.ready.store(false, .release);
    }
    pub fn engine(r: *Router) api.Engine {
        return .{ .ctx = r, .vtable = &.{ .info = info, .secure_info = secureInfo,
            .secure_shared = true, .submit = submit, .cancel = cancel,
            .status = status, .memory = memory, .secure_reset = reset, .keepalive = keepalive,
            .secure_cache_enter = cacheEnter, .secure_cache_leave = cacheLeave, .secure_forget = cacheForget } };
    }
    fn self(p: *anyopaque) *Router { return @ptrCast(@alignCast(p)); }
    fn info(p: *anyopaque) api.Info { return self(p).ordinary.engine().info(); }
    fn secureInfo(p: *anyopaque) api.Info {
        var out = info(p);
        out.prompt_cache = self(p).cache_enabled.load(.acquire);
        out.vision = null;
        return out;
    }
    fn submit(p: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const r = self(p);
        if (request.secure_request and !r.ready.load(.acquire)) return error.Closed;
        return r.ordinary.engine().submit(id, request, sink);
    }
    fn cancel(p: *anyopaque, id: api.Id) void { self(p).ordinary.engine().cancel(id); }
    fn status(p: *anyopaque, out: *api.Status, tokens: []u32) void {
        // LaneHost excludes private requests from its ordinary public counters.
        self(p).ordinary.engine().status(out, tokens);
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        // Whole-process peaks cannot be attributed to ordinary requests alone.
        return null;
    }
    fn keepalive(p: *anyopaque) ?api.keepalive.Target {
        return self(p).ordinary.engine().keepaliveTarget();
    }
    fn cacheEnter(p: *anyopaque, ticket: *api.SecureCacheTicket) !void {
        const r = self(p);
        if (!r.ready.load(.acquire)) return error.SecureUnavailable;
        try r.ordinary.secure_cache.enter(r.ordinary, ticket);
    }
    fn cacheLeave(p: *anyopaque, ticket: *api.SecureCacheTicket) void {
        const r = self(p);
        r.ordinary.secure_cache.leave(r.ordinary, ticket);
    }
    fn cacheForget(p: *anyopaque, scope: ?*const [32]u8) !usize {
        const r = self(p);
        if (!r.ready.load(.acquire)) return error.SecureUnavailable;
        return r.ordinary.secure_cache.forget(r.ordinary, scope);
    }
    fn reset(p: *anyopaque, entering: bool) !void {
        const r = self(p);
        if (!entering) {
            if (!r.ready.load(.acquire)) return error.SecureUnavailable;
            return;
        }
        if (r.ready.load(.acquire)) return;
        // Startup only: serialize the existing host command stream before listeners
        // open. Shared Gate never invokes this callback per request.
        try r.ordinary.secure_pause.acquire(r.ordinary, 120000);
        defer r.ordinary.secure_pause.release(r.ordinary);
        try r.ctx.makeCurrent();
        try ds.native.sharedSecureCheck(r.family);
        r.cache_enabled.store(ds.native.sharedSecureCacheEnabled(r.family), .release);
        r.ready.store(true, .release);
    }
};

test "shared secure router retains ordinary capacity and refuses unqualified private submission" {
    var host: api.LaneHost = undefined;
    host.info_ = .{ .lanes = 12, .context_window = 1048576, .prompt_cache = true };
    var router: Router = .{ .ordinary = &host, .ctx = undefined, .family = undefined };
    const engine = router.engine();
    try std.testing.expect(engine.vtable.secure_shared);
    try std.testing.expectEqual(@as(u32, 12), engine.secureInfo().lanes);
    try std.testing.expectEqual(engine.info().context_window, engine.secureInfo().context_window);
    try std.testing.expect(engine.info().prompt_cache and !engine.secureInfo().prompt_cache);
    const request: api.Request = .{ .prompt = &.{1}, .max_tokens = 1, .secure_request = true };
    const sink: api.Sink = undefined; // readiness must refuse before forwarding or touching the sink
    try std.testing.expectError(error.Closed, engine.submit(1, &request, sink));
}
