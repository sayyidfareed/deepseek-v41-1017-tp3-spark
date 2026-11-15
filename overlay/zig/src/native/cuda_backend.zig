// Modified for the DeepSeek 1017 TP3 integration; upstream copyrights retained. See CREDITS.md and provenance.json.
//! CUDA context-binding forwarding table, mechanically extracted from cuda.zig.
const std = @import("std");
const lanes = @import("lanes");
const Allocator = std.mem.Allocator;

pub fn backend(comptime Host: type, comptime bind: fn (*anyopaque) *Host, h: *Host) lanes.backend.Backend {
    const v = h.inner.vtable;
    h.vtable = .{
        .prefill = struct {
            fn f(p: *anyopaque, s: *lanes.Stream) anyerror!void {
                const x = bind(p);
                return x.inner.vtable.prefill(x.inner.ptr, s);
            }
        }.f,
        .prefill_part = if (v.prefill_part != null) struct {
            fn f(p: *anyopaque, s: *lanes.Stream, chunks: usize) anyerror!bool {
                const x = bind(p);
                return x.inner.vtable.prefill_part.?(x.inner.ptr, s, chunks);
            }
        }.f else null,
        .first = struct {
            fn f(p: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
                const x = bind(p);
                return x.inner.vtable.first(x.inner.ptr, s, position);
            }
        }.f,
        .queue = struct {
            fn f(p: *anyopaque, s: *lanes.Stream, feed: lanes.backend.Feed, position: u64) anyerror!u64 {
                const x = bind(p);
                return x.inner.vtable.queue(x.inner.ptr, s, feed, position);
            }
        }.f,
        .read = struct {
            fn f(p: *anyopaque, handle: u64) anyerror!u32 {
                const x = bind(p);
                return x.inner.vtable.read(x.inner.ptr, handle);
            }
        }.f,
        .verify = struct {
            fn f(p: *anyopaque, w: []const lanes.backend.Window, out: []lanes.backend.Verified) anyerror!void {
                const x = bind(p);
                return x.inner.vtable.verify(x.inner.ptr, w, out);
            }
        }.f,
        .keep = struct {
            fn f(p: *anyopaque, w: []const lanes.backend.Window, paths: []const []const u32) anyerror!void {
                const x = bind(p);
                return x.inner.vtable.keep(x.inner.ptr, w, paths);
            }
        }.f,
        .draft = struct {
            fn f(p: *anyopaque, r: []const lanes.backend.DraftRequest) anyerror!void {
                const x = bind(p);
                return x.inner.vtable.draft(x.inner.ptr, r);
            }
        }.f,
        .unspeculate = if (v.unspeculate != null) struct {
            fn f(p: *anyopaque, s: *lanes.Stream) anyerror!void {
                const x = bind(p);
                return x.inner.vtable.unspeculate.?(x.inner.ptr, s);
            }
        }.f else null,
        .probabilities = if (v.probabilities != null) struct {
            fn f(p: *anyopaque, s: *lanes.Stream, out: []f64) anyerror!bool {
                const x = bind(p);
                return x.inner.vtable.probabilities.?(x.inner.ptr, s, out);
            }
        }.f else null,
        .tree = if (v.tree != null) struct {
            fn f(p: *anyopaque, s: *lanes.Stream, gpa: Allocator) anyerror!?lanes.stream.Held {
                const x = bind(p);
                return x.inner.vtable.tree.?(x.inner.ptr, s, gpa);
            }
        }.f else null,
        .alternatives = if (v.alternatives != null) struct {
            fn f(p: *anyopaque, s: *lanes.Stream, out: []lanes.backend.Alternative) anyerror!usize {
                const x = bind(p);
                return x.inner.vtable.alternatives.?(x.inner.ptr, s, out);
            }
        }.f else null,
        .redraw = if (v.redraw != null) struct {
            fn f(p: *anyopaque, r: lanes.backend.Redraw, allow: lanes.backend.Allow) anyerror!?u32 {
                const x = bind(p);
                return x.inner.vtable.redraw.?(x.inner.ptr, r, allow);
            }
        }.f else null,
        .private_cache_poll = if (v.private_cache_poll != null) struct {
            fn f(p: *anyopaque) anyerror!void {
                const x = bind(p);
                try x.inner.privateCachePoll();
            }
        }.f else null,
        .private_cache_forget = if (v.private_cache_forget != null) struct {
            fn f(p: *anyopaque, scope: ?*const [32]u8) anyerror!usize {
                const x = bind(p);
                return x.inner.privateForget(scope);
            }
        }.f else null,
        .release = struct {
            fn f(p: *anyopaque, s: *lanes.Stream) void {
                const x = bind(p);
                x.inner.vtable.release(x.inner.ptr, s);
            }
        }.f,
    };
    return .{ .ptr = h, .vtable = &h.vtable };
}
