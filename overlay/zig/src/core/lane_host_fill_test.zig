// Modified for the DeepSeek 1017 TP3 integration; upstream copyrights retained. See CREDITS.md and provenance.json.
//! Late admission and per-stream prompt state through the served lane host.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const gpa = std.testing.allocator;

const Action = enum { none, cancel_long, cancel_short, refuse_short, full };

const Seen = struct {
    action: Action = .none,
    engine: api.Engine = undefined,
    short: *const api.Request,
    long_chunks: usize = 0,
    short_at: ?usize = null,
    submitted: bool = false,
    tokens: [2]std.ArrayList(u32) = @splat(.empty),
    reasons: [2]?api.Reason = .{ null, null },
    done: std.atomic.Value(u32) = .init(0),

    fn hook(ctx: *anyopaque, s: *lanes.Stream, _: usize) void {
        const x: *Seen = @ptrCast(@alignCast(ctx));
        if (s.prompt_len != 32) return;
        x.long_chunks += 1;
        if (x.long_chunks == 1) switch (x.action) {
            .cancel_long => x.engine.cancel(1),
            .cancel_short => x.engine.cancel(2),
            else => {},
        };
        if (!x.submitted) {
            x.submitted = true;
            x.engine.submit(2, x.short, .{ .ctx = x, .event = event }) catch unreachable;
        }
    }

    fn event(ctx: *anyopaque, id: api.Id, e: *const api.Event) void {
        const x: *Seen = @ptrCast(@alignCast(ctx));
        const i: usize = @intCast(id - 1);
        switch (e.*) {
            .tokens => |t| {
                if (id == 2 and x.short_at == null) x.short_at = x.long_chunks;
                x.tokens[i].appendSlice(gpa, t) catch unreachable;
            },
            .finished => |f| {
                x.reasons[i] = f.reason;
                _ = x.done.fetchAdd(1, .release);
            },
            else => {},
        }
    }

    fn wait(x: *Seen) !void {
        for (0..2000) |_| {
            if (x.done.load(.acquire) == 2) return;
            try std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake);
        }
        return error.RequestsDidNotFinish;
    }

    fn deinit(x: *Seen) void {
        for (&x.tokens) |*t| t.deinit(gpa);
    }
};

fn reference(cfg: *const lanes.Config, prompt: []const u32) ![]u32 {
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, cfg, target.backend(), clock.clock());
    defer core.deinit();
    var s = try lanes.Stream.init(gpa, .{ .id = "reference", .prompt = prompt, .max_new = 8, .drafts = false });
    defer s.deinit(gpa);
    try core.addStream(&s);
    while (core.activeCount() > 0) try core.step();
    return gpa.dupe(u32, s.emitted());
}

test "a late short request completes before another prompt finishes filling, with the same tokens as solo" {
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    const long: [32]u32 = @splat(3);
    const short = [_]u32{ 1, 2, 3 };
    const short_request: api.Request = .{ .prompt = &short, .max_tokens = 8, .drafts = false };
    var seen: Seen = .{ .short = &short_request };
    defer seen.deinit();
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 8, .prefill_hook = Seen.hook, .prefill_hook_ctx = &seen };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.chunkedBackend(), clock.clock());
    defer core.deinit();
    core.bounded_admission = true;
    var host = api.LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    seen.engine = host.engine();
    try host.start();
    defer host.stop();
    const request: api.Request = .{ .prompt = &long, .max_tokens = 8, .drafts = false };
    try seen.engine.submit(1, &request, .{ .ctx = &seen, .event = Seen.event });
    try seen.wait();
    try std.testing.expect(seen.short_at.? < 8);
    try std.testing.expectEqual(api.Reason.length, seen.reasons[0].?);
    try std.testing.expectEqual(api.Reason.length, seen.reasons[1].?);
    const want_long = try reference(&cfg, &long);
    defer gpa.free(want_long);
    const want_short = try reference(&cfg, &short);
    defer gpa.free(want_short);
    try std.testing.expectEqualSlices(u32, want_long, seen.tokens[0].items);
    try std.testing.expectEqualSlices(u32, want_short, seen.tokens[1].items);
}

test "partial admission releases cancelled prompts, isolates refusal, and retries a full pool with another filler" {
    for ([_]Action{ .cancel_long, .cancel_short, .refuse_short, .full }) |action| {
        var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
        defer cfg.deinit(gpa);
        const long: [32]u32 = @splat(3);
        const short = [_]u32{ 1, 2, 3 };
        const short_request: api.Request = .{
            .prompt = &short,
            .max_tokens = 8,
            .drafts = false,
            .sampling = if (action == .refuse_short) .{ .temperature = 1, .seed = 7 } else null,
        };
        var seen: Seen = .{ .short = &short_request, .action = action, .submitted = true };
        defer seen.deinit();
        var target: lanes.fake.Fake = .{
            .gpa = gpa,
            .prefill_hook = Seen.hook,
            .prefill_hook_ctx = &seen,
            .refuse_sampled = action == .refuse_short,
            .refuse_full = if (action == .full) 1 else 0,
        };
        defer target.deinit();
        var clock: lanes.fake.FixedClock = .{};
        var core = lanes.Engine.init(gpa, &cfg, target.chunkedBackend(), clock.clock());
        defer core.deinit();
        core.bounded_admission = true;
        var host = api.LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
        seen.engine = host.engine();
        const request: api.Request = .{ .prompt = &long, .max_tokens = 8, .drafts = false };
        try seen.engine.submit(1, &request, .{ .ctx = &seen, .event = Seen.event });
        try seen.engine.submit(2, &short_request, .{ .ctx = &seen, .event = Seen.event });
        try host.start();
        defer host.stop();
        try seen.wait();
        const want_long: api.Reason = if (action == .cancel_long) .cancelled else .length;
        const want_short: api.Reason = switch (action) {
            .cancel_short => .cancelled,
            .refuse_short => .failed,
            else => .length,
        };
        try std.testing.expectEqual(want_long, seen.reasons[0].?);
        try std.testing.expectEqual(want_short, seen.reasons[1].?);
        if (want_long == .length) {
            const want = try reference(&cfg, &long);
            defer gpa.free(want);
            try std.testing.expectEqualSlices(u32, want, seen.tokens[0].items);
        } else try std.testing.expectEqual(@as(usize, 0), seen.tokens[0].items.len);
        if (want_short == .length) {
            const want = try reference(&cfg, &short);
            defer gpa.free(want);
            try std.testing.expectEqualSlices(u32, want, seen.tokens[1].items);
        } else try std.testing.expectEqual(@as(usize, 0), seen.tokens[1].items.len);
        try std.testing.expectEqual(@as(usize, 0), target.lanes.count());
    }
}

test "four and eight mixed prompt fills keep independent histories while short replies decode" {
    const Batch = struct {
        tokens: [8]std.ArrayList(u32) = @splat(.empty),
        reasons: [8]?api.Reason = @splat(null),
        long_chunks: usize = 0,
        short_at: ?usize = null,
        done: std.atomic.Value(u32) = .init(0),

        fn hook(ctx: *anyopaque, s: *lanes.Stream, _: usize) void {
            const x: *@This() = @ptrCast(@alignCast(ctx));
            if (s.prompt_len == 32) x.long_chunks += 1;
        }
        fn event(ctx: *anyopaque, id: api.Id, e: *const api.Event) void {
            const x: *@This() = @ptrCast(@alignCast(ctx));
            const i: usize = @intCast(id - 1);
            switch (e.*) {
                .tokens => |t| {
                    if (i % 2 == 1 and x.short_at == null) x.short_at = x.long_chunks;
                    x.tokens[i].appendSlice(gpa, t) catch unreachable;
                },
                .finished => |f| {
                    x.reasons[i] = f.reason;
                    _ = x.done.fetchAdd(1, .release);
                },
                else => {},
            }
        }
    };
    for ([_]usize{ 4, 8 }) |count| {
        var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
        defer cfg.deinit(gpa);
        var seen: Batch = .{};
        defer for (&seen.tokens) |*t| t.deinit(gpa);
        var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_hook = Batch.hook, .prefill_hook_ctx = &seen };
        defer target.deinit();
        var clock: lanes.fake.FixedClock = .{};
        var core = lanes.Engine.init(gpa, &cfg, target.chunkedBackend(), clock.clock());
        defer core.deinit();
        core.bounded_admission = true;
        var host = api.LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = @intCast(count) });
        const e = host.engine();
        var prompts: [8][32]u32 = undefined;
        var requests: [8]api.Request = undefined;
        for (0..count) |i| {
            prompts[i] = @splat(@intCast(i + 1));
            requests[i] = .{ .prompt = prompts[i][0..if (i % 2 == 0) @as(usize, 32) else 3], .max_tokens = 8, .drafts = false };
            try e.submit(@intCast(i + 1), &requests[i], .{ .ctx = &seen, .event = Batch.event });
        }
        try host.start();
        defer host.stop();
        for (0..2000) |_| {
            if (seen.done.load(.acquire) == count) break;
            try std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake);
        }
        try std.testing.expectEqual(count, seen.done.load(.acquire));
        try std.testing.expect(seen.short_at.? < count / 2 * 8);
        for (0..count) |i| {
            try std.testing.expectEqual(api.Reason.length, seen.reasons[i].?);
            const want = try reference(&cfg, requests[i].prompt);
            defer gpa.free(want);
            try std.testing.expectEqualSlices(u32, want, seen.tokens[i].items);
        }
        try std.testing.expectEqual(@as(usize, 0), target.lanes.count());
    }
}

test "bounded cold fill gives streaming peers multiple decode rounds and takes their cancellation" {
    const Traffic = struct {
        engine: api.Engine = undefined,
        cold: *const api.Request,
        cancel_peer: bool,
        submitted: bool = false,
        cold_chunks: usize = 0,
        peer_tokens: usize = 0,
        at_first_chunk: usize = 0,
        at_second_chunk: usize = 0,
        peer_finished_at: usize = 0,
        contended_seen: bool = false,
        solo_seen: bool = false,
        reasons: [2]?api.Reason = .{ null, null },
        done: std.atomic.Value(u32) = .init(0),

        fn hook(ctx: *anyopaque, s: *lanes.Stream, _: usize) void {
            const x: *@This() = @ptrCast(@alignCast(ctx));
            if (s.prompt_len != 32) return;
            x.cold_chunks += 1;
            if (s.prefill_contended) x.contended_seen = true else x.solo_seen = true;
            if (x.cold_chunks == 1) x.at_first_chunk = x.peer_tokens;
            if (x.cold_chunks == 2) x.at_second_chunk = x.peer_tokens;
            std.Io.sleep(std.testing.io, .fromMilliseconds(3), .awake) catch unreachable;
        }
        fn event(ctx: *anyopaque, id: api.Id, e: *const api.Event) void {
            const x: *@This() = @ptrCast(@alignCast(ctx));
            switch (e.*) {
                .tokens => |t| if (id == 1) {
                    x.peer_tokens += t.len;
                    if (!x.submitted) {
                        x.submitted = true;
                        x.engine.submit(2, x.cold, .{ .ctx = x, .event = event }) catch unreachable;
                    }
                    if (x.cancel_peer and x.cold_chunks == 1 and x.peer_tokens >= x.at_first_chunk + 4)
                        x.engine.cancel(1);
                    std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch unreachable;
                },
                .finished => |f| {
                    x.reasons[@intCast(id - 1)] = f.reason;
                    if (id == 1) x.peer_finished_at = x.cold_chunks;
                    if (id == 2) x.engine.cancel(1);
                    _ = x.done.fetchAdd(1, .release);
                },
                else => {},
            }
        }
    };
    for ([_]bool{ false, true }) |cancel_peer| {
        var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
        defer cfg.deinit(gpa);
        const long: [32]u32 = @splat(3);
        const short = [_]u32{ 1, 2, 3 };
        const cold: api.Request = .{ .prompt = &long, .max_tokens = 8, .drafts = false };
        const peer: api.Request = .{ .prompt = &short, .max_tokens = 4096, .drafts = false };
        var seen: Traffic = .{ .cold = &cold, .cancel_peer = cancel_peer };
        var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 8, .prefill_hook = Traffic.hook, .prefill_hook_ctx = &seen };
        defer target.deinit();
        var clock: lanes.fake.FixedClock = .{};
        var core = lanes.Engine.init(gpa, &cfg, target.chunkedBackend(), clock.clock());
        defer core.deinit();
        core.bounded_admission = true;
        var host = api.LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
        host.interleave_ns = 20 * std.time.ns_per_ms;
        seen.engine = host.engine();
        try seen.engine.submit(1, &peer, .{ .ctx = &seen, .event = Traffic.event });
        try host.start();
        defer host.stop();
        for (0..3000) |_| {
            if (seen.done.load(.acquire) == 2) break;
            try std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake);
        }
        try std.testing.expectEqual(@as(u32, 2), seen.done.load(.acquire));
        try std.testing.expectEqual(api.Reason.length, seen.reasons[1].?);
        try std.testing.expect(seen.at_second_chunk >= seen.at_first_chunk + 4);
        try std.testing.expect(seen.contended_seen);
        try std.testing.expectEqual(api.Reason.cancelled, seen.reasons[0].?);
        if (cancel_peer) {
            try std.testing.expect(seen.solo_seen);
            try std.testing.expectEqual(@as(usize, 1), seen.peer_finished_at);
        } else try std.testing.expectEqual(@as(usize, 8), seen.peer_finished_at);
        try std.testing.expectEqual(@as(usize, 8), seen.cold_chunks);
        try std.testing.expectEqual(@as(usize, 0), target.lanes.count());
    }
}

test "image-bearing cold fill must not starve an already streaming peer" {
    const Traffic = struct {
        engine: api.Engine = undefined,
        cold: *const api.Request,
        cancel_peer: bool,
        submitted: bool = false,
        cold_chunks: usize = 0,
        peer_tokens: usize = 0,
        at_first_chunk: usize = 0,
        at_second_chunk: usize = 0,
        peer_finished_at: usize = 0,
        contended_seen: bool = false,
        solo_seen: bool = false,
        reasons: [2]?api.Reason = .{ null, null },
        done: std.atomic.Value(u32) = .init(0),

        fn hook(ctx: *anyopaque, s: *lanes.Stream, _: usize) void {
            const x: *@This() = @ptrCast(@alignCast(ctx));
            if (s.prompt_len != 32) return;
            x.cold_chunks += 1;
            if (s.prefill_contended) x.contended_seen = true else x.solo_seen = true;
            if (x.cold_chunks == 1) x.at_first_chunk = x.peer_tokens;
            if (x.cold_chunks == 2) x.at_second_chunk = x.peer_tokens;
            std.Io.sleep(std.testing.io, .fromMilliseconds(3), .awake) catch unreachable;
        }
        fn event(ctx: *anyopaque, id: api.Id, e: *const api.Event) void {
            const x: *@This() = @ptrCast(@alignCast(ctx));
            switch (e.*) {
                .tokens => |t| if (id == 1) {
                    x.peer_tokens += t.len;
                    if (!x.submitted) {
                        x.submitted = true;
                        x.engine.submit(2, x.cold, .{ .ctx = x, .event = event }) catch unreachable;
                    }
                    if (x.cancel_peer and x.cold_chunks == 1 and x.peer_tokens >= x.at_first_chunk + 4)
                        x.engine.cancel(1);
                    std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch unreachable;
                },
                .finished => |f| {
                    x.reasons[@intCast(id - 1)] = f.reason;
                    if (id == 1) x.peer_finished_at = x.cold_chunks;
                    if (id == 2) x.engine.cancel(1);
                    _ = x.done.fetchAdd(1, .release);
                },
                else => {},
            }
        }
    };
    for ([_]bool{ false, true }) |cancel_peer| {
        var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
        defer cfg.deinit(gpa);
        const long: [32]u32 = @splat(3);
        const short = [_]u32{ 1, 2, 3 };
        const images = [_]api.Image{.{ .at = 1, .tokens = 1, .bytes = "synthetic", .rows = true }};
        const cold: api.Request = .{ .prompt = &long, .images = &images, .max_tokens = 8, .drafts = false };
        const peer: api.Request = .{ .prompt = &short, .max_tokens = 4096, .drafts = false };
        var seen: Traffic = .{ .cold = &cold, .cancel_peer = cancel_peer };
        var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 8, .prefill_hook = Traffic.hook, .prefill_hook_ctx = &seen };
        defer target.deinit();
        var clock: lanes.fake.FixedClock = .{};
        var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
        defer core.deinit();
        core.bounded_admission = true;
        var host = api.LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
        host.interleave_ns = 20 * std.time.ns_per_ms;
        seen.engine = host.engine();
        try seen.engine.submit(1, &peer, .{ .ctx = &seen, .event = Traffic.event });
        try host.start();
        defer host.stop();
        for (0..3000) |_| {
            if (seen.done.load(.acquire) == 2) break;
            try std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake);
        }
        try std.testing.expectEqual(@as(u32, 2), seen.done.load(.acquire));
        try std.testing.expectEqual(api.Reason.length, seen.reasons[1].?);
        try std.testing.expect(seen.at_second_chunk >= seen.at_first_chunk + 4);
        try std.testing.expectEqual(core.bounded_admission, seen.contended_seen);
        try std.testing.expectEqual(api.Reason.cancelled, seen.reasons[0].?);
        if (cancel_peer) {
            try std.testing.expect(seen.solo_seen);
            try std.testing.expectEqual(@as(usize, 1), seen.peer_finished_at);
        } else try std.testing.expectEqual(@as(usize, 8), seen.peer_finished_at);
        try std.testing.expectEqual(@as(usize, 8), seen.cold_chunks);
        try std.testing.expectEqual(@as(usize, 0), target.lanes.count());
    }
}

test "bounded image fill keeps later image and text fills exclusive through completion or cancellation" {
    const ExclusiveSeen = struct {
        engine: api.Engine = undefined,
        cancel_first: bool,
        chunks: [3]usize = .{ 0, 0, 0 },
        first_finished: bool = false,
        crossed: bool = false,
        reasons: [3]?api.Reason = .{ null, null, null },
        done: std.atomic.Value(u32) = .init(0),
        fn hook(ctx: *anyopaque, s: *lanes.Stream, _: usize) void {
            const x: *@This() = @ptrCast(@alignCast(ctx));
            const i: usize = switch (s.prompt_len) { 32 => 0, 40 => 1, 8 => 2, else => unreachable };
            x.chunks[i] += 1;
            if (i != 0 and !x.first_finished) x.crossed = true;
            if (i == 0 and x.cancel_first and x.chunks[0] == 3) x.engine.cancel(1);
        }
        fn event(ctx: *anyopaque, id: api.Id, e: *const api.Event) void {
            const x: *@This() = @ptrCast(@alignCast(ctx));
            switch (e.*) {
                .finished => |f| {
                    x.reasons[@intCast(id - 1)] = f.reason;
                    if (id == 1) x.first_finished = true;
                    _ = x.done.fetchAdd(1, .release);
                },
                else => {},
            }
        }
    };
    for ([_]bool{ false, true }) |cancel_first| {
        var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
        defer cfg.deinit(gpa);
        const first: [32]u32 = @splat(3);
        const second: [40]u32 = @splat(4);
        const text: [8]u32 = @splat(5);
        const images = [_]api.Image{.{ .at = 1, .tokens = 1, .bytes = "synthetic", .rows = true }};
        const requests = [_]api.Request{
            .{ .prompt = &first, .images = &images, .max_tokens = 1, .drafts = false },
            .{ .prompt = &second, .images = &images, .max_tokens = 1, .drafts = false },
            .{ .prompt = &text, .max_tokens = 1, .drafts = false },
        };
        var seen: ExclusiveSeen = .{ .cancel_first = cancel_first };
        var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 8, .prefill_hook = ExclusiveSeen.hook, .prefill_hook_ctx = &seen };
        defer target.deinit();
        var clock: lanes.fake.FixedClock = .{};
        var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
        defer core.deinit();
        core.bounded_admission = true;
        var host = api.LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 3 });
        seen.engine = host.engine();
        for (&requests, 1..) |*r, id| try seen.engine.submit(@intCast(id), r, .{ .ctx = &seen, .event = ExclusiveSeen.event });
        try host.start();
        defer host.stop();
        for (0..3000) |_| {
            if (seen.done.load(.acquire) == 3) break;
            try std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake);
        }
        try std.testing.expectEqual(@as(u32, 3), seen.done.load(.acquire));
        try std.testing.expect(!seen.crossed);
        try std.testing.expectEqual(if (cancel_first) api.Reason.cancelled else api.Reason.length, seen.reasons[0].?);
        try std.testing.expectEqual(api.Reason.length, seen.reasons[1].?);
        try std.testing.expectEqual(api.Reason.length, seen.reasons[2].?);
        try std.testing.expectEqual(@as(usize, if (cancel_first) 3 else 8), seen.chunks[0]);
        try std.testing.expectEqual(@as(usize, 8), seen.chunks[1]);
        try std.testing.expectEqual(@as(usize, 8), seen.chunks[2]);
        try std.testing.expectEqual(@as(usize, 0), target.lanes.count());
    }
}
