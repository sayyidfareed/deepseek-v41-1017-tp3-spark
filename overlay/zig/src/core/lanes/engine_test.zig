// Modified for the DeepSeek 1017 TP3 integration; upstream copyrights retained. See CREDITS.md and provenance.json.
//! The round loop on the fake target: drafted == one-token rounds, shared rounds == solo, greedy and sampled.
const std = @import("std");
const Config = @import("config.zig").Config;
const Engine = @import("engine.zig").Engine;
const sm = @import("stream.zig");
const fake = @import("fake.zig");
const SuffixLookup = @import("proposer.zig").SuffixLookup;
const Sampling = @import("sampling.zig").Sampling;

const gpa = std.testing.allocator;

const Case = struct {
    prompt: []const u32,
    max_new: u32 = 40,
    sampling: ?Sampling = null,
    drafts: bool = true,
    think_budget: u32 = 0,
};

fn model() !Config {
    var costs: [16]@import("config.zig").Cost = undefined;
    for (&costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 5.0 + 0.8 * @as(f64, @floatFromInt(w)) };
    return Config.init(gpa, .{ .exact_width = 16, .gpu_tokens = true, .mtp = true, .speculate = true, .speculate_early = false, .drafts = 4, .window_costs = &costs, .mtp_step_ms = 0.5, .hidden_rows = true, .batch_rows = 32, .max_streams = 8, .draft_streams = true }, 16, 15);
}

/// Every case's emitted tokens, the cases admitted together and stepped until done.
fn run(cases: []const Case) ![][]u32 {
    var cfg = try model();
    defer cfg.deinit(gpa);
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    const streams = try gpa.alloc(sm.Stream, cases.len);
    defer gpa.free(streams);
    const proposers = try gpa.alloc(SuffixLookup, cases.len);
    defer gpa.free(proposers);
    for (cases, streams, proposers) |c, *s, *p| {
        p.* = try SuffixLookup.init(gpa, .{ .min_match = 4 });
        s.* = try sm.Stream.init(gpa, .{ .id = "s", .prompt = c.prompt, .max_new = c.max_new, .eos = &.{96}, .sampling = c.sampling, .drafts = c.drafts, .proposer = p.proposer(), .think_budget = c.think_budget, .think_close = &.{ 90, 91, 92 }, .think_end = 91 });
    }
    defer for (streams, proposers) |*s, *p| {
        s.deinit(gpa);
        p.deinit();
    };
    for (streams) |*s| try engine.addStream(s);
    while (engine.activeCount() > 0) try engine.step();
    const out = try gpa.alloc([]u32, cases.len);
    for (out, streams) |*o, *s| o.* = try gpa.dupe(u32, s.emitted());
    return out;
}

fn free(runs: [][]u32) void {
    for (runs) |r| gpa.free(r);
    gpa.free(runs);
}

const GuardCase = struct {
    prompt: []const u32,
    max_new: u32 = 400,
    drafts: bool = true,
    think_budget: u32 = 0,
    loop_guard: bool = true,
    think_open: ?bool = null,
    cycle_after: usize = 70,
    pattern: []const u32 = &.{ 11, 12, 13 },
    answer_cycles: bool = false,
};

const GuardOut = struct {
    emitted: []u32,
    reason: sm.Reason,
    period: ?u32,
    finished: bool,
};

fn runGuard(cases: []const GuardCase, step_limit: usize) ![]GuardOut {
    var cfg = try model();
    defer cfg.deinit(gpa);
    const c = cases[0];
    var target: fake.Fake = .{ .gpa = gpa, .cycle_after = c.cycle_after, .probe_prompt = c.prompt.len, .pattern = c.pattern, .answer_cycles = c.answer_cycles };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    const streams = try gpa.alloc(sm.Stream, cases.len);
    defer gpa.free(streams);
    const proposers = try gpa.alloc(SuffixLookup, cases.len);
    defer gpa.free(proposers);
    for (cases, streams, proposers) |item, *s, *p| {
        p.* = try SuffixLookup.init(gpa, .{ .min_match = 4 });
        s.* = try sm.Stream.init(gpa, .{ .id = "guard", .prompt = item.prompt, .max_new = item.max_new, .eos = &.{96}, .drafts = item.drafts, .proposer = p.proposer(), .think_budget = item.think_budget, .think_close = &.{ 90, 91, 92 }, .think_end = 91, .think_open = item.think_open, .loop_guard = item.loop_guard });
    }
    defer for (streams, proposers) |*s, *p| {
        s.deinit(gpa);
        p.deinit();
    };
    for (streams) |*s| try engine.addStream(s);
    var steps: usize = 0;
    while (engine.activeCount() > 0 and steps < step_limit) : (steps += 1) try engine.step();
    const out = try gpa.alloc(GuardOut, cases.len);
    for (out, streams) |*result, *s| result.* = .{ .emitted = try gpa.dupe(u32, s.emitted()), .reason = s.reason, .period = s.loop_period, .finished = s.finished };
    return out;
}

fn freeGuard(runs: []GuardOut) void {
    for (runs) |run_result| gpa.free(run_result.emitted);
    gpa.free(runs);
}

const p1 = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5 };
const p2 = [_]u32{ 2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 5, 9 };

test "drafted rounds commit the one-token decode, greedy and sampled" {
    for ([_]?Sampling{ null, .{ .seed = 5, .temperature = 0.7, .top_k = 0, .top_p = 0.95 } }) |s| {
        const drafted = try run(&.{.{ .prompt = &p1, .sampling = s }});
        defer free(drafted);
        const plain = try run(&.{.{ .prompt = &p1, .sampling = s, .drafts = false }});
        defer free(plain);
        try std.testing.expectEqualSlices(u32, plain[0], drafted[0]);
        // and the fake target's own decode
        var history: std.ArrayList(u32) = .empty;
        defer history.deinit(gpa);
        try history.appendSlice(gpa, &p1);
        for (drafted[0]) |t| {
            try std.testing.expectEqual(fake.next(history.items, s, history.items.len), t);
            try history.append(gpa, t);
        }
    }
}

test "shared rounds commit what each stream commits alone" {
    const sampled: Sampling = .{ .seed = 9, .temperature = 1.0, .top_k = 0, .top_p = 0.9 };
    const together = try run(&.{ .{ .prompt = &p1 }, .{ .prompt = &p2, .sampling = sampled, .max_new = 30 } });
    defer free(together);
    const one = try run(&.{.{ .prompt = &p1 }});
    defer free(one);
    const two = try run(&.{.{ .prompt = &p2, .sampling = sampled, .max_new = 30 }});
    defer free(two);
    try std.testing.expectEqualSlices(u32, one[0], together[0]);
    try std.testing.expectEqualSlices(u32, two[0], together[1]);
}

test "the thinking budget's forced close is the same drafted, plain and shared" {
    const drafted = try run(&.{.{ .prompt = &p2, .think_budget = 9 }});
    defer free(drafted);
    const plain = try run(&.{.{ .prompt = &p2, .think_budget = 9, .drafts = false }});
    defer free(plain);
    const shared = try run(&.{ .{ .prompt = &p2, .think_budget = 9 }, .{ .prompt = &p1, .drafts = false } });
    defer free(shared);
    try std.testing.expectEqualSlices(u32, plain[0], drafted[0]);
    try std.testing.expectEqualSlices(u32, plain[0], shared[0]);
    try std.testing.expectEqual(@as(u32, 90), drafted[0][8]);
}

test "the loop guard matches drafted and plain, reports its period, and answers after close" {
    const plain = try runGuard(&.{.{ .prompt = &p2, .drafts = false }}, 5000);
    defer freeGuard(plain);
    const drafted = try runGuard(&.{.{ .prompt = &p2 }}, 5000);
    defer freeGuard(drafted);
    try std.testing.expectEqualSlices(u32, plain[0].emitted, drafted[0].emitted);
    try std.testing.expectEqual(@as(?u32, 3), drafted[0].period);
    try std.testing.expect(plain[0].finished and plain[0].reason == .stop);
    try std.testing.expectEqualSlices(u32, &.{ 90, 91, 92, 40, 41, 42, 96 }, plain[0].emitted[329..]);

    const off = try runGuard(&.{.{ .prompt = &p2, .loop_guard = false, .think_open = true }}, 5000);
    defer freeGuard(off);
    try std.testing.expect(off[0].finished and off[0].reason == .length and off[0].emitted.len == 400);
}

test "a cycle in the answer does not refire or reclose" {
    const plain = try runGuard(&.{.{ .prompt = &p2, .drafts = false, .max_new = 800, .answer_cycles = true }}, 5000);
    defer freeGuard(plain);
    const drafted = try runGuard(&.{.{ .prompt = &p2, .max_new = 800, .answer_cycles = true }}, 5000);
    defer freeGuard(drafted);
    try std.testing.expectEqualSlices(u32, plain[0].emitted, drafted[0].emitted);
    try std.testing.expectEqual(@as(?u32, 3), plain[0].period);
    try std.testing.expect(plain[0].finished and plain[0].reason == .length);
    var close_count: usize = 0;
    for (plain[0].emitted) |token| close_count += @intFromBool(token == 90);
    try std.testing.expectEqual(@as(usize, 1), close_count);
}

// Ported Laky actual-engine integration cases; retained target sampler and grammar.

const be = @import("backend.zig");
const AdaptiveClock = struct {
    ms: f64=0,
    fn start(ptr: *anyopaque) void { const c: *@This()=@ptrCast(@alignCast(ptr)); c.ms=0; }
    fn elapsed(ptr: *anyopaque, _: be.Mark) f64 { const c: *@This()=@ptrCast(@alignCast(ptr)); c.ms+=20; return c.ms; }
    fn clock(c: *@This()) be.Clock { return .{.ptr=c,.vtable=&.{.start=start,.elapsed_ms=elapsed}}; }
};
fn runAdaptiveCases(cases: []const Case, learning: bool, staggered: bool) ![][]u32 {
    var costs: [48]@import("config.zig").Cost = undefined;
    for (&costs, 1..) |*cost, width| cost.* = .{ .width=@intCast(width), .ms=5+@as(f64,@floatFromInt(width))*0.8 };
    var cfg=try Config.init(gpa,.{.adaptive_learning=learning,.exact_width=16,.gpu_tokens=true,.mtp=true,.speculate=true,.speculate_early=false,.drafts=4,.window_costs=&costs,.shared_costs=&costs,.mtp_step_ms=0.5,.hidden_rows=true,.batch_rows=48,.max_streams=12,.draft_streams=true},16,15);
    defer cfg.deinit(gpa);
    var target: fake.Fake=.{.gpa=gpa};
    defer target.deinit();
    var clock: AdaptiveClock=.{};
    var engine=Engine.init(gpa,&cfg,target.backend(),clock.clock());
    defer engine.deinit();
    const streams=try gpa.alloc(sm.Stream,cases.len);
    defer gpa.free(streams);
    const proposers=try gpa.alloc(SuffixLookup,cases.len);
    defer gpa.free(proposers);
    for (cases,streams,proposers) |case,*stream,*proposer| {
        proposer.*=try SuffixLookup.init(gpa,.{.min_match=4});
        stream.*=try sm.Stream.init(gpa,.{.id="adaptive-integration",.prompt=case.prompt,.max_new=case.max_new,.eos=&.{96},.sampling=case.sampling,.drafts=case.drafts,.proposer=proposer.proposer(),.think_budget=case.think_budget,.think_close=&.{90,91,92},.think_end=91});
    }
    defer for (streams,proposers) |*stream,*proposer| { stream.deinit(gpa); proposer.deinit(); };
    const first=if(staggered) @max(1,cases.len/2) else cases.len;
    for (streams[0..first]) |*stream| try engine.addStream(stream);
    var added=first;
    var steps: usize=0;
    while (engine.activeCount()>0 or added<cases.len) : (steps+=1) {
        if (added<cases.len and (steps>=3 or engine.activeCount()==0)) {
            for (streams[added..]) |*stream| try engine.addStream(stream);
            added=cases.len;
        }
        if (steps>1000) return error.AdaptiveIntegrationDidNotFinish;
        try engine.step();
    }
    if (learning) {
        try std.testing.expect(engine.learner!=null);
        try std.testing.expect(engine.learner.?.rounds>0);
        try std.testing.expect(engine.learner.?.acceptance_updates>0);
        // The fake target can reach EOS in its first two cold rounds. Such a
        // completion must not manufacture warm timing samples.
        if(engine.learner.?.rounds>2) {
            try std.testing.expect(engine.learner.?.cost_updates>0);
        } else try std.testing.expectEqual(@as(u64,0),engine.learner.?.cost_updates);
    } else try std.testing.expect(engine.learner==null);
    const out=try gpa.alloc([]u32,cases.len);
    for(out,streams) |*value,*stream| value.*=try gpa.dupe(u32,stream.emitted());
    return out;
}

test "active adaptive engine handles forced thinking closure mixed plain streams and early finish" {
    const cases=[_]Case{
        .{.prompt=&p2,.max_new=48,.think_budget=9},
        .{.prompt=&p1,.max_new=8,.drafts=false},
        .{.prompt=&p2,.max_new=1},
        .{.prompt=&p1,.max_new=52},
    };
    const old=try runAdaptiveCases(&cases,false,false);
    defer free(old);
    const new=try runAdaptiveCases(&cases,true,true);
    defer free(new);
    for(old,new) |a,b| try std.testing.expectEqualSlices(u32,a,b);
}

test "active adaptive engine matches original and serial for C1 through C12 greedy sampled and join leave" {
    for ([_]?Sampling{null,.{.seed=42,.temperature=0.7,.top_k=0,.top_p=0.95}}) |sampling| {
        var cases: [12]Case=undefined;
        for(1..13) |n| {
            for(cases[0..n],0..) |*case,i| case.*=.{.prompt=if(i%2==0) &p1 else &p2,.max_new=@intCast(32+i),.sampling=sampling};
            const reference=try runAdaptiveCases(cases[0..n],false,false);
            defer free(reference);
            const learned=try runAdaptiveCases(cases[0..n],true,true);
            defer free(learned);
            for(reference,learned,0..) |old,new,i| {
                try std.testing.expectEqualSlices(u32,old,new);
                const serial=try run(&.{.{.prompt=cases[i].prompt,.max_new=cases[i].max_new,.sampling=sampling,.drafts=false}});
                defer free(serial);
                try std.testing.expectEqualSlices(u32,serial[0],new);
            }
        }
    }
}
test "graph maintenance suppresses timing learning while preserving acceptance and output" {
    const cases = [_]Case{ .{ .prompt = &p1, .max_new = 64 }, .{ .prompt = &p2, .max_new = 58 } };
    for ([_]usize{ 1, 2 }) |n| {
        const reference = try runAdaptiveCases(cases[0..n], true, false);
        defer free(reference);
        var costs: [48]@import("config.zig").Cost = undefined;
        for (&costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 5 + @as(f64, @floatFromInt(w)) * 0.8 };
        var cfg = try Config.init(gpa, .{ .adaptive_learning = true, .exact_width = 16, .gpu_tokens = true, .mtp = true, .speculate = true, .speculate_early = false, .drafts = 4, .window_costs = &costs, .shared_costs = &costs, .mtp_step_ms = 0.5, .hidden_rows = true, .batch_rows = 48, .max_streams = 12, .draft_streams = true }, 16, 15);
        defer cfg.deinit(gpa);
        var target: fake.Fake = .{ .gpa = gpa };
        defer target.deinit();
        const baseline = target.backend();
        var vtable = baseline.vtable.*;
        const Dirty = struct {
            fn clean(_: *anyopaque) bool {
                return false;
            }
        };
        vtable.timing_clean = Dirty.clean;
        var clock: AdaptiveClock = .{};
        var engine = Engine.init(gpa, &cfg, .{ .ptr = baseline.ptr, .vtable = &vtable }, clock.clock());
        defer engine.deinit();
        var streams: [2]sm.Stream = undefined;
        var proposers: [2]SuffixLookup = undefined;
        for (cases[0..n], streams[0..n], proposers[0..n]) |c, *s, *pr| {
            pr.* = try SuffixLookup.init(gpa, .{ .min_match = 4 });
            s.* = try sm.Stream.init(gpa, .{ .id = "graph-timing", .prompt = c.prompt, .max_new = c.max_new, .eos = &.{96}, .drafts = true, .proposer = pr.proposer() });
        }
        defer for (streams[0..n], proposers[0..n]) |*s, *pr| {
            s.deinit(gpa);
            pr.deinit();
        };
        for (streams[0..n]) |*s| try engine.addStream(s);
        var steps: usize = 0;
        while (engine.activeCount() > 0) : (steps += 1) {
            if (steps > 1000) return error.DidNotFinish;
            try engine.step();
        }
        const learned = engine.learner orelse return error.MissingLearner;
        try std.testing.expectEqual(@as(u64, 0), learned.cost_updates);
        try std.testing.expect(learned.acceptance_updates > 0);
        for (reference, streams[0..n]) |expected, *s| try std.testing.expectEqualSlices(u32, expected, s.emitted());
    }
}

test "clean verify then late draft qualification skips finished learning and normal rounds resume" {
    const Gate = struct {
        threadlocal var base: @import("backend.zig").Backend = undefined;
        threadlocal var clean_sample: bool = true;
        threadlocal var verifies: usize = 0;
        threadlocal var dirtied: bool = false;
        fn clean(_: *anyopaque) bool { return clean_sample; }
        fn verify(ptr: *anyopaque, windows: []const @import("backend.zig").Window, out: []@import("backend.zig").Verified) !void {
            clean_sample = true;
            verifies += 1;
            try base.vtable.verify(ptr, windows, out);
        }
        fn draft(ptr: *anyopaque, requests: []const @import("backend.zig").DraftRequest) !void {
            try base.vtable.draft(ptr, requests);
            // A new native (N,steps) oracle is first needed AFTER clean verify.
            if (verifies == 3) { clean_sample = false; dirtied = true; }
        }
    };
    for ([_]usize{ 1, 2 }) |n| {
        // These fake-model prompts avoid early EOS, so late qualification and
        // subsequent steady-state learning both actually execute.
        const cases = [_]Case{ .{ .prompt = &.{1}, .max_new = 128 }, .{ .prompt = &.{4}, .max_new = 122 } };
        const reference = try runAdaptiveCases(cases[0..n], true, false);
        defer free(reference);
        var costs: [48]@import("config.zig").Cost = undefined;
        for (&costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 5 + @as(f64, @floatFromInt(w)) * 0.8 };
        var cfg = try Config.init(gpa, .{ .adaptive_learning = true, .exact_width = 16, .gpu_tokens = true, .mtp = true, .speculate = true, .speculate_early = false, .drafts = 4, .window_costs = &costs, .shared_costs = &costs, .mtp_step_ms = 0.5, .hidden_rows = true, .batch_rows = 48, .max_streams = 12, .draft_streams = true }, 16, 15);
        defer cfg.deinit(gpa);
        var target: fake.Fake = .{ .gpa = gpa };
        defer target.deinit();
        Gate.base = target.backend(); Gate.verifies = 0; Gate.clean_sample = true; Gate.dirtied = false;
        var vtable = Gate.base.vtable.*;
        vtable.verify = Gate.verify; vtable.draft = Gate.draft; vtable.timing_clean = Gate.clean;
        var clock: AdaptiveClock = .{};
        var engine = Engine.init(gpa, &cfg, .{ .ptr = Gate.base.ptr, .vtable = &vtable }, clock.clock());
        defer engine.deinit();
        var streams: [2]sm.Stream = undefined;
        var proposers: [2]SuffixLookup = undefined;
        for (cases[0..n], streams[0..n], proposers[0..n]) |c, *s, *pr| {
            pr.* = try SuffixLookup.init(gpa, .{ .min_match = 4 });
            s.* = try sm.Stream.init(gpa, .{ .id = "late-oracle", .prompt = c.prompt, .max_new = c.max_new, .eos = &.{96}, .drafts = true });
        }
        defer for (streams[0..n], proposers[0..n]) |*s, *pr| { s.deinit(gpa); pr.deinit(); };
        for (streams[0..n]) |*s| try engine.addStream(s);
        var dirty_seen = false;
        var clean_learning_resumed = false;
        var steps: usize = 0;
        while (engine.activeCount() > 0) : (steps += 1) {
            if (steps > 1000) return error.DidNotFinish;
            const learner = engine.learner orelse { try engine.step(); continue; };
            const before = learner.overhead;
            const acceptance = learner.acceptance_updates;
            Gate.dirtied = false;
            try engine.step();
            if (Gate.dirtied) {
                dirty_seen = true;
                try std.testing.expect(!learner.frame.warm);
                try std.testing.expect(std.meta.eql(before, learner.overhead));
                try std.testing.expect(learner.acceptance_updates > acceptance);
            } else if (dirty_seen and !std.meta.eql(before, learner.overhead)) clean_learning_resumed = true;
        }
        std.debug.print("late_oracle_regression streams={d} verifies={d} steps={d} dirty={} resumed={}\n",.{n,Gate.verifies,steps,dirty_seen,clean_learning_resumed});
        try std.testing.expect(dirty_seen);
        try std.testing.expect(clean_learning_resumed);
        for (reference, streams[0..n]) |expected, *s| try std.testing.expectEqualSlices(u32, expected, s.emitted());
    }
}

test "a cut-in needs picture rows only for a picture ending past its resume point, as dsv41's pictureNeedsRows" {
    const img = [_]sm.Image{.{ .at = 4, .tokens = 6, .bytes = "" }};
    try std.testing.expect(!Engine.picturesPast(&img, 10)); // ends exactly at the resume point: in the kept state
    try std.testing.expect(!Engine.picturesPast(&img, 12)); // wholly before
    try std.testing.expect(Engine.picturesPast(&img, 9)); // straddles it
    try std.testing.expect(Engine.picturesPast(&img, 0)); // a fresh pass
    try std.testing.expect(!Engine.picturesPast(&.{}, 0));
}
