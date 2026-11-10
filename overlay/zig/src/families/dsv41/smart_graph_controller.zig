// Modified for the DeepSeek 1017 TP3 integration; upstream copyrights retained. See CREDITS.md and provenance.json.
//! Smart CUDA Graph Controller 4.0.3. Single model-owner thread, CPU metadata.
//! All-rank capture admission and explicit upload follow TensorFold's
//! graph_cache contract (ba7b2c4, Apache-2.0); no upstream code is vendored.
const std = @import("std");
pub const policy = @import("smart_graph_policy.zig");
const wire = @import("smart_graph_protocol.zig");
const decision = @import("smart_graph_decision.zig");
const state = @import("smart_graph_state.zig");
const memory = @import("graph_retention.zig");
const testing = @import("builtin").is_test;
const link = if (testing) @import("smart_graph_fake_link.zig") else @import("link.zig");
const build = if (testing) struct {
    pub const fingerprint: u64 = 0x403;
} else @import("smart_graph_build.zig");
pub const Group = policy.Group;
pub const Leaf = policy.Leaf;

extern "c" fn rename([*:0]const u8, [*:0]const u8) c_int;
extern "c" fn fsync(c_int) c_int;
extern "c" fn getpagesize() c_int;
extern "c" fn malloc_trim(usize) c_int;

fn number(comptime name: [:0]const u8, fallback: u64) !u64 {
    const raw = std.c.getenv(name) orelse return fallback;
    return std.fmt.parseInt(u64, std.mem.span(raw), 10) catch error.BadGraphConfiguration;
}
fn flag(comptime name: [:0]const u8, fallback: bool) !bool {
    const v = try number(name, @intFromBool(fallback));
    if (v > 1) return error.BadGraphConfiguration;
    return v == 1;
}
pub fn config() !policy.Config {
    const enabled = try flag("TF_DS_SMART_GRAPH", false);
    if (!enabled) return .{}; // old/zero-valued experimental controls are inert
    const c: policy.Config = .{
        .enabled = enabled,
        .reserve = std.math.mul(u64, try number("TF_DS_GRAPH_RESERVE_MIB", 1024), policy.mib) catch return error.BadGraphConfiguration,
        .trim = try flag("TF_DS_GRAPH_TRIM_HOST_HEAP", true),
        .limit = std.math.cast(u32, try number("TF_DS_SMART_GRAPH_LIMIT", policy.max_resident)) orelse return error.BadGraphConfiguration,
        .poll_primitives = std.math.cast(u32, try number("TF_DS_SMART_GRAPH_POLL", 128)) orelse return error.BadGraphConfiguration,
    };
    if (!c.valid()) return error.BadGraphConfiguration;
    return c;
}
fn rssBytes() u64 {
    if (testing) return @import("smart_graph_fake_link.zig").rss;
    const fd = std.c.open("/proc/self/statm", .{ .ACCMODE = .RDONLY, .CLOEXEC = true });
    if (fd < 0) return 0;
    defer _ = std.c.close(fd);
    var b: [256]u8 = undefined;
    const n = std.c.read(fd, &b, b.len);
    if (n <= 0) return 0;
    var it = std.mem.tokenizeAny(u8, b[0..@intCast(n)], " \r\n");
    _ = it.next();
    const pages = std.fmt.parseInt(u64, it.next() orelse return 0, 10) catch return 0;
    // Linux runtime page size, not a hardcoded 4 KiB assumption on aarch64.
    const page = getpagesize();
    if (page <= 0) return 0;
    return std.math.mul(u64, pages, @intCast(page)) catch 0;
}
fn availableBytes() u64 {
    if (testing) return @import("smart_graph_fake_link.zig").available;
    return memory.availableBytes() orelse 0;
}
pub const Runtime = struct {
    cfg: policy.Config,
    digest: u64,
    rank: u32,
    world: u32,
    peers: [3]?link.Link = @splat(null),
    ledger: ?policy.Ledger = null,
    owner: *anyopaque,
    retire_fn: *const fn (*anyopaque, Group) anyerror!void,
    mask_fn: *const fn (*anyopaque, Group) u64,
    serial: u64 = 0,
    failed: bool = false,
    timing_clean: bool = true,
    before_available: u64 = 0,
    before_rss: u64 = 0,
    capture_started: u64 = 0,
    pending: ?Leaf = null,
    recovery_at: u64 = 0,
    recovery_before: u64 = 0,
    last_trim_ns: u64 = 0,
    last_report_ns: u64 = 0,
    last_save_ns: u64 = 0,
    stats_path: ?[:0]const u8 = null,
    persistence: ?*PersistJob = null,
    persist_thread: ?std.Thread = null,
    persist_prepare_ns: u64 = 0,

    pub fn open(gpa: std.mem.Allocator, c: policy.Config, profile: u64, rank: u32, world: u32, peers: []const ?link.Link, owner: *anyopaque, retire_fn: *const fn (*anyopaque, Group) anyerror!void, mask_fn: *const fn (*anyopaque, Group) u64) !Runtime {
        if (world == 0 or world > 4 or rank >= world or peers.len < (if (rank == 0) world - 1 else 1)) return error.BadGraphTopology;
        var profile_id: u64 = 0;
        if (!testing and c.enabled) {
            const raw = std.c.getenv("TF_DS_GRAPH_PROFILE_ID") orelse return error.MissingGraphModelProfile;
            const text = std.mem.span(raw);
            if (text.len == 0 or text.len > 512) return error.BadGraphModelProfile;
            profile_id = std.hash.Wyhash.hash(0x403, text);
        }
        var s: Runtime = .{ .cfg = c, .digest = c.digest(profile ^ profile_id ^ build.fingerprint, world), .rank = rank, .world = world, .owner = owner, .retire_fn = retire_fn, .mask_fn = mask_fn };
        for (peers[0..@min(3, peers.len)], 0..) |p, i| s.peers[i] = p;
        if (c.enabled) s.ledger = policy.Ledger.init(gpa) catch null;
        errdefer if (s.ledger) |*l| l.deinit();
        errdefer if (s.persistence) |job| job.gpa.destroy(job);
        if (!testing and rank == 0 and s.ledger != null) {
            if (std.c.getenv("TF_DS_GRAPH_STATS_PATH")) |raw| {
                const path = std.mem.span(raw);
                if (path.len == 0 or path.len > 4000) return error.BadGraphStatsPath;
                s.stats_path = path;
                s.loadFile(path, false);
                s.persistence = gpa.create(PersistJob) catch null;
                if (s.persistence) |job| job.* = .{ .gpa = gpa, .path = path };
            }
            if (std.c.getenv("TF_DS_GRAPH_CALIBRATION_PATH")) |raw| s.loadFile(std.mem.span(raw), true);
        }
        var boot = s.frame(.boot, .{});
        boot.ok = !c.enabled or s.ledger != null;
        boot.cost_ns = if (s.ledger) |*l| l.clock else 0;
        const all = try s.exchange(boot);
        if (!all.ok) return error.GraphControllerInitFailed;
        if (s.ledger) |*l| l.clock = all.cost_ns;
        std.debug.print("smart_graph_boot version=4.0.3 rank={d} world={d} enabled={} digest={x} generation=0 history={d} max_resident={d} max_ghost={d}\n", .{ rank, world, c.enabled, s.digest, if (s.ledger) |*l| l.ghost.count() else 0, c.limit, policy.max_ghost });
        return s;
    }
    pub fn close(s: *Runtime) void {
        if (s.persist_thread) |t| {
            t.join();
            s.persist_thread = null;
        }
        if (s.ledger) |*l| {
            if (s.rank == 0) s.saveFile();
            l.deinit();
        }
        if (s.persist_thread) |t| {
            t.join();
            s.persist_thread = null;
        }
        if (s.persistence) |job| job.gpa.destroy(job);
    }
    pub fn poison(s: *Runtime) void {
        s.failed = true;
        s.timing_clean = false;
    }
    fn frame(s: *Runtime, phase: wire.Phase, key: Leaf) wire.Frame {
        return .{ .phase = phase, .rank = @intCast(s.rank), .digest = s.digest, .serial = s.serial, .generation = if (s.ledger) |*l| l.generation else 0, .inventory = if (s.ledger) |*l| l.inventory() else 0, .leaf = key };
    }
    fn exchange(s: *Runtime, mine: wire.Frame) !wire.Frame {
        errdefer s.poison();
        if (s.failed) return error.GraphControllerFailed;
        const started = link.Link.monotonicNs();
        const seconds: u64 = if (mine.phase == .boot) 120 else if (mine.phase == .captured) 30 else 5;
        const deadline = started + seconds * std.time.ns_per_s;
        var out: [wire.capacity]u8 = undefined;
        var input: [wire.capacity]u8 = undefined;
        var answer = mine;
        if (s.rank == 0) {
            for (s.peers[0 .. s.world - 1], 1..) |maybe, rank| {
                const peer = maybe orelse return error.BadGraphTopology;
                const f = try wire.Frame.decode(try peer.recvDeadline(&input, deadline));
                try wire.validate(mine, f, @intCast(rank));
                if (f.action != .keep or f.plan.n != 0 or f.trim) return error.BadGraphRequest;
                wire.aggregate(&answer, f);
            }
            if (mine.phase == .offer or mine.phase == .probe) {
                decision.choose(&s.ledger.?, s.cfg, &answer, s.recovery_at > s.ledger.?.clock);
                if (answer.action == .retire) answer.trim = s.cfg.trim and started -| s.last_trim_ns >= 5 * std.time.ns_per_s;
            } else if (!answer.ok) answer.action = .abort;
            const data = try answer.encode(&out);
            for (s.peers[0 .. s.world - 1]) |peer| try peer.?.sendDeadline(data, deadline);
        } else {
            const peer = s.peers[0] orelse return error.BadGraphTopology;
            try peer.sendDeadline(try mine.encode(&out), deadline);
            answer = try wire.Frame.decode(try peer.recvDeadline(&input, deadline));
            try wire.validate(mine, answer, 0);
        }
        s.serial +|= 1;
        if (s.ledger) |*l| l.control_ns +|= link.Link.monotonicNs() - started;
        if (answer.action == .abort) return error.GraphCaptureUnconfirmed;
        if (mine.phase == .offer and answer.action != .capture and answer.action != .eager and answer.action != .retire) return error.BadGraphDecision;
        if (mine.phase != .offer and mine.phase != .probe and answer.action != .keep) return error.BadGraphDecision;
        return answer;
    }
    fn observation(s: *Runtime, phase: wire.Phase, key: Leaf) wire.Frame {
        var f = s.frame(phase, key);
        f.available = availableBytes();
        f.rss = rssBytes();
        return f;
    }
    fn apply(s: *Runtime, plan: wire.Frame) !wire.Frame {
        if (plan.action != .retire) return plan;
        const l = &s.ledger.?;
        // Validate the COMPLETE plan before destroying even one executable.
        var ready = s.frame(.retire_ready, plan.leaf);
        for (plan.plan.keys[0..plan.plan.n]) |k| if (!l.resident.contains(k) or std.meta.eql(k, plan.leaf.group) or
            s.mask_fn(s.owner, k) != l.resident.get(k).?.leaves)
        {
            ready.ok = false;
        };
        _ = try s.exchange(ready); // unanimous owner validation BEFORE destruction
        const start = link.Link.monotonicNs();
        s.timing_clean = false;
        for (plan.plan.keys[0..plan.plan.n]) |k| {
            try s.retire_fn(s.owner, k);
            l.retire(k);
        }
        l.destroy_ns +|= link.Link.monotonicNs() - start;
        if (plan.trim) {
            const trim = link.Link.monotonicNs();
            if (!testing) _ = malloc_trim(1 << 20);
            l.trim_ns +|= link.Link.monotonicNs() - trim;
            s.last_trim_ns = trim;
        }
        l.generation +|= 1;
        s.recovery_before = plan.available;
        s.recovery_at = l.clock +| 4;
        const receipt = try s.exchange(s.observation(.retired, plan.leaf));
        std.debug.print("smart_graph_retire rank={d} generation={d} victims={d} available_before={d} available_immediate={d} delayed_recovery_pending=true destroy_total_ns={d} trim_total_ns={d} inventory={x}\n", .{ s.rank, l.generation, plan.plan.n, plan.available, receipt.available, l.destroy_ns, l.trim_ns, l.inventory() });
        return receipt;
    }
    /// Called once per verify/pass on ALL ranks, before kernels. No file or
    /// transport on normal ticks; at most one bounded poll per 128 primitives.
    pub fn boundary(s: *Runtime, verify: bool, protected: Group) !void {
        if (!s.cfg.enabled) return;
        errdefer s.poison();
        if (s.failed or s.pending != null) return error.GraphControllerFailed;
        if (verify) s.timing_clean = true;
        const l = &s.ledger.?;
        l.tick();
        const recovering = s.recovery_at != 0 and l.clock >= s.recovery_at;
        if (!recovering and l.clock % s.cfg.poll_primitives != 0) return;
        s.timing_clean = false;
        const all = try s.exchange(s.observation(.probe, .{ .group = protected }));
        l.polls +|= 1;
        if (recovering) {
            l.recovery_checks +|= 1;
            const recovered = all.available -| s.recovery_before;
            l.recovery_bytes +|= recovered;
            std.debug.print("smart_graph_recovery rank={d} generation={d} available_delayed={d} host_available_growth={d} attribution=noisy_host_measurement\n", .{ s.rank, l.generation, all.available, recovered });
            s.recovery_at = 0;
        }
        _ = try s.apply(all);
        if (s.rank == 0) {
            const now = link.Link.monotonicNs();
            if (now -| s.last_report_ns >= 60 * std.time.ns_per_s) {
                s.report(all.available, all.rss);
                s.last_report_ns = now;
            }
            if (now -| s.last_save_ns >= 60 * std.time.ns_per_s) {
                s.saveFile();
                s.last_save_ns = now;
            }
        }
    }
    pub fn use(s: *Runtime, key: Group) void {
        if (s.ledger) |*l| l.use(key);
    }
    pub fn hit(s: *Runtime) void {
        if (s.ledger) |*l| l.hit();
    }
    pub fn prepare(s: *Runtime, key: Leaf, host_ready: bool) !bool {
        if (!s.cfg.enabled) return true;
        errdefer s.poison();
        if (!key.valid() or key.group.kind == .none or s.pending != null) return error.BadGraphCapture;
        const l = &s.ledger.?;
        l.miss(key.group);
        s.timing_clean = false;
        if (s.mask_fn(s.owner, key.group) != (if (l.resident.get(key.group)) |r| r.leaves else 0)) return error.GraphOwnerMismatch;
        var offer = s.observation(.offer, key);
        offer.ok = host_ready;
        var all = try s.exchange(offer);
        if (all.action == .retire) {
            _ = try s.apply(all);
            // Fresh measured room, never add estimated freed bytes to RAM.
            // Keep the one-batch work budget: a second miss may run eagerly.
            // Economics may use head-only calibration; agree on the final
            // post-retirement action before entering collective capture.
            var confirm = s.observation(.offer, key);
            confirm.ok = host_ready;
            all = try s.exchange(confirm);
        }
        if (all.action != .capture) {
            l.eager +|= 1;
            return false;
        }
        if (!l.resident.contains(key.group) and l.resident.count() >= s.cfg.limit) return error.GraphMetadataFull;
        s.before_available = availableBytes();
        s.before_rss = rssBytes();
        s.capture_started = link.Link.monotonicNs();
        s.pending = key;
        return true;
    }
    pub fn captured(s: *Runtime, key: Leaf, ok: bool) !void {
        if (!s.cfg.enabled) return;
        errdefer s.poison();
        if (s.pending == null or !std.meta.eql(s.pending.?, key)) return error.BadGraphCapture;
        defer s.pending = null;
        const stopped = link.Link.monotonicNs();
        var f = s.observation(.captured, key);
        f.ok = ok;
        f.cost_ns = stopped - s.capture_started;
        const delta = @max(if (f.available > 0 and s.before_available > 0) s.before_available -| f.available else 0, if (f.rss > 0 and s.before_rss > 0) f.rss -| s.before_rss else 0);
        // Whole-host deltas can include unrelated load. Large outliers carry
        // no attribution confidence; the conservative prior remains active.
        f.bytes = if (delta <= 128 * policy.mib) delta else 0;
        const all = try s.exchange(f);
        if (!all.ok) return error.GraphCaptureUnconfirmed;
        const l = &s.ledger.?;
        l.captured(key, all.cost_ns, all.bytes);
        std.debug.print("smart_graph_capture rank={d} generation={d} kind={s} rows={d} bucket={d} steps={d} piece={d} greedy={} slowest_host_build_ns={d} marginal_bytes_estimate={d} byte_confident={} inventory={x}\n", .{ s.rank, l.generation, @tagName(key.group.kind), key.group.rows, key.group.bucket, key.group.steps, key.piece, key.greedy, all.cost_ns, all.bytes, all.bytes > 0, l.inventory() });
    }
    fn report(s: *Runtime, available: u64, rss_max: u64) void {
        const l = &s.ledger.?;
        std.debug.print("smart_graph_stats version=4.0.3 digest={x} generation={d} resident={d} ghost={d} hits={d} misses={d} captures={d} eager={d} evictions={d} ghost_returns={d} quick_returns={d} capture_host_ns={d} control_ns={d} polls={d} destroy_ns={d} trim_ns={d} available_min={d} reserve={d} recovery_checks={d} recovery_growth_noisy={d}\n", .{ s.digest, l.generation, l.resident.count(), l.ghost.count(), l.hits, l.misses, l.captures, l.eager, l.evictions, l.ghost_returns, l.quick_returns, l.capture_ns, l.control_ns, l.polls, l.destroy_ns, l.trim_ns, available, s.cfg.reserve, l.recovery_checks, l.recovery_bytes });
        if (s.persistence) |job| std.debug.print("smart_graph_persistence prepare_total_ns={d} writes={d} failures={d} busy={} asynchronous=true\n", .{ s.persist_prepare_ns, job.writes.load(.acquire), job.failures.load(.acquire), job.busy.load(.acquire) });
        std.debug.print("smart_graph_memory rss_max_rank={d} available_min_rank={d} host_deltas_are_estimates=true\n", .{ rss_max, available });
    }
    fn loadFile(s: *Runtime, path: [:0]const u8, calibration: bool) void {
        if (testing) return;
        const fd = std.c.open(path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true });
        if (fd < 0) return;
        defer _ = std.c.close(fd);
        var buf: [state.capacity + 1]u8 = undefined;
        var n: usize = 0;
        while (n < buf.len) {
            const got = std.c.read(fd, buf[n..].ptr, buf.len - n);
            if (got < 0) return;
            if (got == 0) break;
            n += @intCast(got);
        }
        if (n > state.capacity) return;
        if (calibration) state.loadCalibration(&s.ledger.?, s.digest, buf[0..n]) catch |err| {
            std.debug.print("smart_graph_calibration ignored={s}\n", .{@errorName(err)});
            return;
        } else state.restore(&s.ledger.?, s.digest, buf[0..n]) catch |err| {
            std.debug.print("smart_graph_history ignored={s}\n", .{@errorName(err)});
            return;
        };
    }
    fn saveFile(s: *Runtime) void {
        if (testing) return;
        const job = s.persistence orelse return;
        if (job.busy.load(.acquire)) return; // one bounded pending snapshot only
        if (s.persist_thread) |t| {
            t.join();
            s.persist_thread = null;
        }
        const started = link.Link.monotonicNs();
        const data = state.save(&s.ledger.?, s.digest, &job.bytes) catch return;
        job.len = data.len;
        job.busy.store(true, .release);
        // Use the native server's standard thread stack. A 256 KiB custom
        // stack can be smaller than glibc's loaded CUDA/Torch static TLS and
        // pthread_create then returns EINVAL (std.Thread treats it as a bug).
        // Stack address space is reserved; untouched pages do not become RSS.
        s.persist_thread = std.Thread.spawn(.{}, PersistJob.run, .{job}) catch blk: {
            _ = job.failures.fetchAdd(1, .monotonic);
            job.busy.store(false, .release);
            break :blk null;
        };
        s.persist_prepare_ns +|= link.Link.monotonicNs() - started;
    }
};

/// File I/O/fsync is never performed by the decode/model owner. The worker
/// sees an immutable bounded snapshot, no GPU objects or mutable Ledger.
const PersistJob = struct {
    gpa: std.mem.Allocator,
    path: [:0]const u8,
    bytes: [state.capacity]u8 = undefined,
    len: usize = 0,
    busy: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    writes: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    failures: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    fn run(job: *PersistJob) void {
        defer job.busy.store(false, .release);
        if (job.write()) |_| {
            _ = job.writes.fetchAdd(1, .monotonic);
        } else |_| {
            _ = job.failures.fetchAdd(1, .monotonic);
        }
    }
    fn write(job: *PersistJob) !void {
        if (testing) return;
        var temp_buf: [4096]u8 = undefined;
        const text = try std.fmt.bufPrint(temp_buf[0 .. temp_buf.len - 1], "{s}.tmp", .{job.path});
        temp_buf[text.len] = 0;
        const temp: [:0]const u8 = temp_buf[0..text.len :0];
        const fd = std.c.open(temp, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true, .CLOEXEC = true }, @as(std.c.mode_t, 0o600));
        if (fd < 0) return error.StatsOpenFailed;
        var closed = false;
        defer if (!closed) {
            _ = std.c.close(fd);
        };
        var at: usize = 0;
        while (at < job.len) {
            const n = std.c.write(fd, job.bytes[at..].ptr, job.len - at);
            if (n <= 0) return error.StatsWriteFailed;
            at += @intCast(n);
        }
        if (fsync(fd) != 0) return error.StatsSyncFailed;
        closed = true;
        if (std.c.close(fd) != 0) return error.StatsCloseFailed;
        if (rename(temp, job.path) != 0) return error.StatsRenameFailed;
    }
};
