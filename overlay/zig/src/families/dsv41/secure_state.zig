// Modified for the DeepSeek 1017 TP3 integration; upstream copyrights retained. See CREDITS.md and provenance.json.
//! One request's private execution state. Immutable weights/kernels/RoPE are borrowed;
//! no ordinary KV, scratch, graph, Engram reader, lane or spill object is reused.
//! The caller owns a separately prepared communicator and the all-rank admission gate.
//! This is not a secure-serving capability by itself: process policy, private lanes,
//! rank agreement and library cleanup must also qualify before HTTP can use it.
const std = @import("std");
const cuda = @import("cuda");
const api = @import("engine_api");
const Model = @import("model.zig").Model;
const prompt = @import("prompt.zig");
const round = @import("round.zig");
const draft = @import("draft.zig");
const sampling = @import("sampling.zig");
const cublas = @import("cublas.zig");
const comm_mod = @import("comm.zig");
const prompt2d = @import("prompt2d.zig");
const engram_io = @import("engram_io.zig");

pub const Options = struct {
    window: usize,
    device_bytes: usize,
};

pub const State = struct {
    parent: std.mem.Allocator,
    zero: api.ZeroFree,
    m: Model,
    phase: enum { prepared, used, erased } = .prepared,
    blas_closed: bool = false,

    /// All ranks must finish local preparation before entering any collective.
    /// The communicator must be private, not the ordinary communicator. This function
    /// performs no NCCL exchange and never changes ordinary model/environment settings.
    pub fn prepare(parent: std.mem.Allocator, ordinary: *const Model, private_comm: *comm_mod.Comm, private_fan: prompt2d.Fan, o: Options) !*State {
        try validate(ordinary.world, ordinary.request_cap, ordinary.prefill_rows, o);
        try memoryAdmission(ordinary, o);
        if (ordinary.kvs != null or ordinary.two != null or ordinary.ring != null) return error.PrivateStateUnsupportedLayout;
        if (private_comm == &ordinary.comm or private_comm.comm == ordinary.comm.comm or
            private_comm.world != ordinary.world or private_comm.rank != ordinary.rank or
            private_comm.isAborted()) return error.PrivateStateNeedsOwnCommunicator;
        const s = try parent.create(State);
        errdefer parent.destroy(s);
        s.parent = parent;
        s.zero = .{ .parent = parent };
        s.phase = .prepared;
        s.blas_closed = false;
        const gpa = s.zero.allocator();
        const m = &s.m;
        m.* = .{
            .gpa = gpa,
            .host = std.heap.ArenaAllocator.init(gpa),
            .io = ordinary.io,
            .ctx = ordinary.ctx,
            .stream = undefined,
            .cfg = ordinary.cfg,
            .rank = ordinary.rank,
            .world = ordinary.world,
            .streams = 1,
            .round_ms = ordinary.round_ms,
            .draft_ms = ordinary.draft_ms,
            .prefill_rows = ordinary.prefill_rows,
            .pool_cap = o.window,
            .request_cap = o.window,
            .window_cap = o.window,
            .cache_dir = ordinary.cache_dir, // immutable model files; never close borrowed handles
            .cache_file = ordinary.cache_file,
            .ix = ordinary.ix,
            .w = ordinary.w,
            .fan = private_fan,
            .comm = private_comm.*,
            .set = ordinary.set,
            .pf = ordinary.pf,
            .lg = ordinary.lg,
            .exk = ordinary.exk,
            .ops = ordinary.ops,
            .ex = ordinary.ex,
            .rope_buf = ordinary.rope_buf,
            .arena = undefined,
            .eng = undefined,
            .ch = undefined,
            .caches = undefined,
            .blas = undefined,
            .rings = &.{},
            .ring_view = &.{},
            .rd = undefined,
            .host_pos = &.{},
            .ids64 = &.{},
            .vocab = ordinary.vocab,
            .logits = &.{},
            .amax = 0,
        };
        // The preparation path cannot accept requests. Its failure releases only
        // allocations made here, never ordinary allocations or caller-owned transport.
        errdefer m.host.deinit();
        m.stream = try cuda.Stream.init(m.ctx.d, true);
        errdefer m.stream.deinit();
        m.arena = try prompt.Arena.private(m.ctx.d, o.device_bytes);
        errdefer m.arena.deinit();
        const a = m.host.allocator();
        m.eng = .{
            .d = m.ctx.d, .s = m.stream, .t = .{ .set = &m.set, .stream = m.stream },
            .blas = undefined, .comm = &m.comm, .pf = &m.pf, .lin = &m.lg,
            .ex = &m.exk, .ops = &m.ops, .exact = &m.ex, .c = &m.cfg, .w = &m.w,
            .world = m.world, .plain = ordinary.eng.plain, .compressed = ordinary.eng.compressed,
            .round_rows = ordinary.eng.round_rows, .plain_rows = ordinary.eng.plain_rows,
        };
        m.ch = try prompt.Chunk.init(&m.eng, &m.arena, m.prefill_rows, o.window);
        m.caches = try prompt.Caches.init(&m.eng, &m.arena, o.window, 1);
        m.blas = try cublas.Blas.open(m.stream, m.ch.blas_ws);
        errdefer m.blas.close();
        m.eng.blas = &m.blas;
        m.rings = try a.alloc(u64, m.cfg.layers);
        m.ring_view = try a.alloc(u64, m.cfg.layers);
        for (m.rings) |*r| {
            r.* = try m.arena.take(m.eng.ringBytes());
            if (m.ctx.d.api.cuMemsetD8_v2(r.*, 0, m.eng.ringBytes()) != cuda.abi.success) return error.PrivateStateInitialize;
        }
        m.rd = try round.Round.init(&m.eng, &m.arena, a, o.window);
        errdefer m.rd.sink.fork.deinit();
        m.amax = try m.arena.take(round.max_rows * 4);
        m.cand = try m.arena.take(round.max_rows * (sampling.max_candidates * 12 + 4));
        // Retain the model's drafter arithmetic, but no graph or learned disk profile.
        if (ordinary.dr != null) {
            m.dpool = try draft.Pool.init(&m.eng, &m.arena, 1);
            m.dr = try draft.Drafter.init(&m.eng, &m.arena, try prompt2d.split(m.rank, m.world));
        }
        errdefer if (m.epool) |p| p.deinit(gpa);
        if (ordinary.eh) |eh| {
            m.epool = try engram_io.Pool.init(gpa, m.io, 4);
            m.eh = try prompt.EngramHost.init(a, &m.cfg, eh.hasher, eh.tables, m.epool.?, m.rank, m.world, m.prefill_rows);
        } else if (m.cfg.engram_layers.slice().len != 0) return error.NoEngramTables;
        m.host_pos = try a.alloc(i64, m.prefill_rows);
        m.ids64 = try a.alloc(i64, m.prefill_rows);
        m.logits = try a.alloc(f32, round.max_rows * m.vocab);
        try m.stream.synchronize();
        return s;
    }

    /// Use only behind the cluster's exclusive gate. Never pass this pointer to
    /// Model.close: borrowed weights, file handles and transport belong elsewhere.
    pub fn begin(s: *State) !*Model {
        if (s.phase != .prepared) return error.PrivateStateAlreadyUsed;
        s.phase = .used;
        return &s.m;
    }

    /// Ends CPU producers first, then all GPU writers, then erases every private
    /// device allocation. On failure the cluster exits without dumps; it must not
    /// retain the failed private state in a serving process.
    /// This does not certify erasure of allocator/library/driver-owned storage.
    pub fn erase(s: *State) !void {
        if (s.phase == .erased) return error.PrivateStateAlreadyErased;
        s.m.joinAhead();
        s.m.fill_seq = &.{};
        if (s.m.epool) |p| {
            p.deinit(s.m.gpa);
            s.m.epool = null;
        }
        // Destroy the private library handle before the final wipe: it may still
        // use its workspace during teardown. Failure keeps all memory quarantined.
        if (!s.blas_closed) {
            if (s.m.ctx.d.api.cuCtxSynchronize() != 0) return error.PrivateArenaSynchronize;
            if (s.m.blas.api.cublasDestroy_v2(s.m.blas.handle) != 0) return error.PrivateBlasCleanup;
            s.m.blas.lib.close();
            s.blas_closed = true;
        }
        try s.m.arena.erase();
        s.phase = .erased;
    }

    /// Drop owned state only after successful erasure. Returning an error is not
    /// permission to force-release it or reopen ordinary admission.
    pub fn release(s: *State) !void {
        if (s.phase != .erased) return error.PrivateStateNotErased;
        s.m.rd.sink.fork.deinit();
        try s.m.arena.releasePrivate();
        if (s.m.ctx.d.api.cuStreamDestroy_v2(s.m.stream.handle) != 0) return error.PrivateStreamCleanup;
        s.m.spans.deinit(s.m.gpa);
        s.m.host.deinit(); // parent is ZeroFree: all host arena blocks are overwritten
        const parent = s.parent;
        std.crypto.secureZero(u8, std.mem.asBytes(s)); // inline logits/candidate arrays too
        parent.destroy(s);
    }
};

/// GB10 shares CPU/GPU RAM. Require both driver-free and OS-available memory;
/// do not count reclaimable ordinary KV or change its pool to make secure fit.
fn memoryAdmission(m: *const Model, o: Options) !void {
    var free: usize = 0;
    var total: usize = 0;
    if (m.ctx.d.api.cuMemGetInfo_v2(&free, &total) != 0) return error.PrivateMemoryUnknown;
    var file = try std.Io.Dir.cwd().openFile(m.io, "/proc/meminfo", .{});
    defer file.close(m.io);
    var buf: [16384]u8 = undefined;
    var reader = file.readerStreaming(m.io, &buf);
    var available: ?usize = null;
    while (try reader.interface.takeDelimiter('\n')) |line| {
        if (!std.mem.startsWith(u8, line, "MemAvailable:")) continue;
        var words = std.mem.tokenizeAny(u8, line[13..], " \t");
        const kb = try std.fmt.parseInt(usize, words.next() orelse return error.PrivateMemoryUnknown, 10);
        if (!std.mem.eql(u8, words.next() orelse "", "kB") or words.next() != null) return error.PrivateMemoryUnknown;
        available = std.math.mul(usize, kb, 1024) catch return error.PrivateMemoryUnknown;
        break;
    }
    try headroom(free, available orelse return error.PrivateMemoryUnknown, o.device_bytes);
}

fn headroom(device_free: usize, os_available: usize, bytes: usize) !void {
    // Host scratch, driver/library workspaces and the ordinary deployment's safety
    // margin remain outside the private device ceiling. Qualification measures fit.
    const reserve = 8 << 30;
    if (bytes > device_free -| reserve or bytes > os_available -| reserve) return error.PrivateMemoryHeadroom;
}

test "private admission keeps host and device reserves without reclaiming ordinary cache" {
    try headroom(16 << 30, 20 << 30, 8 << 30);
    try std.testing.expectError(error.PrivateMemoryHeadroom, headroom(15 << 30, 20 << 30, 8 << 30));
    try std.testing.expectError(error.PrivateMemoryHeadroom, headroom(20 << 30, 15 << 30, 8 << 30));
}

fn validate(world: u32, ordinary_window: usize, chunk: usize, o: Options) !void {
    if (world != 3) return error.PrivateStateNeedsThreeRanks;
    if (o.window == 0 or o.window > ordinary_window or o.window < chunk or o.window % @import("model.zig").chunk_rows != 0)
        return error.PrivateStateWindow;
    if (o.device_bytes < 256 << 20) return error.PrivateArenaBudget;
}

test "private state limits never silently shrink the ordinary model" {
    const t = std.testing;
    try validate(3, 1048576, 2048, .{ .window = 65536, .device_bytes = 4 << 30 });
    try t.expectError(error.PrivateStateNeedsThreeRanks, validate(2, 1048576, 2048, .{ .window = 65536, .device_bytes = 4 << 30 }));
    try t.expectError(error.PrivateStateWindow, validate(3, 65536, 2048, .{ .window = 131072, .device_bytes = 4 << 30 }));
    try t.expectError(error.PrivateStateWindow, validate(3, 65536, 2048, .{ .window = 1, .device_bytes = 4 << 30 }));
    try t.expectError(error.PrivateArenaBudget, validate(3, 65536, 2048, .{ .window = 65536, .device_bytes = 1 }));
}

test "private state refuses release before erasure without touching resources" {
    var s: State = undefined;
    s.phase = .used;
    try std.testing.expectError(error.PrivateStateNotErased, s.release());
}

test {
    // Compile the real CUDA allocation/cleanup functions without invoking a GPU.
    _ = &State.prepare;
    _ = &State.begin;
    _ = &State.erase;
    _ = &State.release;
}
