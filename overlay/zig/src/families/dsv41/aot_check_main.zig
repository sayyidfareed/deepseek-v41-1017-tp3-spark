// Modified for the DeepSeek 1017 TP3 integration; upstream copyrights retained. See CREDITS.md and provenance.json.
//! CPU-only native launch matrix. No CUDA allocation, driver, model or network.
//! This covers the named wrappers, not a complete model execution trace.
const std = @import("std");
const ds = @import("dsv41");
const cuda = @import("cuda");
const tri = ds.tri;
const hc = ds.tri_hc;
const attn = ds.tri_attn;
const norm = ds.tri_norm;
const index = ds.tri_index;
const markov = ds.tri_markov;

fn fake(n: u64) u64 {
    return 0x100000 + n * 0x100;
}

// CUDA top-k is outside this Triton manifest audit. Its callback must be
// inert here; we still collect every subsequent Triton selection launch.
fn skipCudaTopk(_: ?*anyopaque, _: tri.Tri, _: u64, _: usize, _: usize, _: usize, _: usize, _: u64) !void {}

fn recordIndexer(t: tri.Tri, rows: usize, prefill: bool) !void {
    const k: index.IndexK = .{ .fp4 = .{ .codes = fake(0), .scales = fake(1) } };
    const top: index.TopK = .{ .top = fake(2), .run = skipCudaTopk };
    const base: ?u64 = if (prefill) null else fake(3);
    for ([_]usize{ 1, 64, 512, 4096, 4097, 16384, 32768, 65536, 131072, 262144, 401408, 524288 }) |n| {
        const kk = @min(n, 512);
        try index.indexScore(t, fake(4), k, fake(5), fake(6), n, fake(7), base, false, null, false, rows, 32, 128);
        try index.scoreKeys(t, fake(7), fake(8), fake(9), rows, n);
        try index.topkSelectPruned(t, fake(9), n, fake(8), kk, fake(6), fake(10), .{ .tpos = fake(11), .cand = fake(12), .every = fake(13) }, top, rows, n);
        if (!prefill) {
            for ([_]bool{ false, true }) |with_cand| {
                try index.indexKeys(t, fake(4), k, fake(5), fake(6), n, base, if (with_cand) .{ .mask = fake(14), .stride = (n + 7) / 8 } else null, 8, fake(8), kk, fake(9), rows, 32, 128);
            }
            try index.indexKeysCand(t, fake(4), k, fake(5), fake(6), n, fake(15), 2048, 2048, 8, base, fake(9), rows, 32, 128);
        }
    }
}

fn record(t: tri.Tri, rows: usize, prefill: bool) !void {
    try ds.tri_basic.hcPost(t, fake(0), fake(1), fake(2), fake(3), fake(4), 3, rows, 5120);
    // Actual TP3 router profile: 384 experts, TOPK=6, shared id=384.
    // C8/K5 verification can use any 1..48 rows, including 40. Prefill
    // projects by the separate dense path audited with projection-shapes.
    if (!prefill) {
        const kc = try norm.rowmmGate(t, fake(0), 5120, fake(1), fake(2), rows, 5120, 384);
        try norm.rowmmWts(t, fake(0), 5120, fake(1), 0.001, fake(2), rows, 5120, 32);
        try norm.route(t, fake(0), kc, fake(1), 6, 2.5, 384, fake(2), fake(3), rows, 384, 7);
    }
    for ([_]bool{ false, true }) |posted| {
        const ps: ?hc.Posted = if (posted) .{ .gathered = fake(20), .world = 3, .h_out = fake(21) } else null;
        if (prefill) {
            _ = try hc.hcPrePf(t, fake(0), fake(1), fake(2), fake(3), fake(4), fake(5), 1e-6, 1e-6, 20, fake(6), fake(7), fake(8), fake(9), fake(10), ps, rows, 5120);
        } else for ([_]bool{ false, true }) |sink| {
            _ = try hc.hcPre2(t, fake(0), fake(1), fake(2), fake(3), fake(4), fake(5), 1e-6, 1e-6, 20, fake(6), fake(7), fake(8), fake(9), fake(10), ps, if (sink) .{ .s = undefined, .fork = undefined } else null, rows, 5120);
        }
    }
    for ([_]bool{ false, true }) |inv| try norm.ropeHeads(t, fake(0), fake(1), fake(2), fake(3), 64, inv, rows, 24, 512);
    for ([_]usize{ 0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 }) |nidx| {
        // Small prompts normally use the ring. Replay at a cut can instead
        // read the gathered window and force a single split (prompt.attention).
        for ([_]bool{ false, true }) |replay| {
            if (replay and (!prefill or rows > ds.prompt.ring_extra)) continue;
            const ring = !prefill or (rows <= ds.prompt.ring_extra and !replay);
            try attn.sparseAttn(t, .{
                .q = fake(0),
                .out = fake(1),
                .rows = rows,
                .h = 24,
                .hd = 512,
                .sink = fake(2),
                .wsrc = fake(3),
                .wsrc_rows = 4096,
                .wlo = fake(4),
                .ring = ring,
                .comp = if (nidx == 0) .none else .{ .fp4 = .{ .codes = fake(5), .scales = fake(6) } },
                .idx = fake(7),
                .n_idx = nidx,
                .pos = fake(8),
                .scale = 0.04419417382415922,
                .window = 128,
                .wbase = if (prefill) null else fake(9),
                .cbase = if (prefill) null else fake(10),
                .ring_rows = 128 + ds.prompt.ring_extra,
                .parts = .{ .pm = fake(11), .pl = fake(12), .po = fake(13) },
                // round.zig fuses inverse RoPE and wo_a input rotation;
                // padded H=24 / three local groups => GH=8.
                .rot = if (prefill) null else .{ .cos = fake(14), .sin = fake(15), .rd = 64, .suh = fake(16), .xh = fake(17), .gh = 8 },
                .one_split = prefill and !ring and rows <= tri.decode_rows,
            });
        }
    }
}

// draft.attention has window-only BF16 block rows, no fused merge rotation,
// and a five-row block per live stream even when fewer steps are requested.
fn recordDraft(t: tri.Tri, rows: usize, c: ds.Config) !void {
    const d = c.hidden;
    for ([_]bool{ false, true }) |posted| {
        const ps: ?hc.Posted = if (posted) .{ .gathered = fake(20), .world = 3, .h_out = fake(21) } else null;
        _ = try hc.hcPre2(t, fake(0), fake(1), fake(2), fake(3), fake(4), fake(5), c.eps, c.hc_eps, c.hc_iters,
            fake(6), fake(7), fake(8), fake(9), fake(10), ps, null, rows, d);
    }
    try ds.tri_basic.hcPost(t, fake(0), fake(1), fake(2), fake(3), fake(4), 3, rows, d);
    var kc: usize = 0;
    if (rows <= tri.decode_rows) kc = try norm.rowmmGate(t, fake(0), d, fake(1), fake(2), rows, d, c.draft_experts);
    // Above the threshold draft.moe uses cuBLAS, not the Triton rowmm fallback.
    try norm.route(t, fake(0), kc, fake(1), c.draft_top_k, c.routed_scaling, c.draft_experts,
        fake(2), fake(3), rows, c.draft_experts, c.draft_top_k + 1);
    try attn.sparseAttn(t, .{
        .q = fake(0), .out = fake(1), .rows = rows, .h = 24, .hd = c.head_dim,
        .sink = fake(2), .wsrc = fake(3), .wsrc_rows = 12 * 128, .wlo = fake(4), .ring = true,
        .comp = .{ .bf16 = fake(5) }, .idx = fake(6), .n_idx = c.dspark_block,
        .pos = fake(7), .scale = 0.04419417382415922, .window = c.window,
        .wbase = fake(8), .cbase = fake(9), .ring_rows = 128,
        .parts = .{ .pm = fake(10), .pl = fake(11), .po = fake(12) },
    });
    try norm.ropeHeads(t, fake(0), fake(1), fake(2), fake(3), c.rope_dim, true, rows, 24, c.head_dim);
}

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 2 or args.len > 4) return error.ExpectedAotJsonPath;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, args[1], a, .limited(1 << 26));
    const parsed = try std.json.parseFromSlice(cuda.aot.SetJson, a, bytes, .{ .ignore_unknown_fields = true });
    const manifest = parsed.value;
    defer parsed.deinit();
    var log: tri.Log = .{ .gpa = a };
    defer log.deinit();
    const t: tri.Tri = .{ .log = &log, .kit = manifest.kernels };
    tri.decode_rows = 48; // C8, target + K5 verification. Restore in normal runtime.
    for (1..49) |rows| try record(t, rows, false);
    tri.decode_rows = 16;
    for ([_]usize{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 31, 32, 63, 64, 128, 256, 512, 1024, 4095, 4096 }) |rows| try record(t, rows, true);
    // TP3 owned row strides round to a multiple of 16 per rank. Keep the ordinary
    // path above, and separately audit the optional HB16 prompt kernels.
    attn.prompt_masked_heads = true;
    for ([_]usize{ 33, 64, 671, 672, 688, 1343, 1344, 1376, 4096 }) |rows| try record(t, rows, true);
    attn.prompt_masked_heads = false;
    // round.forward temporarily selects the round bound for the indexer,
    // independently of the normal prompt/drafter threshold above.
    tri.decode_rows = 48;
    for ([_]usize{ 1, 2, 3, 8, 16, 24, 32, 40, 48 }) |rows| try recordIndexer(t, rows, false);
    tri.decode_rows = 16;
    for ([_]usize{ 1, 16, 48, 49, 128, 4096 }) |rows| try recordIndexer(t, rows, true);
    for ([_]usize{ 671, 672, 688, 1343, 1344, 1376 }) |rows| try recordIndexer(t, rows, true);
    var projections: usize = 0;
    if (args.len >= 3) {
        const shape_bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, args[2], a, .limited(1 << 22));
        const ShapeFile = struct { projections: []struct { k: usize, n: usize } };
        const shapes = try std.json.parseFromSlice(ShapeFile, a, shape_bytes, .{ .ignore_unknown_fields = true });
        defer shapes.deinit();
        projections = shapes.value.projections.len;
        if (projections == 0) return error.EmptyProjectionShapes;
        for (shapes.value.projections) |shape| {
            if (shape.k == 0 or shape.n == 0 or shape.k % 128 != 0 or shape.n % 128 != 0) return error.BadProjectionShape;
            for ([_]usize{ 129, 256, 671, 672, 688, 1024, 1343, 1344, 1376, 4096 }) |rows| {
                for ([_]markov.OutType{ .bf16, .fp32 }) |ty| {
                    try markov.matmul(t, fake(0), fake(1), fake(2), fake(3), null, fake(4), ty, shape.n, rows, shape.k, shape.n);
                }
            }
        }
    }
    if (args.len == 4) {
        const c = try ds.Config.read(a, init.io, args[3]);
        try c.checkWorld(3);
        for (1..13) |streams| try recordDraft(t, streams * c.dspark_block, c);
        const Gather = struct {
            fn run(_: *anyopaque, _: u64, _: usize) !u64 { return fake(31); }
        };
        var marker: u8 = 0;
        for (0..3) |rank| {
            var mk = try markov.Markov.init(c, .{ .rank = @intCast(rank), .world = 3 }, fake(0), fake(1), fake(2));
            try mk.fill(t, fake(3), 256, fake(4), fake(5));
            for (1..13) |streams| for (1..c.dspark_block+1) |steps| {
                try mk.steps(t, fake(6), fake(7), steps+1, streams, c.dspark_block, steps,
                    .{ .pv=fake(8), .pi=fake(9), .send=fake(10), .stage=fake(11) },
                    .{ .ctx=&marker, .run=Gather.run });
            };
        }
    }
    var missing: std.ArrayList(tri.Log.Kept) = .empty;
    var dense_launches: std.ArrayList(tri.Log.Kept) = .empty;
    for (log.items.items) |k| if (std.mem.eql(u8, k.name, "_gemm")) try dense_launches.append(a, k);
    // The scheduler takes views into i64 position arrays. Odd row offsets
    // keep natural i64 alignment but lose Triton's optional div16 guarantee.
    // Audit the resolver with actual wrapper arguments; no GPU allocation.
    var position_view_launches: usize = 0;
    for (log.items.items) |k| {
        if (!cuda.aot.specsHave(manifest.kernels, k.name, k.args, k.consts)) try missing.append(a, k);
        const viewed = try a.dupe(cuda.aot.Arg, k.args);
        var has_position = false;
        for (viewed) |*arg| {
            if (std.mem.eql(u8, arg.name, "POS") and arg.value == .ptr and std.mem.eql(u8, arg.value.ptr.ty, "*i64")) {
                arg.value.ptr.addr += 8;
                has_position = true;
            }
        }
        if (has_position) {
            position_view_launches += 1;
            if (!cuda.aot.specsHave(manifest.kernels, k.name, viewed, k.consts)) try missing.append(a, .{ .name = k.name, .grid = k.grid, .args = viewed, .consts = k.consts });
        }
    }
    var buf: [65536]u8 = undefined;
    var out = std.Io.File.stdout().writer(init.io, &buf);
    const json = try std.json.Stringify.valueAlloc(a, .{
        .status = if (missing.items.len == 0) "MATRIX_COVERED_NOT_GPU_QUALIFIED" else "MISSING_VARIANTS",
        .scope = "mHC, main RoPE, sparse attention, indexer, Triton top-k and optional native-plan dense prefill GEMM; CUDA launches excluded",
        .dense_projection_shapes = projections,
        .dense_launches = dense_launches.items,
        .launches = log.items.items.len,
        .position_view_launches = position_view_launches,
        .missing = missing.items,
    }, .{});
    try out.interface.writeAll(json);
    try out.interface.writeAll("\n");
    try out.interface.flush();
    return if (missing.items.len == 0) 0 else 2;
}
