// Modified for the DeepSeek 1017 TP3 integration; upstream copyrights retained. See CREDITS.md and provenance.json.
//! Owned device blocks for prompt execution; private copies have a hard budget and explicit erasure.
const std = @import("std");
const cuda = @import("cuda");

test { _ = @import("device_arena_test.zig"); }

pub const Arena = struct {
    d: *const cuda.Driver,
    blocks: [max_blocks]cuda.DeviceBuffer = undefined,
    n: usize = 0,
    cur: ?usize = null, // the block small pieces come from
    at: usize = 0, // its next free byte
    used: usize = 0,
    reserved: usize = 0,
    limit: usize = std.math.maxInt(usize),
    private_request: bool = false,
    // Startup classification only: ordinary inference never walks this table.
    // Persistent KV and immutable constants remain outside scratch_scope.
    scratch_scope: bool = false,
    scratch: [4096]Scratch = undefined,
    scratch_n: usize = 0,

    const Scratch = struct { ptr: u64, bytes: usize, value: u8 = 0, enabled: bool = true };

    const max_blocks = 1024;
    const block: usize = 256 << 20;
    const own: usize = 64 << 20;

    pub fn init(d: *const cuda.Driver, bytes: usize) !Arena {
        var a: Arena = .{ .d = d };
        if (bytes > 0) {
            a.cur = try a.grow(bytes);
            a.at = 0;
        }
        return a;
    }

    fn grow(a: *Arena, bytes: usize) !usize {
        if (bytes > a.limit -| a.reserved) return error.PrivateArenaBudget;
        if (a.n == max_blocks) return error.ArenaFull;
        if (a.private_request) {
            // Do not add secure allocation sizes to ordinary usage/peak telemetry.
            var ptr: u64 = 0;
            if (a.d.api.cuMemAlloc_v2(&ptr, bytes) != 0) return error.PrivateArenaAllocation;
            a.blocks[a.n] = .{ .d = a.d, .ptr = ptr, .len = bytes };
        } else a.blocks[a.n] = try cuda.DeviceBuffer.alloc(a.d, bytes);
        a.reserved += bytes;
        a.n += 1;
        return a.n - 1;
    }

    /// Private execution supplies a hard ceiling; ordinary init retains its allocation policy.
    pub fn private(d: *const cuda.Driver, limit: usize) !Arena {
        if (limit < block) return error.PrivateArenaBudget;
        return .{ .d = d, .limit = limit, .private_request = true };
    }

    /// Wipe whole owned allocations, including padding, only after all writers finish.
    /// No block is released on any failure. The caller must keep admission closed.
    /// Errors deliberately bypass Driver.check, whose diagnostics are not secure output.
    pub fn erase(a: *Arena) !void {
        if (a.d.api.cuCtxSynchronize() != cuda.abi.success) return error.PrivateArenaSynchronize;
        for (a.blocks[0..a.n]) |b| {
            if (a.d.api.cuMemsetD8_v2(b.ptr, 0, b.len) != cuda.abi.success) return error.PrivateArenaErase;
        }
        if (a.d.api.cuCtxSynchronize() != cuda.abi.success) return error.PrivateArenaSynchronize;
    }

    pub fn deinit(a: *Arena) void {
        if (a.private_request) { a.releasePrivate() catch std.process.exit(70); return; }
        for (a.blocks[0..a.n]) |*b| b.free();
        a.n = 0;
    }

    pub fn releasePrivate(a: *Arena) !void {
        if (!a.private_request) return error.NotPrivateArena;
        for (a.blocks[0..a.n]) |*b| {
            if (b.ptr != 0 and a.d.api.cuMemFree_v2(b.ptr) != 0) return error.PrivateArenaRelease;
            b.ptr = 0;
        }
        a.n = 0;
    }

    pub fn take(a: *Arena, bytes: usize) !u64 {
        if (a.scratch_scope and a.scratch_n == a.scratch.len) return error.ScratchRegistryFull;
        const ptr = try a.takeRaw(bytes);
        if (a.scratch_scope and bytes != 0) {
            a.scratch[a.scratch_n] = .{ .ptr = ptr, .bytes = bytes };
            a.scratch_n += 1;
        }
        return ptr;
    }

    /// Remove a constant initialized alongside transient buffers from erasure.
    pub fn keepScratch(a: *Arena, ptr: u64) !void {
        for (a.scratch[0..a.scratch_n]) |*s| if (s.ptr == ptr) {
            s.enabled = false;
            return;
        };
        return error.UnknownScratch;
    }

    /// Some work queues need their original empty sentinel, not zero.
    pub fn scratchValue(a: *Arena, ptr: u64, value: u8) !void {
        for (a.scratch[0..a.scratch_n]) |*s| if (s.ptr == ptr) {
            s.value = value;
            return;
        };
        return error.UnknownScratch;
    }

    /// No allocation or graph replacement. Validate all spans before writes;
    /// only call between complete rounds, after every stream consumed its rows.
    pub fn eraseScratch(a: *Arena) !void {
        try a.beginScratchErase();
        try a.finishScratchErase();
    }

    /// Queue exact device spans after their writers finish. Independent host
    /// buffers may be erased while these run; finish is mandatory before reuse.
    pub fn beginScratchErase(a: *Arena) !void {
        for (a.scratch[0..a.scratch_n]) |s| {
            if (!s.enabled) continue;
            const end = std.math.add(u64, s.ptr, s.bytes) catch return error.BadScratchRange;
            var found = false;
            for (a.blocks[0..a.n]) |b| {
                const bend = std.math.add(u64, b.ptr, b.len) catch return error.BadScratchRange;
                if (s.ptr >= b.ptr and end <= bend) { found = true; break; }
            }
            if (!found) return error.BadScratchRange;
        }
        if (a.d.api.cuCtxSynchronize() != cuda.abi.success) return error.PrivateArenaSynchronize;
        var run: ?Scratch = null;
        for (a.scratch[0..a.scratch_n]) |s| {
            if (!s.enabled) continue;
            if (run) |*r| {
                // Exact adjacency only: never include padding/constants or cross
                // an allocation boundary even when virtual addresses touch.
                const end = r.ptr + r.bytes;
                var same_block = false;
                if (end == s.ptr and r.value == s.value) {
                    for (a.blocks[0..a.n]) |b| {
                        const bend = std.math.add(u64, b.ptr, b.len) catch continue;
                        if (r.ptr >= b.ptr and s.ptr + s.bytes <= bend) {
                            same_block = true;
                            break;
                        }
                    }
                }
                if (same_block) { r.bytes += s.bytes; continue; }
                try a.clearScratchRun(r.*);
            }
            run = s;
        }
        if (run) |r| try a.clearScratchRun(r);
    }

    pub fn finishScratchErase(a: *Arena) !void {
        if (a.d.api.cuCtxSynchronize() != cuda.abi.success) return error.PrivateArenaSynchronize;
    }

    fn clearScratchRun(a: *Arena, s: Scratch) !void {
        // All original spans were validated and writers fenced before enqueue.
        if (a.d.api.cuMemsetD8Async(s.ptr, s.value, s.bytes, null) != cuda.abi.success) return error.PrivateArenaErase;
    }

    fn takeRaw(a: *Arena, bytes: usize) !u64 {
        if (bytes >= own) {
            const i = try a.grow(std.mem.alignForward(usize, bytes, 256));
            a.used += bytes;
            return a.blocks[i].ptr;
        }
        var at = std.mem.alignForward(usize, a.at, 256);
        if (a.cur == null or at + bytes > a.blocks[a.cur.?].len) {
            a.cur = try a.grow(block);
            at = 0;
        }
        a.at = at + bytes;
        a.used += bytes;
        return a.blocks[a.cur.?].ptr + at;
    }
};
