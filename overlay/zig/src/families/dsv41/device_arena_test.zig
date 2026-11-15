// Modified for the DeepSeek 1017 TP3 integration; upstream copyrights retained. See CREDITS.md and provenance.json.
//! Driver-boundary tests only: no GPU, model or inference is run.
const std = @import("std");
const cuda = @import("cuda");
const Arena = @import("device_arena.zig").Arena;

const Driver = struct {
    var next: u64 = 4096;
    var erased: [8]u64 = @splat(0);
    var sizes: [8]usize = @splat(0);
    var values: [8]u8 = @splat(0);
    var writes: usize = 0;
    var releases: usize = 0;
    var syncs: usize = 0;
    var fail_write: bool = false;
    var fail_sync: usize = 0;
    var async_writes: usize = 0;

    fn alloc(p: *u64, _: usize) callconv(.c) c_int {
        p.* = next;
        next += 4096;
        return 0;
    }
    fn free(_: u64) callconv(.c) c_int {
        releases += 1;
        return 0;
    }
    fn fill(p: u64, v: u8, n: usize) callconv(.c) c_int {
        if (fail_write or writes == erased.len) return 1;
        erased[writes] = p;
        values[writes] = v;
        sizes[writes] = n;
        writes += 1;
        return 0;
    }
    fn sync() callconv(.c) c_int {
        syncs += 1;
        return if (syncs == fail_sync) 1 else 0;
    }
    fn fillAsync(p: u64, v: u8, n: usize, stream: cuda.abi.Stream) callconv(.c) c_int {
        if (stream != null) return 1;
        async_writes += 1;
        return fill(p, v, n);
    }
    fn driver() cuda.Driver {
        next = 4096;
        erased = @splat(0);
        sizes = @splat(0);
        writes = 0;
        values = @splat(0);
        releases = 0;
        syncs = 0;
        fail_write = false;
        fail_sync = 0;
        async_writes = 0;
        var d: cuda.Driver = undefined;
        d.api.cuMemAlloc_v2 = alloc;
        d.api.cuMemFree_v2 = free;
        d.api.cuMemsetD8_v2 = fill;
        d.api.cuMemsetD8Async = fillAsync;
        d.api.cuCtxSynchronize = sync;
        return d;
    }
};

test "private erasure covers owned padding but never ordinary blocks" {
    const d = Driver.driver();
    var ordinary = try Arena.init(&d, 4096);
    defer ordinary.deinit();
    var private = try Arena.private(&d, 512 << 20);
    defer private.deinit();
    const ordinary_usage = cuda.usage(false);
    const first = try private.take(9); // owns a whole block, not just these bytes
    const second = try private.take(64 << 20);
    try private.erase();
    try std.testing.expectEqual(@as(usize, 2), Driver.writes);
    try std.testing.expectEqual(first, Driver.erased[0]);
    try std.testing.expectEqual(second, Driver.erased[1]);
    try std.testing.expectEqual(@as(usize, 256 << 20), Driver.sizes[0]);
    try std.testing.expectEqual(@as(usize, 64 << 20), Driver.sizes[1]);
    try std.testing.expect(ordinary.blocks[0].ptr != Driver.erased[0]);
    try std.testing.expectEqual(@as(usize, 0), Driver.releases);
    try std.testing.expectEqual(@as(usize, 2), Driver.syncs);
    try std.testing.expectEqualDeep(ordinary_usage, cuda.usage(false));
}

test "private erasure failure never frees blocks or claims success" {
    const d = Driver.driver();
    var a = try Arena.private(&d, 256 << 20);
    defer a.deinit(); // synthetic test teardown, not a serving recovery policy
    _ = try a.take(1);
    Driver.fail_write = true;
    try std.testing.expectError(error.PrivateArenaErase, a.erase());
    try std.testing.expectEqual(@as(usize, 0), Driver.releases);
    try std.testing.expectEqual(@as(usize, 1), a.n);
    Driver.fail_write = false;
    Driver.fail_sync = Driver.syncs + 2;
    try std.testing.expectError(error.PrivateArenaSynchronize, a.erase());
    try std.testing.expectEqual(@as(usize, 0), Driver.releases);
    try std.testing.expectEqual(@as(usize, 1), a.n);
}

test "private arena refuses over budget before driver allocation" {
    const d = Driver.driver();
    var a = try Arena.private(&d, 256 << 20);
    defer a.deinit();
    _ = try a.take(1);
    try std.testing.expectError(error.PrivateArenaBudget, a.take(64 << 20));
    try std.testing.expectEqual(@as(usize, 1), a.n);
}


test "shared scratch erases only registered work and restores empty sentinels" {
    const d = Driver.driver();
    var a = try Arena.init(&d, 4096);
    defer a.deinit();
    const persistent = try a.take(16);
    a.scratch_scope = true;
    const work = try a.take(9);
    const constant = try a.take(32);
    const queue = try a.take(16);
    try a.keepScratch(constant);
    try a.scratchValue(queue, 0xff);
    a.scratch_scope = false;
    _ = try a.take(16);
    const before_used = a.used;
    try a.eraseScratch();
    try std.testing.expectEqual(@as(usize, 2), Driver.writes);
    try std.testing.expectEqual(work, Driver.erased[0]);
    try std.testing.expectEqual(@as(usize, 2), Driver.async_writes);
    try std.testing.expectEqual(@as(usize, 2), Driver.syncs);
    try std.testing.expectEqual(@as(usize, 9), Driver.sizes[0]);
    try std.testing.expectEqual(@as(u8, 0), Driver.values[0]);
    try std.testing.expectEqual(queue, Driver.erased[1]);
    try std.testing.expectEqual(@as(u8, 0xff), Driver.values[1]);
    try std.testing.expect(persistent != work and constant != work and constant != queue);
    try std.testing.expectEqual(before_used, a.used);
    try std.testing.expectEqual(@as(usize, 0), Driver.releases);
    try std.testing.expectError(error.UnknownScratch, a.keepScratch(persistent));
}

test "shared scratch validates every span before erase and fails closed" {
    const d = Driver.driver();
    var a = try Arena.init(&d, 4096);
    defer a.deinit();
    a.scratch_scope = true;
    _ = try a.take(16);
    _ = try a.take(16);
    const valid = a.scratch[1].ptr;
    a.scratch[1].ptr = 0;
    try std.testing.expectError(error.BadScratchRange, a.eraseScratch());
    try std.testing.expectEqual(@as(usize, 0), Driver.writes);
    a.scratch[1].ptr = valid;
    Driver.fail_write = true;
    try std.testing.expectError(error.PrivateArenaErase, a.eraseScratch());
    try std.testing.expectEqual(@as(usize, 0), Driver.releases);
}

test "shared scratch cannot acknowledge queued clears before final fence succeeds" {
    const d = Driver.driver();
    var a = try Arena.init(&d, 4096);
    defer a.deinit();
    a.scratch_scope = true;
    _ = try a.take(16);
    _ = try a.take(32);
    Driver.fail_sync = 2;
    try std.testing.expectError(error.PrivateArenaSynchronize, a.eraseScratch());
    try std.testing.expectEqual(@as(usize, 2), Driver.async_writes);
    try std.testing.expectEqual(@as(usize, 0), Driver.releases);
    try std.testing.expectEqual(@as(usize, 1), a.n);
}


test "shared scratch coalesces adjacent equal fills without extending their union" {
    const d = Driver.driver();
    var a = try Arena.init(&d, 4096);
    defer a.deinit();
    a.scratch_scope = true;
    const start = try a.take(256);
    _ = try a.take(512);
    const sentinel = try a.take(256);
    try a.scratchValue(sentinel, 0xff);
    try a.eraseScratch();
    try std.testing.expectEqual(@as(usize, 2), Driver.async_writes);
    try std.testing.expectEqual(start, Driver.erased[0]);
    try std.testing.expectEqual(@as(usize, 768), Driver.sizes[0]);
    try std.testing.expectEqual(sentinel, Driver.erased[1]);
    try std.testing.expectEqual(@as(u8, 0xff), Driver.values[1]);
    try std.testing.expectEqual(@as(usize, 2), Driver.syncs);
}

test "shared scratch merging never crosses a constant padding or allocation boundary" {
    const d = Driver.driver();
    var a = try Arena.init(&d, 4096);
    defer a.deinit();
    a.scratch_scope = true;
    const first = try a.take(256);
    const constant = try a.take(256);
    try a.keepScratch(constant);
    const short = try a.take(1);
    const after_padding = try a.take(256);
    try a.eraseScratch();
    try std.testing.expectEqual(@as(usize, 3), Driver.writes);
    try std.testing.expectEqual(first, Driver.erased[0]);
    try std.testing.expectEqual(short, Driver.erased[1]);
    try std.testing.expectEqual(@as(usize, 1), Driver.sizes[1]);
    try std.testing.expectEqual(after_padding, Driver.erased[2]);

    // Contiguous virtual addresses can still be separate owned allocations.
    Driver.writes = 0;
    Driver.async_writes = 0;
    a.blocks[0].len = 256;
    a.blocks[1] = .{ .d = &d, .ptr = first + 256, .len = 256 };
    a.n = 2;
    a.scratch[0] = .{ .ptr = first, .bytes = 256 };
    a.scratch[1] = .{ .ptr = first + 256, .bytes = 256 };
    a.scratch_n = 2;
    try a.eraseScratch();
    try std.testing.expectEqual(@as(usize, 2), Driver.writes);
    try std.testing.expectEqual(@as(usize, 256), Driver.sizes[0]);
    try std.testing.expectEqual(@as(usize, 256), Driver.sizes[1]);
}


test "shared scratch overlap keeps queued device work fenced before acknowledgement" {
    const d = Driver.driver();
    var a = try Arena.init(&d, 4096);
    defer a.deinit();
    a.scratch_scope = true;
    _ = try a.take(16);
    try a.beginScratchErase();
    try std.testing.expectEqual(@as(usize, 1), Driver.syncs);
    try std.testing.expectEqual(@as(usize, 1), Driver.async_writes);
    // Independent host cleanup fits between the two GPU boundaries.
    var host: [16]u8 = @splat(0xa5);
    std.crypto.secureZero(u8, &host);
    Driver.fail_sync = 2;
    try std.testing.expectError(error.PrivateArenaSynchronize, a.finishScratchErase());
    try std.testing.expect(std.mem.allEqual(u8, &host, 0));
    try std.testing.expectEqual(@as(usize, 0), Driver.releases);
}
