// Modified for the DeepSeek 1017 TP3 integration; upstream copyrights retained. See CREDITS.md and provenance.json.
//! Prevent freed protected-process allocations from retaining request bytes.
const std = @import("std");
pub const ZeroFree = struct {
    parent: std.mem.Allocator,
    pub fn allocator(z: *ZeroFree) std.mem.Allocator {
        return .{ .ptr = z, .vtable = &.{ .alloc = alloc, .resize = std.mem.Allocator.noResize, .remap = std.mem.Allocator.noRemap, .free = free } };
    }
    fn alloc(p: *anyopaque, n: usize, alignment: std.mem.Alignment, ra: usize) ?[*]u8 {
        const z: *ZeroFree = @ptrCast(@alignCast(p));
        return z.parent.rawAlloc(n, alignment, ra);
    }
    fn free(p: *anyopaque, bytes: []u8, alignment: std.mem.Alignment, ra: usize) void {
        const z: *ZeroFree = @ptrCast(@alignCast(p));
        std.crypto.secureZero(u8, bytes);
        z.parent.rawFree(bytes, alignment, ra);
    }
};

test "protected-process allocator erases released storage including replaced allocations" {
    var bytes: [1024]u8 = @splat(0x77);
    var fixed = std.heap.FixedBufferAllocator.init(&bytes);
    var zero: ZeroFree = .{ .parent = fixed.allocator() };
    const a = zero.allocator();
    const original = try a.alloc(u8, 32);
    @memset(original, 0xa5);
    const grown = try a.realloc(original, 64);
    try std.testing.expect(std.mem.allEqual(u8, original, 0));
    try std.testing.expect(std.mem.allEqual(u8, grown[0..32], 0xa5));
    @memset(grown, 0xa6);
    a.free(grown);
    try std.testing.expect(std.mem.allEqual(u8, grown, 0));
}
