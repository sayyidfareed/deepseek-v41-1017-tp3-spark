// Modified for the DeepSeek 1017 TP3 integration; upstream copyrights retained. See CREDITS.md and provenance.json.
//! Startup numeric guard: all GPU bytes compared by SHA-256, bounded host scratch.
const std = @import("std");
const cuda = @import("cuda");

pub fn digest(s: cuda.Stream, ptr: u64, bytes: usize) ![32]u8 {
    try s.synchronize();
    const alloc = std.heap.page_allocator;
    const buf = try alloc.alloc(u8, @min(bytes, 1 << 20));
    defer alloc.free(buf);
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    var offset: usize = 0;
    while (offset < bytes) {
        const n = @min(buf.len, bytes - offset);
        try s.d.check(s.d.api.cuMemcpyDtoH_v2(buf.ptr, ptr + offset, n), "prompt kernel numeric guard");
        h.update(buf[0..n]);
        offset += n;
    }
    return h.finalResult();
}
