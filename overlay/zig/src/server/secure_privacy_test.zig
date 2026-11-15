// Modified for the DeepSeek 1017 TP3 integration; upstream copyrights retained. See CREDITS.md and provenance.json.
//! CPU-only boundary checks; no GPU/OS no-retention claim.
const std = @import("std");
const Server = @import("server.zig").Server;
const metrics = @import("metrics.zig");

test "secure completion does not change any ordinary counter or histogram" {
    var srv: Server = undefined;
    srv.config = .{ .served_name = "synthetic", .model_ids = &.{"synthetic"}, .secure_request = true };
    srv.io = std.testing.io;
    srv.metrics = .{ .gpa = std.testing.allocator, .prompt = 11, .cached = 7, .generation = 3 };
    srv.noteRequest(100, 40, 20, 5, 3, 2, 0, 1, 2, 0.5);
    try std.testing.expectEqual(@as(u64, 11), srv.metrics.prompt);
    try std.testing.expectEqual(@as(u64, 7), srv.metrics.cached);
    try std.testing.expectEqual(@as(u64, 3), srv.metrics.generation);
    inline for (.{"drafted", "accepted", "rounds", "disconnects"}) |field|
        try std.testing.expectEqual(@as(u64, 0), @field(srv.metrics, field));
    inline for (.{"latency", "ttft", "decode", "prefill", "tpot"}) |field|
        try std.testing.expectEqual(@as(u64, 0), @field(srv.metrics, field).n);
    srv.config.secure_request = false;
    srv.noteRequest(100, 40, 20, 5, 3, 2, 0, 1, 2, 0.5);
    try std.testing.expectEqual(@as(u64, 111), srv.metrics.prompt);
    try std.testing.expectEqual(@as(u64, 47), srv.metrics.cached);
    try std.testing.expectEqual(@as(u64, 23), srv.metrics.generation);
    try std.testing.expectEqual(@as(u64, 1), srv.metrics.latency.n);
}

test "secure cross-thread request allocator erases memory without changing ordinary allocation" {
    var bytes: [1024]u8 = @splat(0);
    var fixed = std.heap.FixedBufferAllocator.init(&bytes);
    var srv: Server = undefined;
    srv.gpa = fixed.allocator();
    srv.request_zero = .{ .parent = srv.gpa };
    srv.config = .{ .served_name = "synthetic", .model_ids = &.{"synthetic"}, .secure_request = true };
    const a = srv.requestAllocator();
    const buffer = try a.dupe(u8, "SYNTHETIC_PRIVATE_BUFFER");
    a.free(buffer);
    try std.testing.expect(std.mem.allEqual(u8, buffer, 0));
    srv.config.secure_request = false;
    try std.testing.expectEqual(srv.gpa.ptr, srv.requestAllocator().ptr);
    try std.testing.expectEqual(srv.gpa.vtable, srv.requestAllocator().vtable);
}
