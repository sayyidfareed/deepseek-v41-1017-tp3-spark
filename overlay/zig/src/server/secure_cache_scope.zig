// Modified for the DeepSeek 1017 TP3 integration; upstream copyrights retained. See CREDITS.md and provenance.json.
//! Header contract for the separate secure cache. No allocation or logging.
const std = @import("std");

pub const header = "X-Secure-Cache-Scope";
pub const Scope = [32]u8;

/// Call only after authentication. Ordinary requests never acquire a scope,
/// even if their header resembles a secure chat's capability.
pub fn parse(secure: bool, count: usize, value: ?[]const u8) !?Scope {
    if (!secure) return null;
    if (count == 0 and value == null) return null;
    if (count != 1) return error.InvalidSecureCacheScope;
    const s = value orelse return error.InvalidSecureCacheScope;
    if (s.len != 64) return error.InvalidSecureCacheScope;
    var result: Scope = undefined;
    errdefer std.crypto.secureZero(u8, &result);
    for (&result, 0..) |*byte, i| {
        byte.* = (try nibble(s[i * 2])) * 16 + try nibble(s[i * 2 + 1]);
    }
    return result;
}

fn nibble(c: u8) !u8 {
    return switch (c) {
        '0'...'9' => c - '0',
        'a'...'f' => c - 'a' + 10,
        else => error.InvalidSecureCacheScope,
    };
}

test "secure scope is exactly 32 bytes, absent is no reuse, ordinary ignores it" {
    const s = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    const got = (try parse(true, 1, s)).?;
    try std.testing.expectEqual(@as(u8, 0x01), got[0]);
    try std.testing.expectEqual(@as(u8, 0xef), got[31]);
    try std.testing.expect((try parse(true, 0, null)) == null);
    try std.testing.expect((try parse(false, 1, s)) == null);
    try std.testing.expect((try parse(false, 2, "invalid")) == null);
}

test "secure scope refuses duplicates, ambiguous encodings and malformed input" {
    const s = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    for ([_][]const u8{ "", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", "gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg",
        " " ++ "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" ++ "\x00", s ++ "," ++ s }) |bad|
        try std.testing.expectError(error.InvalidSecureCacheScope, parse(true, 1, bad));
    try std.testing.expectError(error.InvalidSecureCacheScope, parse(true, 2, s));
    try std.testing.expectError(error.InvalidSecureCacheScope, parse(true, 1, null));
    try std.testing.expectError(error.InvalidSecureCacheScope, parse(true, 0, s));
}
