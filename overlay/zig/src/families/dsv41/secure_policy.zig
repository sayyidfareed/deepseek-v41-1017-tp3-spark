// Modified for the DeepSeek 1017 TP3 integration; upstream copyrights retained. See CREDITS.md and provenance.json.
//! Only a guard loaded before serving threads exist can authorize a secure phase.
//! Do not dlopen a guard here: late injection cannot protect already-running threads.
//! Deployment must pin and qualify the guard as well as this engine.
const std = @import("std");

pub const Guard = struct {
    check: *const fn () callconv(.c) c_int,
    begin: *const fn () callconv(.c) c_int,
    end: *const fn () callconv(.c) c_int,
    active: *const fn () callconv(.c) c_int,

    pub fn dormant() !Guard {
        const g: Guard = .{
            .check = try symbol("tf_private_policy_check"),
            .begin = try symbol("tf_private_begin"),
            .end = try symbol("tf_private_end"),
            .active = try symbol("tf_private_active"),
        };
        if (g.check() != 0 or g.active() != 0) return error.SecurePolicyUnavailable;
        return g;
    }

    pub fn enter() !Guard {
        const g = try dormant();
        if (g.begin() != 0 or g.active() != 1) return error.SecurePolicyUnavailable;
        return g;
    }

    pub fn leave(g: Guard) !void {
        if (g.check() != 0 or g.active() != 1 or g.end() != 0 or g.active() != 0) return error.SecurePolicyCleanup;
    }
};

fn symbol(name: [:0]const u8) !*const fn () callconv(.c) c_int {
    return @ptrCast(@alignCast(std.c.dlsym(null, name) orelse return error.SecurePolicyUnavailable));
}

test "private policy refuses absent protection instead of installing it late" {
    try std.testing.expectError(error.SecurePolicyUnavailable, Guard.enter());
}
