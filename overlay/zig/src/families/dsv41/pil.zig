// Modified for the DeepSeek 1017 TP3 integration; upstream copyrights retained. See CREDITS.md and provenance.json.
//! Ordinary image decoding using the installed Pillow in process. Fork/exec
//! remain denied by the unchanged secure guard; secure vision remains refused
//! before this path. No child, socket, disk exchange or per-request Python global.
const std = @import("std");
const picture = @import("picture.zig");
const python = @import("pillow_api.zig");
const Allocator = std.mem.Allocator;
pub const script = @embedFile("pillow_decode.py");

pub const Pil = struct {
    io: std.Io,
    api: python.Api,
    namespace: *python.Object,
    function: *python.Object, // borrowed from namespace
    mutex: std.Io.Mutex = .init,
    version: [32]u8 = @splat(0),
    version_len: usize = 0,

    pub fn start(gpa: Allocator, io: std.Io) !*Pil {
        const api = try python.acquire(io);
        const gil = api.PyGILState_Ensure();
        defer api.PyGILState_Release(gil);
        defer api.PyErr_Clear(); // Never print a Python traceback.
        const namespace = api.PyDict_New() orelse return error.OutOfMemory;
        errdefer {
            api.PyDict_Clear(namespace);
            api.Py_DecRef(namespace);
        }
        const result = api.PyRun_StringFlags(script, 257, namespace, namespace, null) orelse return error.NoPillow;
        api.Py_DecRef(result);
        const function = api.PyDict_GetItemString(namespace, "decode") orelse return error.NoPillow;
        const version = api.PyDict_GetItemString(namespace, "VERSION") orelse return error.NoPillow;
        var data: ?[*]u8 = null;
        var n: isize = 0;
        if (api.PyBytes_AsStringAndSize(version, &data, &n) != 0 or n <= 0 or n > 32 or data == null) return error.NoPillow;
        const p = try gpa.create(Pil);
        p.* = .{ .io = io, .api = api, .namespace = namespace, .function = function, .version_len = @intCast(n) };
        @memcpy(p.version[0..p.version_len], data.?[0..p.version_len]);
        return p;
    }

    pub fn stop(p: *Pil, gpa: Allocator) void {
        const gil = p.api.PyGILState_Ensure();
        p.api.PyDict_Clear(p.namespace); // Break function -> globals cycles.
        p.api.Py_DecRef(p.namespace);
        p.api.PyErr_Clear();
        p.api.PyGILState_Release(gil);
        gpa.destroy(p);
    }

    pub fn versionText(p: *const Pil) []const u8 {
        return p.version[0..p.version_len];
    }

    pub const Decoded = union(enum) { rgb: picture.Rgb, refused: []const u8 };

    pub fn decode(p: *Pil, a: Allocator, data: []const u8) !Decoded {
        if (data.len > std.math.maxInt(isize)) return error.BadPillowReply;
        p.mutex.lockUncancelable(p.io);
        defer p.mutex.unlock(p.io);
        const api = p.api;
        const gil = api.PyGILState_Ensure();
        defer api.PyGILState_Release(gil);
        defer api.PyErr_Clear();
        const input = api.PyBytes_FromStringAndSize(data.ptr, @intCast(data.len)) orelse return error.OutOfMemory;
        defer api.Py_DecRef(input);
        const result = api.PyObject_CallOneArg(p.function, input) orelse return error.ImageDecodeFailed;
        defer api.Py_DecRef(result);
        var raw: ?[*]u8 = null;
        var size: isize = 0;
        if (api.PyBytes_AsStringAndSize(result, &raw, &size) != 0 or size < 1 or raw == null) return error.BadPillowReply;
        const reply = raw.?[0..@intCast(size)];
        if (reply.len == 1 and reply[0] == 1) return .{ .refused = "image input: image exceeds pixel limit" };
        if (reply.len == 1 and reply[0] == 2) return .{ .refused = "image input: cannot identify or decode image" };
        if (reply.len < 25 or reply[0] != 0) return error.BadPillowReply;
        const w = std.mem.readInt(u64, reply[1..9], .little);
        const h = std.mem.readInt(u64, reply[9..17], .little);
        const n = std.mem.readInt(u64, reply[17..25], .little);
        if (w == 0 or h == 0 or w > picture.max_pixels or h > picture.max_pixels or w > picture.max_pixels / h) return error.BadPillowReply;
        if (n != w * h * 3 or n != reply.len - 25) return error.BadPillowReply;
        return .{ .rgb = .{ .w = @intCast(w), .h = @intCast(h), .px = try a.dupe(u8, reply[25..]) } };
    }
};
