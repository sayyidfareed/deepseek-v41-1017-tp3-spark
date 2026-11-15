// Modified for the DeepSeek 1017 TP3 integration; upstream copyrights retained. See CREDITS.md and provenance.json.
//! Pinned Linux CPython 3.12 ABI, loaded in process. No executable or child.
//! The installed image already supplies this runtime and Pillow 12.3.0.
const std = @import("std");
pub const Object = opaque {};
extern "c" fn dlopen([*:0]const u8, c_int) ?*anyopaque;
extern "c" fn dlsym(*anyopaque, [*:0]const u8) ?*anyopaque;

pub const Api = struct {
    Py_IsInitialized: *const fn () callconv(.c) c_int,
    Py_InitializeEx: *const fn (c_int) callconv(.c) void,
    PyEval_SaveThread: *const fn () callconv(.c) ?*anyopaque,
    PyGILState_Ensure: *const fn () callconv(.c) c_int,
    PyGILState_Release: *const fn (c_int) callconv(.c) void,
    PyDict_New: *const fn () callconv(.c) ?*Object,
    PyDict_Clear: *const fn (*Object) callconv(.c) void,
    PyDict_GetItemString: *const fn (*Object, [*:0]const u8) callconv(.c) ?*Object,
    PyRun_StringFlags: *const fn ([*:0]const u8, c_int, *Object, *Object, ?*anyopaque) callconv(.c) ?*Object,
    PyBytes_FromStringAndSize: *const fn ([*]const u8, isize) callconv(.c) ?*Object,
    PyBytes_AsStringAndSize: *const fn (*Object, *?[*]u8, *isize) callconv(.c) c_int,
    PyObject_CallOneArg: *const fn (*Object, *Object) callconv(.c) ?*Object,
    Py_DecRef: *const fn (*Object) callconv(.c) void,
    PyErr_Clear: *const fn () callconv(.c) void,
};

var mutex: std.Io.Mutex = .init;
var ready: ?Api = null;

pub fn acquire(io: std.Io) !Api {
    mutex.lockUncancelable(io);
    defer mutex.unlock(io);
    if (ready) |api| return api;
    // RTLD_NOW | RTLD_GLOBAL: Pillow extension modules resolve CPython symbols.
    // Keep the library and interpreter for the process lifetime: dlclose after
    // importing extension modules or finalizing from arbitrary threads is unsafe.
    const lib = dlopen("libpython3.12.so.1.0", 2 | 0x100) orelse return error.NoPillow;
    var api: Api = undefined;
    const fields = @typeInfo(Api).@"struct";
    inline for (fields.field_names, fields.field_types) |name, T| {
        @field(api, name) = @as(T, @ptrCast(@alignCast(dlsym(lib, name ++ "\x00") orelse return error.NoPillow)));
    }
    // Do not attach to a foreign interpreter with unknown initialization policy.
    if (api.Py_IsInitialized() != 0) return error.ForeignPythonRuntime;
    inline for (.{ "Py_IsolatedFlag", "Py_IgnoreEnvironmentFlag", "Py_NoUserSiteDirectory", "Py_DontWriteBytecodeFlag" }) |name| {
        const flag: *c_int = @ptrCast(@alignCast(dlsym(lib, name) orelse return error.NoPillow));
        flag.* = 1;
    }
    api.Py_InitializeEx(0); // Preserve the engine's signal handlers.
    if (api.Py_IsInitialized() == 0) return error.NoPillow;
    _ = api.PyEval_SaveThread(); // Decoding threads explicitly acquire the GIL.
    ready = api;
    return api;
}
