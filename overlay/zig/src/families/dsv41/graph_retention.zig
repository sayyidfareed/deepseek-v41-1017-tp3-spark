// Modified for the DeepSeek 1017 TP3 integration; upstream copyrights retained. See CREDITS.md and provenance.json.
//! CPU metadata only. Opt-in bounded, aging LFU of complete graph shapes.
//! No buffer ownership, tensor changes, GPU calls, timers or file I/O here.
const std = @import("std");

pub fn configured(comptime name: [:0]const u8) usize {
    const raw = std.c.getenv(name) orelse return 0;
    const n = std.fmt.parseInt(usize, std.mem.span(raw), 10) catch
        std.debug.panic("invalid {s}", .{name});
    if (n > 8192) std.debug.panic("{s} exceeds 8192 groups", .{name});
    return n;
}

pub fn reserveBytes() usize { return configured("TF_DS_GRAPH_RESERVE_MIB") * 1024 * 1024; }
extern "c" fn malloc_trim(usize) c_int;
pub fn trimHostHeap() void {
    // Only after coordinated pressure eviction, never periodically on hits.
    // glibc releases unused pages, preserving all live tensor/buffer owners.
    if(@import("builtin").os.tag==.linux) {
        const raw=std.c.getenv("TF_DS_GRAPH_TRIM_HOST_HEAP") orelse return;
        if(std.mem.eql(u8,std.mem.span(raw),"1")) _=malloc_trim(1 << 20);
    }
}
pub fn shapeLimit(comptime name: [:0]const u8) usize {
    const limit=configured(name);
    return if(limit==0 and reserveBytes()>0) 8192 else limit;
}

pub fn retireCount(available:u64,reserve:u64,count:usize,minimum:usize) usize {
    if(reserve==0 or available==0 or available>=reserve+(256 << 20)) return 0;
    return @min(8,count-|minimum);
}

/// Read host availability without allocations, CUDA calls or file writes.
/// Called by the coordinated control frame, never by a graph hit.
pub fn availableBytes() ?u64 {
    if(@import("builtin").os.tag!=.linux) return null;
    const fd=std.c.open("/proc/meminfo",.{.ACCMODE=.RDONLY,.CLOEXEC=true});
    if(fd<0) return null;
    defer _=std.c.close(fd);
    var buf:[4096]u8=undefined;
    const n=std.c.read(fd,&buf,buf.len);
    if(n<=0) return null;
    return parseAvailable(buf[0..@intCast(n)]);
}

fn parseAvailable(raw:[]const u8) ?u64 {
    var lines=std.mem.splitScalar(u8,raw,'\n');
    while(lines.next()) |line| {
        if(!std.mem.startsWith(u8,line,"MemAvailable:")) continue;
        var fields=std.mem.tokenizeAny(u8,line[13..]," \t\r");
        const value=std.fmt.parseInt(u64,fields.next() orelse return null,10) catch return null;
        if(!std.mem.eql(u8,fields.next() orelse return null,"kB")) return null;
        return std.math.mul(u64,value,1024) catch null;
    }
    return null;
}

pub fn Policy(comptime Key: type) type {
    return struct {
        const Self = @This();
        // Lazy exponential aging uses logical replay events, not wall clocks.
        // All ranks receiving the same graph keys make identical decisions.
        const age_events: u64 = 4096;
        const Use = struct {
            last: u64,
            epoch: u64,
            frequency: u16 = 1,
            fn aged(u: Use, epoch: u64) u16 {
                const elapsed = epoch -| u.epoch;
                return if (elapsed >= 16) 0 else u.frequency >> @as(u4, @intCast(elapsed));
            }
        };
        gpa: std.mem.Allocator,
        limit: usize,
        clock: u64 = 0,
        seen: std.AutoHashMapUnmanaged(Key, Use) = .empty,
        hits: u64 = 0,
        misses: u64 = 0,
        evictions: u64 = 0,

        pub fn init(gpa: std.mem.Allocator, limit: usize) Self {
            return .{ .gpa=gpa, .limit=limit };
        }
        pub fn deinit(s: *Self) void { s.seen.deinit(s.gpa); }
        pub fn forget(s:*Self,key:Key) void {
            if(s.seen.remove(key)) s.evictions +|= 1;
        }
        pub fn coldest(s:*Self,excluded:[]const Key) ?Key {
            var victim:?Key=null;
            var oldest:u64=std.math.maxInt(u64);
            var least:u16=std.math.maxInt(u16);
            var it=s.seen.iterator();
            outer: while(it.next()) |e| {
                for(excluded) |key| if(std.meta.eql(key,e.key_ptr.*)) continue :outer;
                const frequency=e.value_ptr.aged(s.clock/age_events);
                if(frequency<least or (frequency==least and e.value_ptr.last<oldest)) {
                    least=frequency;oldest=e.value_ptr.last;victim=e.key_ptr.*;
                }
            }
            return victim;
        }
        pub fn hit(s: *Self, key: Key) void {
            if (s.limit==0) return;
            s.clock +|= 1;
            if(s.seen.getPtr(key)) |use| {
                const epoch = s.clock / age_events;
                use.frequency = use.aged(epoch) +| 1;
                use.epoch = epoch;
                use.last = s.clock;
                s.hits +|= 1;
            }
        }
        /// Call only on a graph miss. Destroy the returned shape's graph
        /// descriptors, preserving every captured tensor/buffer owner.
        pub fn admit(s: *Self, key: Key) !?Key {
            if(s.limit==0) return null;
            if(s.seen.contains(key)) { s.hit(key); return null; }
            s.misses +|= 1;
            var victim: ?Key=null;
            if(s.seen.count()>=s.limit) {
                victim=s.coldest(&.{});
            }
            s.clock +|= 1;
            // Allocate first: an allocation error cannot forget an owner.
            try s.seen.put(s.gpa,key,.{.last=s.clock,.epoch=s.clock/age_events});
            if(victim) |v| {
                _=s.seen.remove(v);
                s.evictions +|= 1;
            }
            return victim;
        }
    };
}

test "hot shapes survive and the cold whole shape leaves" {
    var p=Policy(u32).init(std.testing.allocator,2);defer p.deinit();
    try std.testing.expectEqual(null,try p.admit(1));
    try std.testing.expectEqual(null,try p.admit(2));
    p.hit(1);
    try std.testing.expectEqual(@as(?u32,2),try p.admit(3));
    try std.testing.expect(p.seen.contains(1));
    try std.testing.expect(p.seen.contains(3));
    try std.testing.expectEqual(@as(usize,2),p.seen.count());
}
test "zero preserves the old behavior without allocating on hits or misses" {
    var p=Policy(u32).init(std.testing.allocator,0);defer p.deinit();
    for(0..100) |i| { p.hit(@intCast(i));try std.testing.expectEqual(null,try p.admit(@intCast(i))); }
    try std.testing.expectEqual(@as(usize,0),p.seen.count());
    try std.testing.expectEqual(@as(u64,0),p.clock);
}
test "many shapes remain bounded and returning shape is admitted again" {
    var p=Policy(u32).init(std.testing.allocator,16);defer p.deinit();
    for(0..10000) |i| {
        _=try p.admit(@intCast(i));
        try std.testing.expect(p.seen.count()<=16);
    }
    try std.testing.expectEqual(@as(?u32,9984),try p.admit(0));
    try std.testing.expect(p.seen.contains(0));
}
test "failed insertion retains every previous owner" {
    var failing=std.testing.FailingAllocator.init(std.testing.allocator,.{});
    var p=Policy(u32).init(failing.allocator(),1);defer p.deinit();
    _=try p.admit(7);
    // The first hashmap has capacity 8; force its next growth to fail.
    for(8..13) |i| try p.seen.put(p.gpa,@intCast(i),.{.last=i,.epoch=0});
    failing.fail_index=failing.alloc_index;
    const before=p.seen.count();
    try std.testing.expectError(error.OutOfMemory,p.admit(99));
    try std.testing.expectEqual(before,p.seen.count());
    try std.testing.expect(p.seen.contains(7));
}

test "frequent older shape survives one-use newer shapes" {
    var p=Policy(u32).init(std.testing.allocator,3);defer p.deinit();
    _=try p.admit(1);
    for(0..100) |_| p.hit(1);
    _=try p.admit(2);_=try p.admit(3);
    try std.testing.expectEqual(@as(?u32,2),try p.admit(4));
    try std.testing.expect(p.seen.contains(1));
    try std.testing.expectEqual(@as(u64,1),p.evictions);
}

test "old popularity decays when the workload changes" {
    var p=Policy(u32).init(std.testing.allocator,2);defer p.deinit();
    _=try p.admit(1);_=try p.admit(2);
    for(0..1000) |_| p.hit(1);
    for(0..20*4096) |_| p.hit(2);
    try std.testing.expectEqual(@as(?u32,1),try p.admit(3));
    try std.testing.expect(p.seen.contains(2));
}

test "equal logical graph traffic makes equal rank decisions" {
    var a=Policy(u32).init(std.testing.allocator,32);defer a.deinit();
    var b=Policy(u32).init(std.testing.allocator,32);defer b.deinit();
    for(0..20000) |i| {
        const key:u32=@intCast(if(i%4==0) 7 else i%71);
        try std.testing.expectEqual(try a.admit(key),try b.admit(key));
    }
    try std.testing.expectEqual(a.hits,b.hits);
    try std.testing.expectEqual(a.evictions,b.evictions);
}

test "pressure selection skips the protected and already selected shapes" {
    var p=Policy(u32).init(std.testing.allocator,100);defer p.deinit();
    _=try p.admit(1);_=try p.admit(2);_=try p.admit(3);
    p.hit(1);
    try std.testing.expectEqual(@as(?u32,2),p.coldest(&.{}));
    try std.testing.expectEqual(@as(?u32,3),p.coldest(&.{2}));
    p.forget(2);
    try std.testing.expectEqual(@as(?u32,3),p.coldest(&.{}));
}

test "MemAvailable is parsed in KiB with invalid input rejected" {
    try std.testing.expectEqual(@as(?u64,1048576),parseAvailable("MemFree: 99 kB\nMemAvailable: 1024 kB\n"));
    try std.testing.expectEqual(null,parseAvailable("MemAvailable: 1 MB\n"));
    try std.testing.expectEqual(null,parseAvailable("MemAvailable: 18446744073709551615 kB\n"));
    try std.testing.expectEqual(null,parseAvailable("MemFree: 999 kB\n"));
}

test "memory pressure preserves all graphs with headroom and trims bounded cold batches" {
    const reserve=1024 << 20;
    try std.testing.expectEqual(@as(usize,0),retireCount(17 << 30,reserve,443,32));
    try std.testing.expectEqual(@as(usize,0),retireCount(1280 << 20,reserve,443,32));
    try std.testing.expectEqual(@as(usize,8),retireCount(1100 << 20,reserve,443,32));
    try std.testing.expectEqual(@as(usize,3),retireCount(900 << 20,reserve,35,32));
    try std.testing.expectEqual(@as(usize,0),retireCount(900 << 20,reserve,32,32));
    try std.testing.expectEqual(@as(usize,0),retireCount(0,reserve,443,32));
}
