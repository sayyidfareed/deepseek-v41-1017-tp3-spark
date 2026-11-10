// Modified for the DeepSeek 1017 TP3 integration; upstream copyrights retained. See CREDITS.md and provenance.json.
//! LakyFx TP3 v129/v130 medium-batch port of TensorFold PR1467/1534 (Apache-2.0).
//! Source/changes: zig/kernels/cuda/exl3_mid_batch/PROVENANCE.json.
//! No allocator, device allocation, environment lookup or clock in launch().
const std=@import("std");
const cuda=@import("cuda");
const weights=@import("weights.zig");

pub const Mpg=struct {halves:u32,warps:u32,p:u32};
pub const mpg_variants=blk:{
    var vs:[16]Mpg=undefined;var i:usize=0;
    for([_]u32{6,8,10,12}) |h| for([_]u32{1,2}) |p| for([_]u32{4,8}) |w| {
        vs[i]=.{.halves=h,.warps=w,.p=p};i+=1;
    };break :blk vs;
};
pub const Wc=struct {halves:u32,mt:u32,ks:u32,fold:bool};
pub const wc_variants=blk:{
    var vs:[16]Wc=undefined;var i:usize=0;
    for([_]u32{8,12}) |h| for([_]bool{false,true}) |f| for([_]u32{2,4}) |mt| for([_]u32{2,4}) |ks| {
        vs[i]=.{.halves=h,.mt=mt,.ks=ks,.fold=f};i+=1;
    };break :blk vs;
};
pub fn wcShared(v:Wc) u32 {
    const stage=v.ks*8*4*v.halves*4+v.ks*16*v.mt*32;
    return @max(4*stage,16*v.mt*128*4);
}
pub const Pick=union(enum){mpg:usize,wc:usize};
pub fn pick(l:weights.Linear,m:usize,sk:u32,wk:u32,shared_limit:u32) ?Pick {
    if(m<17 or m>128 or sk==0 or (wk!=4 and wk!=8) or l.n==0 or l.n%128!=0 or l.k%128!=0) return null;
    if(l.k/16%(sk*wk)!=0) return null;
    const per=l.k/16/sk/wk;const nb=l.n/128;
    if((l.k2==8 or l.k2==12) and per%2==0) {
        const fold=sk>1 and nb>=32;
        if(fold or l.k2!=8 or m<=32 or m>40) {
            const want:Wc=.{.halves=l.k2,.fold=fold,.ks=if(per%4==0 and !(l.k2==12 and m>96)) 4 else 2,
                .mt=if(m<=32 or (m>64 and m<=96) or (fold and m>96 and nb>=64)) 2 else 4};
            for(wc_variants,0..) |v,i| if(std.meta.eql(v,want) and wcShared(v)<=shared_limit) return .{.wc=i};
        }
    }
    for(mpg_variants,0..) |v,i| if(v.halves==l.k2 and v.warps==wk and v.p==(if(m>48) @as(u32,2) else 1)) return .{.mpg=i};
    return null;
}

// CUDA takes ONE by-value struct parameter, not an array of scalar pointers.
// WcArgs appends warps at offset 92 (before LinearArgs' trailing padding).
pub const LinearArgs=extern struct {
    xh:u64,trellis:u64,stride_k:i64,stride_nb:i64,svh:u64,bias:u64=0,y:u64,z:u64,counters:u64,
    y_dtype:i32,n:i32,sk:i32,m:i32,k:i32,discard:i32,
};
pub const WcArgs=extern struct {
    xh:u64,trellis:u64,stride_k:i64,stride_nb:i64,svh:u64,bias:u64=0,y:u64,z:u64,counters:u64,
    y_dtype:i32,n:i32,sk:i32,m:i32,k:i32,warps:i32,discard:i32,
};
comptime {
    std.debug.assert(@sizeOf(LinearArgs)==96 and @sizeOf(WcArgs)==104);
    std.debug.assert(@offsetOf(WcArgs,"warps")==92 and @offsetOf(LinearArgs,"discard")==92 and @offsetOf(WcArgs,"discard")==96);
}
pub const Kernels=struct {
    module:cuda.Module,
    mpg:[16]cuda.Function=undefined,
    wc:[16]?cuda.Function=@splat(null),
    shared_limit:u32,
    pub fn load(d:*const cuda.Driver,blob:[]const u8,max_shared:u32) !Kernels {
        @setEvalBranchQuota(200000);
        // Device-reported block limit, with space for this kernel's static state.
        var k:Kernels=.{.module=try cuda.Module.load(d,blob),.shared_limit=max_shared-|1024};errdefer k.module.unload();
        inline for(mpg_variants,0..) |v,i| k.mpg[i]=try k.module.function(std.fmt.comptimePrint("tf_exl3_mpg_{d}_{d}_{d}",.{v.halves,v.warps,v.p}));
        inline for(wc_variants,0..) |v,i| {
            if(wcShared(v)<=k.shared_limit) {
                const f=try k.module.function(std.fmt.comptimePrint("tf_exl3_wc_{d}_{d}_{d}_{d}",.{v.halves,v.mt,v.ks,@intFromBool(v.fold)}));
                try f.allowDynamicShared(wcShared(v));k.wc[i]=f;
            }
        }
        return k;
    }
    pub fn unload(k:*Kernels) void {k.module.unload();k.*=undefined;}
    /// Eligible singleton calls only: keep all existing grouped/fused epilogues.
    pub fn attempt(k:*const Kernels,stream:cuda.Stream,c:anytype,m:usize,z:u64,sk:u32,wk:u32,discard:bool) !bool {
        if(c.rot!=null or c.rope!=null or c.ldy!=c.layer.n) return false;
        const how=pick(c.layer,m,sk,wk,k.shared_limit) orelse return false;
        const l=c.layer;const tw=@as(i64,l.k2)*4;
        const a:LinearArgs=.{.xh=c.xh,.trellis=l.words,.stride_k=8*tw,.stride_nb=@divExact(@as(i64,l.k),16)*8*tw,
            .svh=l.svh,.y=c.y,.z=if(sk>1) z else 0,.counters=c.counters,.y_dtype=@intFromEnum(c.y_dtype),
            .n=@intCast(l.n),.sk=@intCast(sk),.m=@intCast(m),.k=@intCast(l.k),.discard=@intFromBool(discard and sk>1)};
        var args:cuda.Args=.{};
        switch(how) {
            .mpg=>|i| {
                args.add(a);const rows=16*mpg_variants[i].p;
                try cuda.launch.launch(k.mpg[i],.{.grid=.{.x=@intCast(l.n/128*((m+rows-1)/rows)),.y=sk},
                    .block=.{.x=wk*32},.shared=@intCast(wk*@min(m,8)*128*4)},stream,&args);
            },
            .wc=>|i| {
                var w:WcArgs=undefined;
                inline for(@typeInfo(LinearArgs).@"struct".field_names) |name| @field(w,name)=@field(a,name);
                w.warps=@intCast(wk);args.add(w);
                const v=wc_variants[i];const rows=16*v.mt;
                try cuda.launch.launch(k.wc[i].?,.{.grid=.{.x=@intCast(l.n/128*((m+rows-1)/rows)),.y=if(v.fold) 1 else sk},
                    .block=.{.x=256},.shared=wcShared(v)},stream,&args);
            },
        }
        return true;
    }
};

test "dispatch stays inside donor contracts and does not change fixed plans" {
    var l:weights.Linear=.{.words=0,.suh=0,.svh=0,.k=5120,.n=4096,.k2=8};
    for(0..150) |m| {
        const p=pick(l,m,8,4,99*1024-1024);
        try std.testing.expect((p!=null)==(m>=17 and m<=128));
        if(p) |q| switch(q){.wc=>|i|try std.testing.expect(wcShared(wc_variants[i])<=98304),else=>{}};
    }
    try std.testing.expectEqual(@as(u32,57344),wcShared(.{.halves=12,.mt=4,.ks=4,.fold=false}));
    l.k2=12;
    for([_]usize{33,49,64}) |m| {
        const p=pick(l,m,1,8,32768).?;
        try std.testing.expect(p==.mpg); // A deliberately constrained device must fall back.
    }
    l.k2=16;try std.testing.expect(pick(l,32,8,4,99*1024)==null);
    l.k2=8;try std.testing.expect(pick(l,32,8,2,99*1024)==null);

}

// Mia EAQ may contain half-bit widths outside the donor extension. Those
// remain on the existing grouped kernel, with no reinterpretation of weights.
test "Mia quant groups use a supported extension or preserve the old path" {
    for (2..17) |halves| {
        const l: weights.Linear = .{.words=0,.suh=0,.svh=0,.k=5120,.n=4096,.k2=@intCast(halves)};
        for ([_]usize{1,16,17,32,48,64,96,128,129}) |m| {
            const eligible=(halves==6 or halves==8 or halves==10 or halves==12) and m>=17 and m<=128;
            try std.testing.expect((pick(l,m,8,4,99*1024)!=null)==eligible);
        }
    }
}
