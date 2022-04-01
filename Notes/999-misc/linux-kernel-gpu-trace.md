---

title: adreno 显卡驱动分析分析(mesa and kmd)

---

# 代码

 https://github.com/facebookincubator/oculus-linux-kernel

# mesa 中freedreno代码

```plantuml

@startuml

package ringbuffer <<Rectangle>> {
    class msm_ringbuffer
    class fd_ringbuffer
}

class msm_ringbuffer {
    struct fd_ringbuffer base;
    union u
---
    fd_pipe * pipe;
    fd_bo * reloc_bos;//array
    set * ring_set;
---
    fd_submit * submit
    msm_cmd* cmds;//array
---
    msm_cmd *cmd;//cur
    fd_bo * ring_bo;
}

class fd_ringbuffer {
    struct fd_ringbuffer_funcs * funcs
    ---
    .funcs = ring_funcs
}



fd_ringbuffer <|-- msm_ringbuffer

interface ring_funcs {
   .grow = msg_ringbuffer_grow,
   .emit_reloc = msg_ringbuffer_emit_reloc,
   .emit_reloc_ring = msg_ringbuffer_emit_reloc_ring,
   .cmd_count = msg_ringbuffer_cmd_count,
   .destroy = msg_ringbuffer_destroy,
}

fd_ringbuffer::ring_funcs .> ring_funcs


package pipe <<Rectangle>> {
    class fd_pipe
    class msm_pipe
}

class fd_pipe {
 
    fd_pipe_id id;
    uint32_t gpu_id;
    int32_t refcnt;
    struct fd_pipe_funcs * funcs;
---
    .funcs = legacy_funcs
    fd_device *dev;
}

interface fd_pipe_funcs {
    .ringbuffer_new_object = msm_ringbuffer_new_object,
    .submit_new = msm_submit_new,
    .get_param = msm_pipe_get_param,
    .wait = msm_pipe_wait,
    .destroy = msm_pipe_destroy,
}

fd_pipe::legacy_funcs .> fd_pipe_funcs

class msm_pipe {
    fd_pipe base;
    ---
    	uint32_t pipe;
	uint32_t gpu_id;
	uint64_t gmem_base;
	uint32_t gmem;
	uint32_t chip_id;
	uint32_t queue_id;
	struct slab_parent_pool ring_pool;
}

fd_pipe <|-- msm_pipe

class fd_submit_funcs {
	*new_ringbuffer
	* **flush**
	*destroy
}

class fd_submit {
	fd_pipe *pipe;
	fd_submit_funcs *funcs;
}

msm_ringbuffer::submit .> fd_submit
fd_submit::funcs .> fd_submit_funcs

package device <<Rectangle>> {
    class fd_device
    class msm_device
}

class fd_device {
    int fd
    fd_version version
    int32_t refcnt
    fd_device_funcs * funcs
    fd_bo_cache bo_cache
    fd_bo_cache ring_cache;
}
class msm_device {
    fd_device base;
    fd_bo_cache ring_cache;
}

fd_device <|-- msm_device

interface fd_device_funcs {
    		.bo_new_handle = msm_bo_new_handle,
		.bo_from_handle = msm_bo_from_handle,
		.pipe_new = msm_pipe_new,
		.destroy = msm_device_destroy,
}

fd_pipe::dev ..> fd_device
msm_ringbuffer::pipe .> fd_pipe
fd_device::funcs .> fd_device_funcs

package context <<Rectangle>> {
    class pipe_context
    class fd_context
}
pipe_context <|-- fd_context
class pipe_context {
    pipe_screen * screen
    void * priv
    void * draw
    u_upload_mgr *stream_uploader
    u_upload_mgr *const_uploader
    
    -- func pointers --
    *destroy
    *draw_vbo
    *set_vertex_buffers
    *set_stream_out_targets
    *blit
    *clear
    *create_surface
    *surface_destroy
    *buffer_subdata
    *texture_subdata
    *resource_commit
    *launch_grid
    a very long list
}

class fd_context {
    pipe_context base;
    list_head node;
    -- uplink --
    fd_device *dev;
    fd_screen * screen;
    fd_pipe * pipe;
    
    -- other contexts --
    blitter_context *blitter
    primconvert_context * primconvert
    
    -- current batch --
    fd_batch * batch

    -- function pointers --
    emit_tile_init
    emit_tile_prep
    emit_tile_mem2gmem
    emit_tile_rendergrep
    emit_tile
    emit_title_gmem2mem
    emit_tile_fini
    emit_sysmem_prep
    emit_sysmem_fini

    draw_vbo
    clear
    blit
    ts_to_ns
    launch_grid
    *query*
    -- end --
    
    
}

fd_context::dev .> fd_device

class cso_context {
    pipe_context *pipe;
}
cso_context::pipe -left-> pipe_context

class st_context {
    st_context_iface iface;
    gl_context *ctx;
    pipe_context *pipe;
    draw_context *draw;

    cso_context *cso_context
}
st_context::pipe -left-> pipe_context


class fd_batch {
    	unsigned num_bins_per_pipe;
	unsigned prim_strm_bits;
	unsigned draw_strm_bits;
	--draw pass cmdstream--
	fd_ringbuffer *draw;
	--binning pass cmdstream--
	fd_ringbuffer *binning;
	--tiling/gmem (IB0) cmdstream--
	fd_ringbuffer *gmem;

	--epilogue cmdstream--
	fd_ringbuffer *epilogue;

	// TODO maybe more generically 
        //split out clear 
        //and clear_binning rings?
	fd_ringbuffer *lrz_clear;
	fd_ringbuffer *tile_setup;
	fd_ringbuffer *tile_fini;

}
fd_context::batch .left.> fd_batch
fd_batch .left.> fd_ringbuffer


@enduml

```

## draw的调用流程

```plantuml
@startuml
app -> _mesa_: _mesa_DrawElements
_mesa_ -> st: Driver->st_draw_vbo
st -> cso: cso_draw_vbo
cso -> pipe: draw_vbo(fd6_draw_vbo)
pipe -> fd_draw: draw_emit
@enduml
```


### mesa adreno code

```plantuml

st -> fd_context: fd6_emit_title_init
fd_context -> fd6_gmem: emit_binning_pass 

```


# kernel code

## 流程
```plantuml
@startuml

actor platform
participant adreno_device
participant dispatcher
participant ringbuffer
participant trace
participant profile

platform -> dispatcher++ : adreno_dispatcher_queue_cmds(drawobjs[])
loop foreach drawobj
dispatcher ->  dispatcher++ : _queue_xxxobj
dispatcher -> dispatcher++ : _queue_drawobj(drawobj)
dispatcher -> dispatcher : ctxt->drawqueue[tail] = drawobj
dispatcher -> trace : trace_adreno_cmdbatch_queued()
dispatcher--
dispatcher--
dispatcher--

end loop


== kthread_worker ==

loop kthread_worker

platform -> dispatcher++ : adreno_dispatcher_work()

== retire ==

loop foreach rb : device.ringbuffers

dispatcher  -> dispatcher++ :  adreno_dispatch_process_drawqueue(rb->dispatch_q)
dispatcher  -> dispatcher++ :  adreno_dispatch_retire_drawqueue(rb->dispatch_q)

loop foreach drawobj : dispatch_q->cmd_q
dispatcher -> dispatcher++ : retire_obj(drawobj)
dispatcher -> profile : cmdobj_profile_ticks()
dispatcher -> trace : trace_adreno_cmdbatch_retired()
dispatcher--
end loop
dispatcher--

== submit ==
dispatcher -> dispatcher++ :  _adreno_dispatcher_issue_cmds(dev)

loop foreach context
dispatcher -> dispatcher++ : dispatcher_context_sendcmds(drawctx)
loop foreach drawobj : ctxt->drawqueue[]
dispatcher -> dispatcher: _retire_markerobj
dispatcher -> dispatcher: _retire_syncobj
dispatcher -> dispatcher++: sendcmd(drawobj as cmdobj)
dispatcher -> ringbuffer++: adreno_ringbuffer_submitcmd
ringbuffer -> profile: kernel started 
ringbuffer -> profile: user submitted 
loop foreach ib in cmdobj
ringbuffer -> ringbuffer: copy to cmd
end loop
ringbuffer -> ringbuffer++: adreno_drawctxt_switch(rb, drawctxt);
ringbuffer -> profile: kernel retired 
ringbuffer -> profile: user retired 
ringbuffer -> ringbuffer++: adreno_ringbuffer_addcmds(cmd)
ringbuffer -> profile : adreno_profile_pre_ib_processing //prepend记录perfcounter指令
ringbuffer -> profile : adreno_profile_post_ib_processing //记录perfcounter
ringbuffer -> ringbuffer : adreno_ringbufferr_submit,修改write ptr
ringbuffer--

ringbuffer -> trace: trace_kgsl_issueibcmds(timestamps, numibs)
ringbuffer--

dispatcher -> trace: trace_adreno_cmdbatch_submitted
dispatcher -> dispatcher : dispatch_q->cmd_q[tail] = cmdobj


end loop

dispatcher--
end loop
dispatcher--

end loop

@enduml



```

## 类图

```plantuml

@startuml
class kgsl_drawobj {
	struct kgsl_device *device;
	struct kgsl_context *context;
	uint32_t type;
	uint32_t timestamp;
	unsigned long flags;
	struct kref refcount;
}

class kgsl_drawobj_cmd {
	struct kgsl_drawobj base;
	unsigned long priv;
	unsigned int global_ts;
	unsigned long fault_policy;
	unsigned long fault_recovery;
	struct list_head cmdlist;
	struct list_head memlist;
	unsigned int marker_timestamp;
	struct kgsl_mem_entry *profiling_buf_entry;
	uint64_t profiling_buffer_gpuaddr;
	unsigned int profile_index;
	uint64_t submit_ticks;
}

kgsl_drawobj <|-down- kgsl_drawobj_cmd

class adreno_dispatcher_drawqueue {
	struct kgsl_drawobj_cmd *cmd_q[ADRENO_DISPATCH_DRAWQUEUE_SIZE];
	unsigned int inflight;
	unsigned int head;
	unsigned int tail;
	int active_context_count;
	unsigned long expires;
}

adreno_dispatcher_drawqueue "has many" -> kgsl_drawobj_cmd


class adreno_ringbuffer {
    struct adreno_dispatcher_drawqueue dispatch_q
}

adreno_ringbuffer *-down->  adreno_dispatcher_drawqueue : dispatch_q

class adreno_device {
	struct kgsl_device dev;  
	unsigned long priv;
	unsigned int *gpmu_cmds;
	struct adreno_ringbuffer ringbuffers[KGSL_PRIORITY_MAX_RB_LEVELS];
}

adreno_device *-down-> adreno_ringbuffer : has 4

class adreno_context {
	struct kgsl_context base;

	struct kgsl_drawobj *drawqueue[ADRENO_CONTEXT_DRAWQUEUE_SIZE];
	unsigned int drawqueue_head;
	unsigned int drawqueue_tail;
}

adreno_context -left-> kgsl_drawobj : drawqueue
@enduml

```


## drawobj如何retire

rb.dispatch_q->cmd_q[i].上的每个drawobj，

```c
struct kgsl_devmemstore {
       volatile unsigned int soptimestamp;
       unsigned int sbz;
       volatile unsigned int eoptimestamp;
       unsigned int sbz2;
       volatile unsigned int preempted;
       unsigned int sbz3;
       volatile unsigned int ref_wait_ts;
       unsigned int sbz4;
       unsigned int current_context;
       unsigned int sbz5;
};
device为每个context存了一个devmemstore结构，gpu会把endofpipe timestamp写进去。
如果endofpipe timestamp大于drawobj->timestamp，说明已经处理完了。

向endofpipe timestamp写入值的微码是在`adreno_ringbuffer_addcmds`函数中追加的：

*ringcmds++ = cp_mem_packet(adreno_dev, CP_EVENT_WRITE, 3, 1);
if (drawctxt || is_internal_cmds(flags))
       *ringcmds++ = CACHE_FLUSH_TS | (1 << 31);
else
       *ringcmds++ = CACHE_FLUSH_TS;

if (drawctxt && !is_internal_cmds(flags)) {
       ringcmds += cp_gpuaddr(adreno_dev, ringcmds,
              MEMSTORE_ID_GPU_ADDR(device, context_id, eoptimestamp));
       *ringcmds++ = timestamp;

       /* Write the end of pipeline timestamp to the ringbuffer too */
       *ringcmds++ = cp_mem_packet(adreno_dev, CP_EVENT_WRITE, 3, 1);
       *ringcmds++ = CACHE_FLUSH_TS;
       ringcmds += cp_gpuaddr(adreno_dev, ringcmds,
              MEMSTORE_RB_GPU_ADDR(device, rb, eoptimestamp));
       *ringcmds++ = rb->timestamp;
} else {
```