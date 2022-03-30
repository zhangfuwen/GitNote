---

title: Linux内核中的trace

---

# 代码

 https://github.com/facebookincubator/oculus-linux-kernel

# mesa 中freedreno代码

```plantuml

@startuml
class msm_ringbuffer {
    struct fd_ringbuffer base;
}

class fd_ringbuffer {
    struct fd_ringbuffer_funcs * funcs
    ---
    .funcs = ring_funcs
}



msm_ringbuffer -|> fd_ringbuffer

class ring_funcs {
   .grow = msg_ringbuffer_grow,
   .emit_reloc = msg_ringbuffer_emit_reloc,
   .emit_reloc_ring = msg_ringbuffer_emit_reloc_ring,
   .cmd_count = msg_ringbuffer_cmd_count,
   .destroy = msg_ringbuffer_destroy,
}

fd_ringbuffer "ring_funcs" ..> ring_funcs

class fd_pipe {
     fd_device *dev;
    fd_pipe_id id;
    uint32_t gpu_id;
    int32_t refcnt;
    struct fd_pipe_funcs * funcs;
---
    .funcs = legacy_funcs
}

class fd_pipe_funcs {
    .ringbuffer_new_object = msm_ringbuffer_new_object,
    .submit_new = msm_submit_new,
    .get_param = msm_pipe_get_param,
    .wait = msm_pipe_wait,
    .destroy = msm_pipe_destroy,
}

fd_pipe "legacy_funcs" ..> fd_pipe_funcs

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

msm_pipe -|> fd_pipe

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

msm_device -|> fd_device

class fd_device_funcs {
    		.bo_new_handle = msm_bo_new_handle,
		.bo_from_handle = msm_bo_from_handle,
		.pipe_new = msm_pipe_new,
		.destroy = msm_device_destroy,
}





@enduml

```

### draw的调用流程

```plantuml
@startuml
app -> _mesa_: _mesa_DrawElements
_mesa_ -> st: Driver->st_draw_vbo
st -> cso: cso_draw_vbo
cso -> pipe: draw_vbo(fd6_draw_vbo)
pipe -> fd_draw: draw_emit
@enduml
```