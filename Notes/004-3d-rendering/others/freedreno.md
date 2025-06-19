```plantuml

class fd_batch {
	seqno : int
	idx : int
	num_draws : int
	num_vertices : int
	num_bins_per_pipe
	prim_strm_bits
	draw_strm_bits
	
	draw : draw pass cmdstream
	binning: binning pass cmdstream
	gmem : tiling/gmem(IB0) cmdstream

	fd_batch_create()
	_flush()
	_add_dep()
	_resource_write()
	_resource_read_slowpath()
	_check_size()
	_describe()
	_destroy()

}

fd_context --|> pipe_context
pipe_context --|> pipe_screen

class pipe_context {
	.draw_vbo
	.flush
	.emit_string_marker
}

class fd_context {
	draw_vbo : fd5_draw_vbo
	clear : fd5_clear

	emit_tile_init
	emit_tile_prep
	emit_tile_mem2gmem
	emit_tile_renderprep
	emit_tile
	emit_tile_gmem2mem
	emit_tile_fini
	emit_system_prep
	emit_system_fini

}
```

```plantuml
fd5_draw.c -> fd5_draw.c++: fd3_draw_vbo()
fd5_draw.c -> fd5_draw.c++ : draw_impl()
fd5_draw.c -> freedreno_draw.h : fd5_draw_emit()
```

```

render_tiles()
	emit_tile_init
	for_each_bin:
		emit_tile_prep
		(restore?)emit_tile_mem2gmem
		emit_tile_renderprep
		emit_tile
		emit_tile_gmem2mem
	emit_tile_fini


fd_gmem_render_tiles
	if(sysmem) 
		render_sysmem
	else
		render_tiles
	flush_ring

batch_flush
	fd_fence_ref(batch->fence)
	fd_gmem_render_tiles
	batch_reset_resources(batch)
	fd_bc_invalidate_batch

fd_batch_flush
	fd_batch_reference
	batch_flush
	~fd_batch_reference


fd_draw_fbo -> 
	batch_draw_tracking -> 
		resource_written -> 
			fd_batch_resource_write -> 
				flush_write_batch -> 
					fd_batch_flush
fd_launch_grid -> fd_batch_flush (compute shading)


fd_gmem_render_tiles()


util_draw_indirect
	for_each_draw_call
		pipe->draw_vbo(info)


st_draw_vbo (st_draw.c state_tracker)
	for_each_mesa_prims
		pipe->draw_vbo


kernel code:
rgmu_irq_handler
oob_irq_handler
hfi_irq_handler
adreno_irq_handler
kgsl_irq_hnadler



```