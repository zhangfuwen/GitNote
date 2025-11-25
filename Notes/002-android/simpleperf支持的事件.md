
# List of hw-cache events:
  # More cache events are available in `simpleperf list raw`.
  branch-load-misses
  branch-loads
  dTLB-load-misses
  dTLB-loads
  iTLB-load-misses
  iTLB-loads
  L1-dcache-load-misses
  L1-dcache-loads
  L1-icache-load-misses
  L1-icache-loads
  LLC-load-misses
  LLC-loads

# List of coresight etm events:
  cs_etm/autofdo/
  cs-etm                # CoreSight ETM instruction tracing

# List of hardware events:
  branch-instructions
  branch-misses
  bus-cycles
  cache-misses
  cache-references
  cpu-cycles
  instructions
  stalled-cycles-backend
  stalled-cycles-frontend

# List of pmu events:
  armv8_pmuv3/br_immed_retired/
  armv8_pmuv3/br_mis_pred/
  armv8_pmuv3/br_mis_pred_retired/
  armv8_pmuv3/br_pred/
  armv8_pmuv3/br_retired/
  armv8_pmuv3/br_return_retired/
  armv8_pmuv3/bus_access/
  armv8_pmuv3/bus_cycles/
  armv8_pmuv3/cid_write_retired/
  armv8_pmuv3/cnt_cycles/
  armv8_pmuv3/cpu_cycles/
  armv8_pmuv3/cti_trigout4/
  armv8_pmuv3/cti_trigout5/
  armv8_pmuv3/cti_trigout6/
  armv8_pmuv3/cti_trigout7/
  armv8_pmuv3/dtlb_walk/
  armv8_pmuv3/exc_return/
  armv8_pmuv3/exc_taken/
  armv8_pmuv3/inst_retired/
  armv8_pmuv3/inst_spec/
  armv8_pmuv3/itlb_walk/
  armv8_pmuv3/l1d_cache/
  armv8_pmuv3/l1d_cache_lmiss_rd/
  armv8_pmuv3/l1d_cache_refill/
  armv8_pmuv3/l1d_cache_wb/
  armv8_pmuv3/l1d_tlb/
  armv8_pmuv3/l1d_tlb_refill/
  armv8_pmuv3/l1i_cache/
  armv8_pmuv3/l1i_cache_lmiss/
  armv8_pmuv3/l1i_cache_refill/
  armv8_pmuv3/l1i_tlb/
  armv8_pmuv3/l1i_tlb_refill/
  armv8_pmuv3/l2d_cache/
  armv8_pmuv3/l2d_cache_allocate/
  armv8_pmuv3/l2d_cache_lmiss_rd/
  armv8_pmuv3/l2d_cache_refill/
  armv8_pmuv3/l2d_cache_wb/
  armv8_pmuv3/l2d_tlb/
  armv8_pmuv3/l2d_tlb_refill/
  armv8_pmuv3/l3d_cache/
  armv8_pmuv3/l3d_cache_allocate/
  armv8_pmuv3/l3d_cache_lmiss_rd/
  armv8_pmuv3/l3d_cache_refill/
  armv8_pmuv3/ld_align_lat/
  armv8_pmuv3/ldst_align_lat/
  armv8_pmuv3/ll_cache_miss_rd/
  armv8_pmuv3/ll_cache_rd/
  armv8_pmuv3/mem_access/
  armv8_pmuv3/mem_access_checked/
  armv8_pmuv3/mem_access_checked_rd/
  armv8_pmuv3/mem_access_checked_wr/
  armv8_pmuv3/op_retired/
  armv8_pmuv3/op_spec/
  armv8_pmuv3/pc_write_retired/
  armv8_pmuv3/remote_access/
  armv8_pmuv3/sample_collision/
  armv8_pmuv3/sample_feed/
  armv8_pmuv3/sample_filtrate/
  armv8_pmuv3/sample_pop/
  armv8_pmuv3/st_align_lat/
  armv8_pmuv3/stall/
  armv8_pmuv3/stall_backend/
  armv8_pmuv3/stall_backend_mem/
  armv8_pmuv3/stall_frontend/
  armv8_pmuv3/stall_slot/
  armv8_pmuv3/stall_slot_backend/
  armv8_pmuv3/stall_slot_frontend/
  armv8_pmuv3/sw_incr/
  armv8_pmuv3/trb_wrap/
  armv8_pmuv3/trcextout0/
  armv8_pmuv3/trcextout1/
  armv8_pmuv3/trcextout2/
  armv8_pmuv3/trcextout3/
  armv8_pmuv3/ttbr_write_retired/
  cs_etm/autofdo/

# List of raw events provided by cpu pmu:
  # Please refer to "PMU common architectural and microarchitectural event numbers"
  # and "ARM recommendations for IMPLEMENTATION DEFINED event numbers" listed in
  # ARMv9 manual for details.
  # A possible link is https://developer.arm.com/documentation/ddi0487.
  raw-ase-inst-spec (supported on cpu 4-7)              # Operation speculatively executed, Advanced SIMD
  raw-ase-spec (supported on cpu 0-7)           # Operation speculatively executed, Advanced SIMD
  raw-ase-sve-int16-spec (supported on cpu 0-7)         # Integer operation speculatively executed, Advanced SIMD or SVE 16-bit
  raw-ase-sve-int32-spec (supported on cpu 0-7)         # Integer operation speculatively executed, Advanced SIMD or SVE 32-bit
  raw-ase-sve-int64-spec (supported on cpu 0-7)         # Integer operation speculatively executed, Advanced SIMD or SVE 64-bit
  raw-ase-sve-int8-spec (supported on cpu 0-7)          # Integer operation speculatively executed, Advanced SIMD or SVE 8-bit
  raw-br-immed-mis-pred-retired (supported on cpu 0-7)          # Branch Instruction architecturally executed, mispredicted immediate
  raw-br-immed-pred-retired (supported on cpu 0-7)              # Branch Instruction architecturally executed, predicted immediate
  raw-br-immed-retired (supported on cpu 0-7)           # Branch Instruction architecturally executed, immediate
  raw-br-immed-spec (supported on cpu 0-3)              # Branch Speculatively executed, immediate branch
  raw-br-immed-taken-retired (supported on cpu 4-7)             # Branch Instruction architecturally executed, immediate, taken
  raw-br-ind-mis-pred-retired (supported on cpu 4-7)            # Branch Instruction architecturally executed, mispredicted indirect
  raw-br-ind-pred-retired (supported on cpu 4-7)                # Branch Instruction architecturally executed, predicted indirect
  raw-br-ind-retired (supported on cpu 0-7)             # Instruction architecturally executed, indirect branch
  raw-br-indirect-spec (supported on cpu 0-3)           # Branch Speculatively executed, indirect branch
  raw-br-indnr-mis-pred-retired (supported on cpu 0-7)          # Branch Instruction architecturally executed, mispredicted indirect excluding procedure return
  raw-br-indnr-pred-retired (supported on cpu 0-7)              # Branch Instruction architecturally executed, predicted indirect excluding procedure return
  raw-br-indnr-taken-retired (supported on cpu 0-7)             # Branch Instruction architecturally executed, indirect excluding procedure return, taken
  raw-br-mis-pred (supported on cpu 0-7)                # Branch instruction Speculatively executed, mispredicted or not predicted
  raw-br-mis-pred-retired (supported on cpu 0-7)                # Branch Instruction architecturally executed, mispredicted
  raw-br-pred (supported on cpu 0-7)            # Predictable branch instruction Speculatively executed
  raw-br-pred-retired (supported on cpu 0-7)            # Branch Instruction architecturally executed, predicted branch
  raw-br-retired (supported on cpu 0-7)         # Instruction architecturally executed, branch
  raw-br-return-mis-pred-retired (supported on cpu 0-7)         # Branch Instruction architecturally executed, mispredicted procedure return
  raw-br-return-pred-retired (supported on cpu 0-7)             # Branch Instruction architecturally executed, predicted procedure return
  raw-br-return-retired (supported on cpu 0-7)          # Branch Instruction architecturally executed, procedure return, taken
  raw-br-return-spec (supported on cpu 0-3)             # Branch Speculatively executed, procedure return
  raw-bus-access (supported on cpu 0-7)         # Bus access
  raw-bus-access-rd (supported on cpu 0-7)              # Bus access, read
  raw-bus-access-wr (supported on cpu 0-7)              # Bus access, write
  raw-bus-cycles (supported on cpu 0-7)         # Bus cycle
  raw-bus-req-rd (supported on cpu 0-3)         # Bus request, read
  raw-bus-req-rd-percyc (supported on cpu 0-3)          # Bus read transactions in progress
  raw-cas-far-spec (supported on cpu 4-7)               # Atomic memory Operation speculatively executed, Compare and Swap far
  raw-cas-near-pass (supported on cpu 4-7)              # Atomic memory Operation speculatively executed, Compare and Swap pass
  raw-cas-near-spec (supported on cpu 4-7)              # Atomic memory Operation speculatively executed, Compare and Swap near
  raw-cid-write-retired (supported on cpu 0-7)          # Instruction architecturally executed, Condition code check pass, write to CONTEXTIDR
  raw-cnt-cycles (supported on cpu 4-7)         # Constant frequency cycles
  raw-cortex-a520-dtlb-walk-hwprf (supported on cpu 0-3)                # Data TLB access, hardware prefetcher
  raw-cortex-a520-inst-spec-ldst-nuke (supported on cpu 0-3)            # Instruction re-executed, read-after-read hazard
  raw-cortex-a520-l1d-tlb-refill-ets (supported on cpu 0-3)             # L1D TLB refill due to ETS replay
  raw-cortex-a520-l1d-ws-mode (supported on cpu 0-3)            # L1 data cache write streaming mode
  raw-cortex-a520-l1d-ws-mode-entry (supported on cpu 0-3)              # L1 data cache entering write streaming mode
  raw-cortex-a520-l2d-cache-refill-hwprf-offset (supported on cpu 0-3)          # L2 cache refill due to L2 offset prefetcher
  raw-cortex-a520-l2d-cache-refill-hwprf-pattern (supported on cpu 0-3)         # L2 cache refill due to L2 pattern prefetcher
  raw-cortex-a520-l2d-cache-refill-hwprf-spatial (supported on cpu 0-3)         # L2 cache refill due to L2 spatial prefetcher
  raw-cortex-a520-l2d-cache-refill-hwprf-tlbd (supported on cpu 0-3)            # L2 cache refill due to L2 TLB prefetcher
  raw-cortex-a520-l2d-cache-stash-dropped (supported on cpu 0-3)                # L2 cache stash dropped
  raw-cortex-a520-l2d-s2-tlb (supported on cpu 0-3)             # L2 TLB IPA cache access
  raw-cortex-a520-l2d-s2-tlb-refill (supported on cpu 0-3)              # L2 TLB IPA cache refill
  raw-cortex-a520-l2d-walk-tlb (supported on cpu 0-3)           # L2 TLB walk cache access
  raw-cortex-a520-l2d-walk-tlb-refill (supported on cpu 0-3)            # L2 TLB walk cache refill
  raw-cortex-a520-l2d-ws-mode (supported on cpu 0-3)            # L2 cache write streaming mode
  raw-cortex-a520-l3d-cache-hwprf-offset (supported on cpu 0-3)         # L3 cache access due to L3 offset prefetcher
  raw-cortex-a520-l3d-cache-hwprf-stride (supported on cpu 0-3)         # L3 cache access due to L3 stride prefetcher
  raw-cortex-a520-l3d-ws-mode (supported on cpu 0-3)            # L3 cache write streaming mode
  raw-cortex-a520-ll-ws-mode (supported on cpu 0-3)             # Last level cache write streaming mode
  raw-cortex-a520-stall-backend-busy-vpu-hazard (supported on cpu 0-3)          # No operation issued due to the backend, VPU hazard
  raw-cortex-a520-stall-backend-ilock-addr (supported on cpu 0-3)               # No operation issued due to the backend, input dependency, address
  raw-cortex-a520-stall-backend-ilock-vpu (supported on cpu 0-3)                # No operation issued due to the backend, input dependency, Vector Processing Unit
  raw-cortex-a520-stall-slot-backend-ilock (supported on cpu 0-3)               # No operation sent for execution on a Slot due to the backend, input dependency
  raw-cpu-cycles (supported on cpu 0-7)         # Cycle
  raw-crypto-spec (supported on cpu 0-7)                # Operation speculatively executed, Cryptographic instruction
  raw-cti-trigout4 (supported on cpu 0-7)               # Cross-trigger Interface output trigger 4
  raw-cti-trigout5 (supported on cpu 0-7)               # Cross-trigger Interface output trigger 5
  raw-cti-trigout6 (supported on cpu 0-7)               # Cross-trigger Interface output trigger 6
  raw-cti-trigout7 (supported on cpu 0-7)               # Cross-trigger Interface output trigger 7
  raw-dmb-spec (supported on cpu 4-7)           # Barrier Speculatively executed, DMB
  raw-dp-spec (supported on cpu 0-7)            # Operation speculatively executed, integer data processing
  raw-dsb-spec (supported on cpu 4-7)           # Barrier Speculatively executed, DSB
  raw-dtlb-hwupd (supported on cpu 0-7)         # Data TLB hardware update of translation table
  raw-dtlb-step (supported on cpu 0-7)          # Data TLB translation table walk, step
  raw-dtlb-walk (supported on cpu 0-7)          # Data TLB access with at least one translation table walk
  raw-dtlb-walk-large (supported on cpu 0-7)            # Data TLB large page translation table walk
  raw-dtlb-walk-percyc (supported on cpu 0-7)           # Event in progress, DTLB_WALK
  raw-dtlb-walk-rw (supported on cpu 0-3)               # Data TLB demand access with at least one translation table walk
  raw-dtlb-walk-small (supported on cpu 0-7)            # Data TLB small page translation table walk
  raw-exc-dabort (supported on cpu 4-7)         # Exception taken, Data Abort or SError
  raw-exc-fiq (supported on cpu 0-7)            # Exception taken, FIQ
  raw-exc-hvc (supported on cpu 4-7)            # Exception taken, Hypervisor Call
  raw-exc-irq (supported on cpu 0-7)            # Exception taken, IRQ
  raw-exc-pabort (supported on cpu 4-7)         # Exception taken, Instruction Abort
  raw-exc-return (supported on cpu 0-7)         #  Instruction architecturally executed, Condition code check pass, exception return
  raw-exc-smc (supported on cpu 4-7)            # Exception taken, Secure Monitor Call
  raw-exc-svc (supported on cpu 4-7)            # Exception taken, Supervisor Call
  raw-exc-taken (supported on cpu 0-7)          # Exception taken
  raw-exc-trap-dabort (supported on cpu 4-7)            # Exception taken, Data Abort or SError not Taken locally
  raw-exc-trap-fiq (supported on cpu 4-7)               # Exception taken, FIQ not Taken locally
  raw-exc-trap-irq (supported on cpu 4-7)               # Exception taken, IRQ not Taken locally
  raw-exc-trap-other (supported on cpu 4-7)             # Exception taken, other traps not Taken locally
  raw-exc-trap-pabort (supported on cpu 4-7)            # Exception taken, Instruction Abort not Taken locally
  raw-exc-undef (supported on cpu 4-7)          # Exception taken, other synchronous
  raw-fp-dp-spec (supported on cpu 0-7)         # Floating-point operation speculatively executed, double precision
  raw-fp-fixed-ops-spec (supported on cpu 4-7)          # Non-scalable floating-point element ALU operations Speculatively executed
  raw-fp-hp-spec (supported on cpu 0-7)         # Floating-point operation speculatively executed, half precision
  raw-fp-scale-ops-spec (supported on cpu 4-7)          # Scalable floating-point element ALU operations Speculatively executed
  raw-fp-sp-spec (supported on cpu 0-7)         # Floating-point operation speculatively executed, single precision
  raw-inst-fetch (supported on cpu 0-7)         # Instruction memory access
  raw-inst-fetch-percyc (supported on cpu 0-7)          # Event in progress, INST_FETCH
  raw-inst-retired (supported on cpu 0-7)               # Instruction architecturally executed
  raw-inst-spec (supported on cpu 0-7)          # Operation speculatively executed
  raw-isb-spec (supported on cpu 4-7)           # Barrier Speculatively executed, ISB
  raw-itlb-hwupd (supported on cpu 0-7)         # Instruction TLB hardware update of translation table
  raw-itlb-step (supported on cpu 0-7)          # Instruction TLB translation table walk, step
  raw-itlb-walk (supported on cpu 0-7)          # Instruction TLB access with at least one translation table walk
  raw-itlb-walk-large (supported on cpu 0-7)            # Instruction TLB large page translation table walk
  raw-itlb-walk-percyc (supported on cpu 0-7)           # Event in progress, ITLB_WALK
  raw-itlb-walk-small (supported on cpu 0-7)            # Instruction TLB small page translation table walk
  raw-l1d-cache (supported on cpu 0-7)          # Level 1 data cache access
  raw-l1d-cache-hwprf (supported on cpu 0-3)            # Level 1 data cache hardware prefetch
  raw-l1d-cache-inval (supported on cpu 4-7)            # Level 1 data cache invalidate
  raw-l1d-cache-lmiss-rd (supported on cpu 0-7)         # Level 1 data cache long-latency read miss
  raw-l1d-cache-prf (supported on cpu 4-7)              # Level 1 data cache, preload or prefetch hit
  raw-l1d-cache-rd (supported on cpu 0-7)               # Level 1 data cache access, read
  raw-l1d-cache-refill (supported on cpu 0-7)           # Level 1 data cache refill
  raw-l1d-cache-refill-hwprf (supported on cpu 0-3)             # Level 1 data cache refill, hardware prefetch
  raw-l1d-cache-refill-inner (supported on cpu 0-7)             # Level 1 data cache refill, inner
  raw-l1d-cache-refill-outer (supported on cpu 0-7)             # Level 1 data cache refill, outer
  raw-l1d-cache-refill-prf (supported on cpu 4-7)               # Level 1 data cache refill, preload or prefetch hit
  raw-l1d-cache-refill-rd (supported on cpu 0-3)                # Level 1 data cache refill, read
  raw-l1d-cache-refill-wr (supported on cpu 0-3)                # Level 1 data cache refill, write
  raw-l1d-cache-rw (supported on cpu 4-7)               # Level 1 data cache demand access
  raw-l1d-cache-wb (supported on cpu 0-7)               #  Level 1 data cache write-back
  raw-l1d-cache-wr (supported on cpu 0-7)               # Level 1 data cache access, write
  raw-l1d-tlb (supported on cpu 0-7)            # Level 1 data TLB access
  raw-l1d-tlb-refill (supported on cpu 0-7)             # Level 1 data TLB refill
  raw-l1i-cache (supported on cpu 0-7)          #  Level 1 instruction cache access
  raw-l1i-cache-lmiss (supported on cpu 0-7)            # Level 1 instruction cache long-latency miss
  raw-l1i-cache-refill (supported on cpu 0-7)           # Level 1 instruction cache refill
  raw-l1i-tlb (supported on cpu 0-7)            # Level 1 instruction TLB access
  raw-l1i-tlb-refill (supported on cpu 0-7)             # Level 1 instruction TLB refill
  raw-l2d-cache (supported on cpu 0-7)          #  Level 2 data cache access
  raw-l2d-cache-allocate (supported on cpu 0-7)         # Level 2 data cache allocation without refill
  raw-l2d-cache-hwprf (supported on cpu 0-3)            # Level 2 data cache hardware prefetch
  raw-l2d-cache-inval (supported on cpu 4-7)            # Level 2 data cache invalidate
  raw-l2d-cache-lmiss-rd (supported on cpu 0-7)         # Level 2 data cache long-latency read miss
  raw-l2d-cache-prf (supported on cpu 4-7)              # Level 2 data cache, preload or prefetch hit
  raw-l2d-cache-rd (supported on cpu 0-7)               # Level 2 data cache access, read
  raw-l2d-cache-refill (supported on cpu 0-7)           # Level 2 data cache refill
  raw-l2d-cache-refill-hwprf (supported on cpu 0-3)             # Level 2 data cache refill, hardware prefetch
  raw-l2d-cache-refill-prf (supported on cpu 4-7)               # Level 2 data cache refill, preload or prefetch hit
  raw-l2d-cache-refill-rd (supported on cpu 0-7)                # Level 2 data cache refill, read
  raw-l2d-cache-refill-wr (supported on cpu 0-7)                # Level 2 data cache refill, write
  raw-l2d-cache-rw (supported on cpu 4-7)               # Level 2 data cache demand access
  raw-l2d-cache-wb (supported on cpu 0-7)               # Level 2 data cache write-back
  raw-l2d-cache-wb-clean (supported on cpu 4-7)         # Level 2 data cache write-back, cleaning and coherency
  raw-l2d-cache-wb-victim (supported on cpu 4-7)                # Level 2 data cache write-back, victim
  raw-l2d-cache-wr (supported on cpu 0-7)               # Level 2 data cache access, write
  raw-l2d-tlb (supported on cpu 0-7)            # Level 2 data TLB access
  raw-l2d-tlb-refill (supported on cpu 0-7)             # Level 2 data TLB refill
  raw-l3d-cache (supported on cpu 0-7)          # Level 3 data cache access
  raw-l3d-cache-allocate (supported on cpu 4-7)         # Level 3 data cache allocation without refill
  raw-l3d-cache-hwprf (supported on cpu 0-3)            # Level 3 data cache hardware prefetch
  raw-l3d-cache-lmiss-rd (supported on cpu 0-7)         # Level 3 data cache long-latency read miss
  raw-l3d-cache-rd (supported on cpu 0-7)               # Level 3 data cache access, read
  raw-l3d-cache-refill (supported on cpu 4-7)           # Level 3 data cache refill
  raw-l3d-cache-refill-rd (supported on cpu 0-3)                # Level 3 data cache refill, read
  raw-ld-align-lat (supported on cpu 0-7)               # Load with additional latency from alignment
  raw-ld-retired (supported on cpu 0-3)         # Instruction architecturally executed, Condition code check pass, load
  raw-ld-spec (supported on cpu 0-7)            # Operation speculatively executed, load
  raw-ldst-align-lat (supported on cpu 0-7)             # Access with additional latency from alignment
  raw-ldst-spec (supported on cpu 0-3)          # Operation speculatively executed, load or store
  raw-ll-cache-miss-rd (supported on cpu 0-7)           # Last level cache miss, read
  raw-ll-cache-rd (supported on cpu 0-7)                # Last level cache access, read
  raw-mem-access (supported on cpu 0-7)         #  Data memory access
  raw-mem-access-checked (supported on cpu 0-7)         # Checked data memory access
  raw-mem-access-rd (supported on cpu 0-7)              # Data memory access, read
  raw-mem-access-rd-checked (supported on cpu 0-7)              # Checked data memory access, read
  raw-mem-access-rd-percyc (supported on cpu 0-7)               # Event in progress, MEM_ACCESS_RD
  raw-mem-access-wr (supported on cpu 0-7)              # Data memory access, write
  raw-mem-access-wr-checked (supported on cpu 0-7)              # Checked data memory access, write
  raw-memory-error (supported on cpu 0-3)               # Local memory error
  raw-op-retired (supported on cpu 0-7)         # Micro-operation architecturally executed
  raw-op-spec (supported on cpu 0-7)            # Micro-operation Speculatively executed
  raw-pc-write-retired (supported on cpu 0-7)           # D, Instruction architecturally executed, Condition code check pass, Software change of the PC
  raw-pc-write-spec (supported on cpu 0-7)              # Operation speculatively executed, Software change of the PC
  raw-rc-ld-spec (supported on cpu 4-7)         # Release consistency operation Speculatively executed, Load-Acquire
  raw-rc-st-spec (supported on cpu 4-7)         # Release consistency operation Speculatively executed, Store-Release
  raw-remote-access (supported on cpu 4-7)              # Access to another socket in a multi-socket system
  raw-remote-access-rd (supported on cpu 0-3)           # Access to another socket in a multi-socket system, read
  raw-sample-collision (supported on cpu 4-7)           # Statistical Profiling sample collided with previous sample
  raw-sample-feed (supported on cpu 4-7)                # Statistical Profiling sample taken
  raw-sample-feed-br (supported on cpu 4-7)             # Statistical Profiling sample taken, branch
  raw-sample-feed-event (supported on cpu 4-7)          # Statistical Profiling sample taken, matching events
  raw-sample-feed-lat (supported on cpu 4-7)            # Statistical Profiling sample taken, exceeding minimum latency
  raw-sample-feed-ld (supported on cpu 4-7)             # Statistical Profiling sample taken, load
  raw-sample-feed-op (supported on cpu 4-7)             # Statistical Profiling sample taken, matching operation type
  raw-sample-feed-st (supported on cpu 4-7)             # Statistical Profiling sample taken, store
  raw-sample-filtrate (supported on cpu 4-7)            # Statistical Profiling sample taken and not removed by filtering
  raw-sample-pop (supported on cpu 4-7)         # Statistical Profiling sample population
  raw-st-align-lat (supported on cpu 0-7)               # Store with additional latency from alignment
  raw-st-retired (supported on cpu 0-3)         # Instruction architecturally executed, Condition code check pass, store  raw-st-spec (supported on cpu 0-7)            # Operation speculatively executed, store
  raw-stall (supported on cpu 0-7)              # No operation sent for execution
  raw-stall-backend (supported on cpu 0-7)              # No operation sent for execution due to the backend
  raw-stall-backend-busy (supported on cpu 0-7)         # Backend stall cycles, backend busy
  raw-stall-backend-cpubound (supported on cpu 4-7)             # Backend stall cycles, processor bound
  raw-stall-backend-ilock (supported on cpu 0-3)                # Backend stall cycles, input dependency
  raw-stall-backend-l1d (supported on cpu 0-7)          # Backend stall cycles, level 1 data cache
  raw-stall-backend-mem (supported on cpu 0-7)          # Memory stall cycles
  raw-stall-backend-membound (supported on cpu 0-7)             # Backend stall cycles, memory bound
  raw-stall-backend-rename (supported on cpu 4-7)               # Backend stall cycles, rename full
  raw-stall-backend-st (supported on cpu 0-7)           # Backend stall cycles, store
  raw-stall-backend-tlb (supported on cpu 0-7)          # Backend stall cycles, TLB
  raw-stall-frontend (supported on cpu 0-7)             # No operation sent for execution due to the frontend
  raw-stall-frontend-cpubound (supported on cpu 0-7)            # Frontend stall cycles, processor bound
  raw-stall-frontend-flow (supported on cpu 0-3)                # Frontend stall cycles, flow control
  raw-stall-frontend-flush (supported on cpu 0-7)               # Frontend stall cycles, flush recovery
  raw-stall-frontend-l1i (supported on cpu 0-7)         # Frontend stall cycles, level 1 instruction cache
  raw-stall-frontend-mem (supported on cpu 0-7)         # Frontend stall cycles, last level PE cache or memory
  raw-stall-frontend-membound (supported on cpu 0-7)            # Frontend stall cycles, memory bound
  raw-stall-frontend-tlb (supported on cpu 0-7)         # Frontend stall cycles, TLB
  raw-stall-slot (supported on cpu 0-7)         # No operation sent for execution on a Slot
  raw-stall-slot-backend (supported on cpu 0-7)         # No operation sent for execution on a Slot due to the backend
  raw-stall-slot-frontend (supported on cpu 0-7)                # No operation sent for execution on a Slot due to the frontend
  raw-strex-fail-spec (supported on cpu 0-7)            # Exclusive operation Speculatively executed, Store-Exclusive fail
  raw-strex-spec (supported on cpu 0-7)         # Exclusive operation Speculatively executed, Store-Exclusive
  raw-sve-inst-retired (supported on cpu 0-3)           # Instruction architecturally executed, SVE
  raw-sve-inst-spec (supported on cpu 0-7)              # Operation speculatively executed, SVE, including load and store
  raw-sve-ldff-fault-spec (supported on cpu 4-7)                # Operation speculatively executed, SVE first-fault load which set FFR bit to 0b0
  raw-sve-ldff-spec (supported on cpu 4-7)              # Operation speculatively executed, SVE first-fault load
  raw-sve-pred-empty-spec (supported on cpu 4-7)                # Operation speculatively executed, SVE predicated with no active predicates
  raw-sve-pred-full-spec (supported on cpu 4-7)         # Operation speculatively executed, SVE predicated with all active predicates
  raw-sve-pred-not-full-spec (supported on cpu 4-7)             # SVE predicated operations Speculatively executed with no active or partially active predicates
  raw-sve-pred-partial-spec (supported on cpu 4-7)              # Operation speculatively executed, SVE predicated with partially active predicates
  raw-sve-pred-spec (supported on cpu 4-7)              # Operation speculatively executed, SVE predicated
  raw-sw-incr (supported on cpu 0-7)            # Instruction architecturally executed, Condition code check pass, software increment
  raw-trb-trig (supported on cpu 0-3)           # Trace buffer Trigger Event
  raw-trb-wrap (supported on cpu 0-7)           # Trace buffer current write pointer wrapped
  raw-trcextout0 (supported on cpu 0-7)         # Trace unit external output 0
  raw-trcextout1 (supported on cpu 0-7)         # Trace unit external output 1
  raw-trcextout2 (supported on cpu 0-7)         # Trace unit external output 2
  raw-trcextout3 (supported on cpu 0-7)         # Trace unit external output 3
  raw-ttbr-write-retired (supported on cpu 0-7)         # Instruction architecturally executed, Condition code check pass, write to TTBR
  raw-vfp-spec (supported on cpu 0-7)           # Operation speculatively executed, scalar floating-point

# List of software events:
  alignment-faults
  context-switches
  cpu-clock
  cpu-migrations
  emulation-faults
  major-faults
  minor-faults
  page-faults
  task-clock

# List of tracepoint events:
  alarmtimer:alarmtimer_cancel
  alarmtimer:alarmtimer_fired
  alarmtimer:alarmtimer_start
  alarmtimer:alarmtimer_suspend
  arm_smmu:iommu_pgtable_add
  arm_smmu:iommu_pgtable_remove
  arm_smmu:map_pages
  arm_smmu:map_sg
  arm_smmu:smmu_init
  arm_smmu:tlbi_end
  arm_smmu:tlbi_start
  arm_smmu:tlbsync_timeout
  arm_smmu:unmap_pages
  asoc:snd_soc_bias_level_done
  asoc:snd_soc_bias_level_start
  asoc:snd_soc_dapm_connected
  asoc:snd_soc_dapm_done
  asoc:snd_soc_dapm_path
  asoc:snd_soc_dapm_start
  asoc:snd_soc_dapm_walk_done
  asoc:snd_soc_dapm_widget_event_done
  asoc:snd_soc_dapm_widget_event_start
  asoc:snd_soc_dapm_widget_power
  asoc:snd_soc_jack_irq
  asoc:snd_soc_jack_notify
  asoc:snd_soc_jack_report
  avc:selinux_audited
  bam_dma:bam_dma_info
  binder:binder_alloc_lru_end
  binder:binder_alloc_lru_start
  binder:binder_alloc_page_end
  binder:binder_alloc_page_start
  binder:binder_command
  binder:binder_free_lru_end
  binder:binder_free_lru_start
  binder:binder_ioctl
  binder:binder_ioctl_done
  binder:binder_lock
  binder:binder_locked
  binder:binder_read_done
  binder:binder_return
  binder:binder_set_priority
  binder:binder_transaction
  binder:binder_transaction_alloc_buf
  binder:binder_transaction_buffer_release
  binder:binder_transaction_failed_buffer_release
  binder:binder_transaction_fd_recv
  binder:binder_transaction_fd_send
  binder:binder_transaction_node_to_ref
  binder:binder_transaction_received
  binder:binder_transaction_ref_to_node
  binder:binder_transaction_ref_to_ref
  binder:binder_transaction_update_buffer_release
  binder:binder_txn_latency_free
  binder:binder_unlock
  binder:binder_unmap_kernel_end
  binder:binder_unmap_kernel_start
  binder:binder_unmap_user_end
  binder:binder_unmap_user_start
  binder:binder_update_page_range
  binder:binder_wait_for_work
  binder:binder_write_done
  block:block_bio_backmerge
  block:block_bio_bounce
  block:block_bio_complete
  block:block_bio_frontmerge
  block:block_bio_queue
  block:block_bio_remap
  block:block_dirty_buffer
  block:block_getrq
  block:block_plug
  block:block_rq_complete
  block:block_rq_error
  block:block_rq_insert
  block:block_rq_issue
  block:block_rq_merge
  block:block_rq_remap
  block:block_rq_requeue
  block:block_split
  block:block_touch_buffer
  block:block_unplug
  bpf_test_run:bpf_test_finish
  bpf_trace:bpf_trace_printk
  bridge:br_fdb_add
  bridge:br_fdb_external_learn_add
  bridge:br_fdb_update
  bridge:fdb_delete
  camera:cam_apply_req
  camera:cam_buf_done
  camera:cam_cci_burst
  camera:cam_cdm_cb
  camera:cam_context_state
  camera:cam_delay_detect
  camera:cam_flush_req
  camera:cam_i2c_write_log_event
  camera:cam_icp_fw_dbg
  camera:cam_irq_activated
  camera:cam_irq_handled
  camera:cam_isp_activated_irq
  camera:cam_log_debug
  camera:cam_log_event
  camera:cam_notify_frame_skip
  camera:cam_req_mgr_add_req
  camera:cam_req_mgr_apply_request
  camera:cam_req_mgr_connect_device
  camera:cam_submit_to_hw
  camera:opcode_name
  camera:poll_i2c_compare
  cfg80211:cfg80211_assoc_comeback
  cfg80211:cfg80211_bss_color_notify
  cfg80211:cfg80211_cac_event
  cfg80211:cfg80211_ch_switch_notify
  cfg80211:cfg80211_ch_switch_started_notify
  cfg80211:cfg80211_chandef_dfs_required
  cfg80211:cfg80211_control_port_tx_status
  cfg80211:cfg80211_cqm_pktloss_notify
  cfg80211:cfg80211_cqm_rssi_notify
  cfg80211:cfg80211_del_sta
  cfg80211:cfg80211_ft_event
  cfg80211:cfg80211_get_bss
  cfg80211:cfg80211_gtk_rekey_notify
  cfg80211:cfg80211_ibss_joined
  cfg80211:cfg80211_inform_bss_frame
  cfg80211:cfg80211_mgmt_tx_status
  cfg80211:cfg80211_michael_mic_failure
  cfg80211:cfg80211_new_sta
  cfg80211:cfg80211_notify_new_peer_candidate
  cfg80211:cfg80211_pmksa_candidate_notify
  cfg80211:cfg80211_pmsr_complete
  cfg80211:cfg80211_pmsr_report
  cfg80211:cfg80211_probe_status
  cfg80211:cfg80211_radar_event
  cfg80211:cfg80211_ready_on_channel
  cfg80211:cfg80211_ready_on_channel_expired
  cfg80211:cfg80211_reg_can_beacon
  cfg80211:cfg80211_report_obss_beacon
  cfg80211:cfg80211_report_wowlan_wakeup
  cfg80211:cfg80211_return_bool
  cfg80211:cfg80211_return_bss
  cfg80211:cfg80211_return_u32
  cfg80211:cfg80211_return_uint
  cfg80211:cfg80211_rx_control_port
  cfg80211:cfg80211_rx_mgmt
  cfg80211:cfg80211_rx_mlme_mgmt
  cfg80211:cfg80211_rx_spurious_frame
  cfg80211:cfg80211_rx_unexpected_4addr_frame
  cfg80211:cfg80211_rx_unprot_mlme_mgmt
  cfg80211:cfg80211_scan_done
  cfg80211:cfg80211_sched_scan_results
  cfg80211:cfg80211_sched_scan_stopped
  cfg80211:cfg80211_send_assoc_failure
  cfg80211:cfg80211_send_auth_timeout
  cfg80211:cfg80211_send_rx_assoc
  cfg80211:cfg80211_send_rx_auth
  cfg80211:cfg80211_stop_iface
  cfg80211:cfg80211_tdls_oper_request
  cfg80211:cfg80211_tx_mgmt_expired
  cfg80211:cfg80211_tx_mlme_mgmt
  cfg80211:cfg80211_update_owe_info_event
  cfg80211:rdev_abort_pmsr
  cfg80211:rdev_abort_scan
  cfg80211:rdev_add_intf_link
  cfg80211:rdev_add_key
  cfg80211:rdev_add_link_station
  cfg80211:rdev_add_mpath
  cfg80211:rdev_add_nan_func
  cfg80211:rdev_add_station
  cfg80211:rdev_add_tx_ts
  cfg80211:rdev_add_virtual_intf
  cfg80211:rdev_assoc
  cfg80211:rdev_auth
  cfg80211:rdev_cancel_remain_on_channel
  cfg80211:rdev_change_beacon
  cfg80211:rdev_change_bss
  cfg80211:rdev_change_mpath
  cfg80211:rdev_change_station
  cfg80211:rdev_change_virtual_intf
  cfg80211:rdev_channel_switch
  cfg80211:rdev_color_change
  cfg80211:rdev_connect
  cfg80211:rdev_crit_proto_start
  cfg80211:rdev_crit_proto_stop
  cfg80211:rdev_deauth
  cfg80211:rdev_del_intf_link
  cfg80211:rdev_del_key
  cfg80211:rdev_del_link_station
  cfg80211:rdev_del_mpath
  cfg80211:rdev_del_nan_func
  cfg80211:rdev_del_pmk
  cfg80211:rdev_del_pmksa
  cfg80211:rdev_del_station
  cfg80211:rdev_del_tx_ts
  cfg80211:rdev_del_virtual_intf
  cfg80211:rdev_disassoc
  cfg80211:rdev_disconnect
  cfg80211:rdev_dump_mpath
  cfg80211:rdev_dump_mpp
  cfg80211:rdev_dump_station
  cfg80211:rdev_dump_survey
  cfg80211:rdev_end_cac
  cfg80211:rdev_external_auth
  cfg80211:rdev_flush_pmksa
  cfg80211:rdev_get_antenna
  cfg80211:rdev_get_channel
  cfg80211:rdev_get_ftm_responder_stats
  cfg80211:rdev_get_key
  cfg80211:rdev_get_mesh_config
  cfg80211:rdev_get_mpath
  cfg80211:rdev_get_mpp
  cfg80211:rdev_get_station
  cfg80211:rdev_get_tx_power
  cfg80211:rdev_get_txq_stats
  cfg80211:rdev_join_ibss
  cfg80211:rdev_join_mesh
  cfg80211:rdev_join_ocb
  cfg80211:rdev_leave_ibss
  cfg80211:rdev_leave_mesh
  cfg80211:rdev_leave_ocb
  cfg80211:rdev_libertas_set_mesh_channel
  cfg80211:rdev_mgmt_tx
  cfg80211:rdev_mgmt_tx_cancel_wait
  cfg80211:rdev_mod_link_station
  cfg80211:rdev_nan_change_conf
  cfg80211:rdev_probe_client
  cfg80211:rdev_probe_mesh_link
  cfg80211:rdev_remain_on_channel
  cfg80211:rdev_reset_tid_config
  cfg80211:rdev_resume
  cfg80211:rdev_return_chandef
  cfg80211:rdev_return_int
  cfg80211:rdev_return_int_cookie
  cfg80211:rdev_return_int_int
  cfg80211:rdev_return_int_mesh_config
  cfg80211:rdev_return_int_mpath_info
  cfg80211:rdev_return_int_station_info
  cfg80211:rdev_return_int_survey_info
  cfg80211:rdev_return_int_tx_rx
  cfg80211:rdev_return_void
  cfg80211:rdev_return_void_tx_rx
  cfg80211:rdev_return_wdev
  cfg80211:rdev_rfkill_poll
  cfg80211:rdev_scan
  cfg80211:rdev_sched_scan_start
  cfg80211:rdev_sched_scan_stop
  cfg80211:rdev_set_antenna
  cfg80211:rdev_set_ap_chanwidth
  cfg80211:rdev_set_bitrate_mask
  cfg80211:rdev_set_coalesce
  cfg80211:rdev_set_cqm_rssi_config
  cfg80211:rdev_set_cqm_rssi_range_config
  cfg80211:rdev_set_cqm_txe_config
  cfg80211:rdev_set_default_beacon_key
  cfg80211:rdev_set_default_key
  cfg80211:rdev_set_default_mgmt_key
  cfg80211:rdev_set_fils_aad
  cfg80211:rdev_set_mac_acl
  cfg80211:rdev_set_mcast_rate
  cfg80211:rdev_set_monitor_channel
  cfg80211:rdev_set_multicast_to_unicast
  cfg80211:rdev_set_noack_map
  cfg80211:rdev_set_pmk
  cfg80211:rdev_set_pmksa
  cfg80211:rdev_set_power_mgmt
  cfg80211:rdev_set_qos_map
  cfg80211:rdev_set_radar_background
  cfg80211:rdev_set_rekey_data
  cfg80211:rdev_set_sar_specs
  cfg80211:rdev_set_tid_config
  cfg80211:rdev_set_tx_power
  cfg80211:rdev_set_txq_params
  cfg80211:rdev_set_wakeup
  cfg80211:rdev_set_wiphy_params
  cfg80211:rdev_start_ap
  cfg80211:rdev_start_nan
  cfg80211:rdev_start_p2p_device
  cfg80211:rdev_start_pmsr
  cfg80211:rdev_start_radar_detection
  cfg80211:rdev_stop_ap
  cfg80211:rdev_stop_nan
  cfg80211:rdev_stop_p2p_device
  cfg80211:rdev_suspend
  cfg80211:rdev_tdls_cancel_channel_switch
  cfg80211:rdev_tdls_channel_switch
  cfg80211:rdev_tdls_mgmt
  cfg80211:rdev_tdls_oper
  cfg80211:rdev_testmode_cmd
  cfg80211:rdev_testmode_dump
  cfg80211:rdev_tx_control_port
  cfg80211:rdev_update_connect_params
  cfg80211:rdev_update_ft_ies
  cfg80211:rdev_update_mesh_config
  cfg80211:rdev_update_mgmt_frame_registrations
  cfg80211:rdev_update_owe_info
  cfg802154:802154_rdev_add_virtual_intf
  cfg802154:802154_rdev_del_virtual_intf
  cfg802154:802154_rdev_resume
  cfg802154:802154_rdev_return_int
  cfg802154:802154_rdev_set_ackreq_default
  cfg802154:802154_rdev_set_backoff_exponent
  cfg802154:802154_rdev_set_cca_ed_level
  cfg802154:802154_rdev_set_cca_mode
  cfg802154:802154_rdev_set_channel
  cfg802154:802154_rdev_set_csma_backoffs
  cfg802154:802154_rdev_set_lbt_mode
  cfg802154:802154_rdev_set_max_frame_retries
  cfg802154:802154_rdev_set_pan_id
  cfg802154:802154_rdev_set_short_addr
  cfg802154:802154_rdev_set_tx_power
  cfg802154:802154_rdev_suspend
  cgroup:cgroup_attach_task
  cgroup:cgroup_destroy_root
  cgroup:cgroup_freeze
  cgroup:cgroup_mkdir
  cgroup:cgroup_notify_frozen
  cgroup:cgroup_notify_populated
  cgroup:cgroup_release
  cgroup:cgroup_remount
  cgroup:cgroup_rename
  cgroup:cgroup_rmdir
  cgroup:cgroup_setup_root
  cgroup:cgroup_transfer_tasks
  cgroup:cgroup_unfreeze
  clk:clk_disable
  clk:clk_disable_complete
  clk:clk_enable
  clk:clk_enable_complete
  clk:clk_prepare
  clk:clk_prepare_complete
  clk:clk_set_duty_cycle
  clk:clk_set_duty_cycle_complete
  clk:clk_set_max_rate
  clk:clk_set_min_rate
  clk:clk_set_parent
  clk:clk_set_parent_complete
  clk:clk_set_phase
  clk:clk_set_phase_complete
  clk:clk_set_rate
  clk:clk_set_rate_complete
  clk:clk_set_rate_range
  clk:clk_unprepare
  clk:clk_unprepare_complete
  clk_gdsc:gdsc_time
  clk_qcom:clk_measure
  clk_qcom:clk_state
  cluster_lpm:cluster_enter
  cluster_lpm:cluster_exit
  cluster_lpm:cluster_pred_hist
  cluster_lpm:cluster_pred_select
  cma:cma_alloc_busy_retry
  cma:cma_alloc_finish
  cma:cma_alloc_start
  cma:cma_release
  compaction:mm_compaction_begin
  compaction:mm_compaction_defer_compaction
  compaction:mm_compaction_defer_reset
  compaction:mm_compaction_deferred
  compaction:mm_compaction_end
  compaction:mm_compaction_finished
  compaction:mm_compaction_isolate_freepages
  compaction:mm_compaction_isolate_migratepages
  compaction:mm_compaction_kcompactd_sleep
  compaction:mm_compaction_kcompactd_wake
  compaction:mm_compaction_migratepages
  compaction:mm_compaction_suitable
  compaction:mm_compaction_try_to_compact_pages
  compaction:mm_compaction_wakeup_kcompactd
  cpucp:cpucp_log
  cpuhp:cpuhp_enter
  cpuhp:cpuhp_exit
  cpuhp:cpuhp_multi_enter
  crm:crm_cache_vcd_votes
  crm:crm_irq
  crm:crm_switch_channel
  crm:crm_write_vcd_votes
  damon:damon_aggregated
  dcvs:bw_hwmon_debug
  dcvs:bw_hwmon_meas
  dcvs:bw_hwmon_update
  dcvs:bwprof_last_sample
  dcvs:memlat_dev_meas
  dcvs:memlat_dev_update
  dcvs:qcom_dcvs_boost
  dcvs:qcom_dcvs_update
  dcvsh:dcvsh_freq
  dcvsh:dcvsh_throttle
  dev:devres_log
  devfreq:devfreq_frequency
  devfreq:devfreq_monitor
  devlink:devlink_health_recover_aborted
  devlink:devlink_health_report
  devlink:devlink_health_reporter_state_update
  devlink:devlink_hwerr
  devlink:devlink_hwmsg
  devlink:devlink_trap_report
  dfc:dfc_adjust_grant
  dfc:dfc_client_state_down
  dfc:dfc_client_state_up
  dfc:dfc_flow_check
  dfc:dfc_flow_ind
  dfc:dfc_flow_info
  dfc:dfc_ll_switch
  dfc:dfc_qmap
  dfc:dfc_qmap_cmd
  dfc:dfc_qmi_tc
  dfc:dfc_set_powersave_mode
  dfc:dfc_tx_link_status_ind
  dfc:dfc_watchdog
  dma_fence:dma_fence_destroy
  dma_fence:dma_fence_emit
  dma_fence:dma_fence_enable_signal
  dma_fence:dma_fence_init
  dma_fence:dma_fence_signaled
  dma_fence:dma_fence_wait_end
  dma_fence:dma_fence_wait_start
  drm:drm_vblank_event
  drm:drm_vblank_event_delivered
  drm:drm_vblank_event_queued
  dwc3:dwc3_alloc_request
  dwc3:dwc3_complete_trb
  dwc3:dwc3_ctrl_req
  dwc3:dwc3_ep_dequeue
  dwc3:dwc3_ep_queue
  dwc3:dwc3_event
  dwc3:dwc3_free_request
  dwc3:dwc3_gadget_ep_cmd
  dwc3:dwc3_gadget_ep_disable
  dwc3:dwc3_gadget_ep_enable
  dwc3:dwc3_gadget_generic_cmd
  dwc3:dwc3_gadget_giveback
  dwc3:dwc3_prepare_trb
  dwc3:dwc3_readl
  dwc3:dwc3_writel
  emulation:instruction_emulation
  erofs:erofs_destroy_inode
  erofs:erofs_fill_inode
  erofs:erofs_lookup
  erofs:erofs_map_blocks_enter
  erofs:erofs_map_blocks_exit
  erofs:erofs_readpage
  erofs:erofs_readpages
  erofs:z_erofs_map_blocks_iter_enter
  erofs:z_erofs_map_blocks_iter_exit
  error_report:error_report_end
  ext4:ext4_alloc_da_blocks
  ext4:ext4_allocate_blocks
  ext4:ext4_allocate_inode
  ext4:ext4_begin_ordered_truncate
  ext4:ext4_collapse_range
  ext4:ext4_da_release_space
  ext4:ext4_da_reserve_space
  ext4:ext4_da_update_reserve_space
  ext4:ext4_da_write_begin
  ext4:ext4_da_write_end
  ext4:ext4_da_write_pages
  ext4:ext4_da_write_pages_extent
  ext4:ext4_discard_blocks
  ext4:ext4_discard_preallocations
  ext4:ext4_drop_inode
  ext4:ext4_error
  ext4:ext4_es_cache_extent
  ext4:ext4_es_find_extent_range_enter
  ext4:ext4_es_find_extent_range_exit
  ext4:ext4_es_insert_delayed_block
  ext4:ext4_es_insert_extent
  ext4:ext4_es_lookup_extent_enter
  ext4:ext4_es_lookup_extent_exit
  ext4:ext4_es_remove_extent
  ext4:ext4_es_shrink
  ext4:ext4_es_shrink_count
  ext4:ext4_es_shrink_scan_enter
  ext4:ext4_es_shrink_scan_exit
  ext4:ext4_evict_inode
  ext4:ext4_ext_convert_to_initialized_enter
  ext4:ext4_ext_convert_to_initialized_fastpath
  ext4:ext4_ext_handle_unwritten_extents
  ext4:ext4_ext_load_extent
  ext4:ext4_ext_map_blocks_enter
  ext4:ext4_ext_map_blocks_exit
  ext4:ext4_ext_remove_space
  ext4:ext4_ext_remove_space_done
  ext4:ext4_ext_rm_idx
  ext4:ext4_ext_rm_leaf
  ext4:ext4_ext_show_extent
  ext4:ext4_fallocate_enter
  ext4:ext4_fallocate_exit
  ext4:ext4_fc_cleanup
  ext4:ext4_fc_commit_start
  ext4:ext4_fc_commit_stop
  ext4:ext4_fc_replay
  ext4:ext4_fc_replay_scan
  ext4:ext4_fc_stats
  ext4:ext4_fc_track_create
  ext4:ext4_fc_track_inode
  ext4:ext4_fc_track_link
  ext4:ext4_fc_track_range
  ext4:ext4_fc_track_unlink
  ext4:ext4_forget
  ext4:ext4_free_blocks
  ext4:ext4_free_inode
  ext4:ext4_fsmap_high_key
  ext4:ext4_fsmap_low_key
  ext4:ext4_fsmap_mapping
  ext4:ext4_get_implied_cluster_alloc_exit
  ext4:ext4_getfsmap_high_key
  ext4:ext4_getfsmap_low_key
  ext4:ext4_getfsmap_mapping
  ext4:ext4_ind_map_blocks_enter
  ext4:ext4_ind_map_blocks_exit
  ext4:ext4_insert_range
  ext4:ext4_invalidate_folio
  ext4:ext4_journal_start
  ext4:ext4_journal_start_reserved
  ext4:ext4_journalled_invalidate_folio
  ext4:ext4_journalled_write_end
  ext4:ext4_lazy_itable_init
  ext4:ext4_load_inode
  ext4:ext4_load_inode_bitmap
  ext4:ext4_mark_inode_dirty
  ext4:ext4_mb_bitmap_load
  ext4:ext4_mb_buddy_bitmap_load
  ext4:ext4_mb_discard_preallocations
  ext4:ext4_mb_new_group_pa
  ext4:ext4_mb_new_inode_pa
  ext4:ext4_mb_release_group_pa
  ext4:ext4_mb_release_inode_pa
  ext4:ext4_mballoc_alloc
  ext4:ext4_mballoc_discard
  ext4:ext4_mballoc_free
  ext4:ext4_mballoc_prealloc
  ext4:ext4_nfs_commit_metadata
  ext4:ext4_other_inode_update_time
  ext4:ext4_prefetch_bitmaps
  ext4:ext4_punch_hole
  ext4:ext4_read_block_bitmap_load
  ext4:ext4_readpage
  ext4:ext4_releasepage
  ext4:ext4_remove_blocks
  ext4:ext4_request_blocks
  ext4:ext4_request_inode
  ext4:ext4_shutdown
  ext4:ext4_sync_file_enter
  ext4:ext4_sync_file_exit
  ext4:ext4_sync_fs
  ext4:ext4_trim_all_free
  ext4:ext4_trim_extent
  ext4:ext4_truncate_enter
  ext4:ext4_truncate_exit
  ext4:ext4_unlink_enter
  ext4:ext4_unlink_exit
  ext4:ext4_update_sb
  ext4:ext4_write_begin
  ext4:ext4_write_end
  ext4:ext4_writepage
  ext4:ext4_writepages
  ext4:ext4_writepages_result
  ext4:ext4_zero_range
  f2fs:f2fs_background_gc
  f2fs:f2fs_bmap
  f2fs:f2fs_compress_pages_end
  f2fs:f2fs_compress_pages_start
  f2fs:f2fs_dataread_end
  f2fs:f2fs_dataread_start
  f2fs:f2fs_datawrite_end
  f2fs:f2fs_datawrite_start
  f2fs:f2fs_decompress_pages_end
  f2fs:f2fs_decompress_pages_start
  f2fs:f2fs_destroy_extent_tree
  f2fs:f2fs_direct_IO_enter
  f2fs:f2fs_direct_IO_exit
  f2fs:f2fs_do_write_data_page
  f2fs:f2fs_drop_inode
  f2fs:f2fs_evict_inode
  f2fs:f2fs_fallocate
  f2fs:f2fs_fiemap
  f2fs:f2fs_file_write_iter
  f2fs:f2fs_filemap_fault
  f2fs:f2fs_gc_begin
  f2fs:f2fs_gc_end
  f2fs:f2fs_get_victim
  f2fs:f2fs_iget
  f2fs:f2fs_iget_exit
  f2fs:f2fs_iostat
  f2fs:f2fs_iostat_latency
  f2fs:f2fs_issue_discard
  f2fs:f2fs_issue_flush
  f2fs:f2fs_issue_reset_zone
  f2fs:f2fs_lookup_age_extent_tree_end
  f2fs:f2fs_lookup_end
  f2fs:f2fs_lookup_extent_tree_start
  f2fs:f2fs_lookup_read_extent_tree_end
  f2fs:f2fs_lookup_start
  f2fs:f2fs_map_blocks
  f2fs:f2fs_new_inode
  f2fs:f2fs_prepare_read_bio
  f2fs:f2fs_prepare_write_bio
  f2fs:f2fs_queue_discard
  f2fs:f2fs_readdir
  f2fs:f2fs_readpage
  f2fs:f2fs_readpages
  f2fs:f2fs_remove_discard
  f2fs:f2fs_replace_atomic_write_block
  f2fs:f2fs_reserve_new_blocks
  f2fs:f2fs_set_page_dirty
  f2fs:f2fs_shrink_extent_tree
  f2fs:f2fs_shutdown
  f2fs:f2fs_submit_page_bio
  f2fs:f2fs_submit_page_write
  f2fs:f2fs_submit_read_bio
  f2fs:f2fs_submit_write_bio
  f2fs:f2fs_sync_dirty_inodes_enter
  f2fs:f2fs_sync_dirty_inodes_exit
  f2fs:f2fs_sync_file_enter
  f2fs:f2fs_sync_file_exit
  f2fs:f2fs_sync_fs
  f2fs:f2fs_truncate
  f2fs:f2fs_truncate_blocks_enter
  f2fs:f2fs_truncate_blocks_exit
  f2fs:f2fs_truncate_data_blocks_range
  f2fs:f2fs_truncate_inode_blocks_enter
  f2fs:f2fs_truncate_inode_blocks_exit
  f2fs:f2fs_truncate_node
  f2fs:f2fs_truncate_nodes_enter
  f2fs:f2fs_truncate_nodes_exit
  f2fs:f2fs_truncate_partial_nodes
  f2fs:f2fs_unlink_enter
  f2fs:f2fs_unlink_exit
  f2fs:f2fs_update_age_extent_tree_range
  f2fs:f2fs_update_read_extent_tree_range
  f2fs:f2fs_vm_page_mkwrite
  f2fs:f2fs_write_begin
  f2fs:f2fs_write_checkpoint
  f2fs:f2fs_write_end
  f2fs:f2fs_writepage
  f2fs:f2fs_writepages
  fastrpc:fastrpc_context_alloc
  fastrpc:fastrpc_context_complete
  fastrpc:fastrpc_context_free
  fastrpc:fastrpc_context_interrupt
  fastrpc:fastrpc_context_restore
  fastrpc:fastrpc_dma_alloc
  fastrpc:fastrpc_dma_free
  fastrpc:fastrpc_dma_map
  fastrpc:fastrpc_dma_unmap
  fastrpc:fastrpc_dspsignal
  fastrpc:fastrpc_msg
  fastrpc:fastrpc_perf_counters
  fastrpc:fastrpc_transport_response
  fastrpc:fastrpc_transport_send
  fib6:fib6_table_lookup
  fib:fib_table_lookup
  filelock:break_lease_block
  filelock:break_lease_noblock
  filelock:break_lease_unblock
  filelock:fcntl_setlk
  filelock:flock_lock_inode
  filelock:generic_add_lease
  filelock:generic_delete_lease
  filelock:leases_conflict
  filelock:locks_get_lock_context
  filelock:locks_remove_posix
  filelock:posix_lock_inode
  filelock:time_out_leases
  filemap:file_check_and_advance_wb_err
  filemap:filemap_set_wb_err
  filemap:mm_filemap_add_to_page_cache
  filemap:mm_filemap_delete_from_page_cache
  ftrace:print
  gadget:usb_ep_alloc_request
  gadget:usb_ep_clear_halt
  gadget:usb_ep_dequeue
  gadget:usb_ep_disable
  gadget:usb_ep_enable
  gadget:usb_ep_fifo_flush
  gadget:usb_ep_fifo_status
  gadget:usb_ep_free_request
  gadget:usb_ep_queue
  gadget:usb_ep_set_halt
  gadget:usb_ep_set_maxpacket_limit
  gadget:usb_ep_set_wedge
  gadget:usb_gadget_activate
  gadget:usb_gadget_clear_selfpowered
  gadget:usb_gadget_connect
  gadget:usb_gadget_deactivate
  gadget:usb_gadget_disconnect
  gadget:usb_gadget_frame_number
  gadget:usb_gadget_giveback_request
  gadget:usb_gadget_set_selfpowered
  gadget:usb_gadget_vbus_connect
  gadget:usb_gadget_vbus_disconnect
  gadget:usb_gadget_vbus_draw
  gadget:usb_gadget_wakeup
  gh_proxy_sched:gh_hcall_vcpu_run
  gh_proxy_sched:gh_susp_res_irq_handler
  gh_proxy_sched:gh_vcpu_irq_handler
  gh_virtio_backend:gh_virtio_backend_irq
  gh_virtio_backend:gh_virtio_backend_irq_inj
  gh_virtio_backend:gh_virtio_backend_queue_notify
  gh_virtio_backend:gh_virtio_backend_wait_event
  gpio:gpio_direction
  gpio:gpio_value
  gpu_mem:gpu_mem_total
  gsi:gsi_qtimer
  gunyah:gh_rm_mem_accept
  gunyah:gh_rm_mem_accept_reply
  gunyah:gh_rm_mem_call_return
  gunyah:gh_rm_mem_donate
  gunyah:gh_rm_mem_lend
  gunyah:gh_rm_mem_notify
  gunyah:gh_rm_mem_reclaim
  gunyah:gh_rm_mem_release
  gunyah:gh_rm_mem_share
  huge_memory:mm_collapse_huge_page
  huge_memory:mm_collapse_huge_page_isolate
  huge_memory:mm_collapse_huge_page_swapin
  huge_memory:mm_khugepaged_scan_file
  huge_memory:mm_khugepaged_scan_pmd
  hwmon:hwmon_attr_show
  hwmon:hwmon_attr_show_string
  hwmon:hwmon_attr_store
  i2c:i2c_read
  i2c:i2c_reply
  i2c:i2c_result
  i2c:i2c_write
  initcall:initcall_finish
  initcall:initcall_level
  initcall:initcall_start
  interconnect:icc_set_bw
  interconnect:icc_set_bw_end
  interconnect_qcom:bcm_voter_commit
  io_uring:io_uring_complete
  io_uring:io_uring_cqe_overflow
  io_uring:io_uring_cqring_wait
  io_uring:io_uring_create
  io_uring:io_uring_defer
  io_uring:io_uring_fail_link
  io_uring:io_uring_file_get
  io_uring:io_uring_link
  io_uring:io_uring_local_work_run
  io_uring:io_uring_poll_arm
  io_uring:io_uring_queue_async_work
  io_uring:io_uring_register
  io_uring:io_uring_req_failed
  io_uring:io_uring_short_write
  io_uring:io_uring_submit_sqe
  io_uring:io_uring_task_add
  io_uring:io_uring_task_work_run
  iocost:iocost_inuse_adjust
  iocost:iocost_inuse_shortage
  iocost:iocost_inuse_transfer
  iocost:iocost_ioc_vrate_adj
  iocost:iocost_iocg_activate
  iocost:iocost_iocg_forgive_debt
  iocost:iocost_iocg_idle
  iomap:iomap_dio_invalidate_fail
  iomap:iomap_invalidate_folio
  iomap:iomap_iter
  iomap:iomap_iter_dstmap
  iomap:iomap_iter_srcmap
  iomap:iomap_readahead
  iomap:iomap_readpage
  iomap:iomap_release_folio
  iomap:iomap_writepage
  iomap:iomap_writepage_map
  iommu:add_device_to_group
  iommu:attach_device_to_domain
  iommu:detach_device_from_domain
  iommu:io_page_fault
  iommu:map
  iommu:remove_device_from_group
  iommu:unmap
  ipa:handle_page_completion
  ipa:idle_sleep_enter3
  ipa:idle_sleep_exit3
  ipa:intr_to_poll3
  ipa:ipa3_napi_poll_entry
  ipa:ipa3_napi_poll_exit
  ipa:ipa3_napi_rx_poll_cnt
  ipa:ipa3_napi_rx_poll_num
  ipa:ipa3_napi_schedule
  ipa:ipa3_replenish_rx_page_recycle
  ipa:ipa3_rx_napi_chain
  ipa:ipa3_tx_done
  ipa:ipa_tx_dp
  ipa:poll_to_intr3
  ipa:rmnet_ipa_netif_rcv_skb3
  ipa:rmnet_ipa_netifni3
  ipa:rmnet_ipa_netifrx3
  ipi:ipi_entry
  ipi:ipi_exit
  ipi:ipi_raise
  irq:irq_handler_entry
  irq:irq_handler_exit
  irq:softirq_entry
  irq:softirq_exit
  irq:softirq_raise
  irq:tasklet_entry
  irq:tasklet_exit
  jbd2:jbd2_checkpoint
  jbd2:jbd2_checkpoint_stats
  jbd2:jbd2_commit_flushing
  jbd2:jbd2_commit_locking
  jbd2:jbd2_commit_logging
  jbd2:jbd2_drop_transaction
  jbd2:jbd2_end_commit
  jbd2:jbd2_handle_extend
  jbd2:jbd2_handle_restart
  jbd2:jbd2_handle_start
  jbd2:jbd2_handle_stats
  jbd2:jbd2_lock_buffer_stall
  jbd2:jbd2_run_stats
  jbd2:jbd2_shrink_checkpoint_list
  jbd2:jbd2_shrink_count
  jbd2:jbd2_shrink_scan_enter
  jbd2:jbd2_shrink_scan_exit
  jbd2:jbd2_start_commit
  jbd2:jbd2_submit_inode_data
  jbd2:jbd2_update_log_tail
  jbd2:jbd2_write_superblock
  kgsl:adreno_cmdbatch_done
  kgsl:adreno_cmdbatch_fault
  kgsl:adreno_cmdbatch_queued
  kgsl:adreno_cmdbatch_ready
  kgsl:adreno_cmdbatch_recovery
  kgsl:adreno_cmdbatch_retired
  kgsl:adreno_cmdbatch_submitted
  kgsl:adreno_cmdbatch_sync
  kgsl:adreno_drawctxt_invalidate
  kgsl:adreno_drawctxt_sleep
  kgsl:adreno_drawctxt_switch
  kgsl:adreno_drawctxt_wait_done
  kgsl:adreno_drawctxt_wait_start
  kgsl:adreno_drawctxt_wake
  kgsl:adreno_gpu_fault
  kgsl:adreno_hw_fence_query
  kgsl:adreno_hw_preempt_clear_to_trig
  kgsl:adreno_hw_preempt_comp_to_clear
  kgsl:adreno_hw_preempt_token_submit
  kgsl:adreno_hw_preempt_trig_to_comp
  kgsl:adreno_hw_preempt_trig_to_comp_int
  kgsl:adreno_ifpc_count
  kgsl:adreno_input_hw_fence
  kgsl:adreno_preempt_done
  kgsl:adreno_preempt_trigger
  kgsl:adreno_sp_tp
  kgsl:adreno_syncobj_query
  kgsl:adreno_syncobj_query_reply
  kgsl:adreno_syncobj_retired
  kgsl:adreno_syncobj_submitted
  kgsl:dispatch_queue_context
  kgsl:gmu_ao_sync
  kgsl:gmu_event
  kgsl:gpu_frequency
  kgsl:kgsl_a3xx_irq_status
  kgsl:kgsl_a5xx_irq_status
  kgsl:kgsl_active_count
  kgsl:kgsl_aux_command
  kgsl:kgsl_bcl_clock_throttling
  kgsl:kgsl_bus
  kgsl:kgsl_buslevel
  kgsl:kgsl_clk
  kgsl:kgsl_clock_throttling
  kgsl:kgsl_constraint
  kgsl:kgsl_context_create
  kgsl:kgsl_context_destroy
  kgsl:kgsl_context_detach
  kgsl:kgsl_drawobj_timeline
  kgsl:kgsl_fire_event
  kgsl:kgsl_gen7_irq_status
  kgsl:kgsl_gen8_irq_status
  kgsl:kgsl_gmu_oob_clear
  kgsl:kgsl_gmu_oob_set
  kgsl:kgsl_gmu_pwrlevel
  kgsl:kgsl_gpubusy
  kgsl:kgsl_hfi_receive
  kgsl:kgsl_hfi_send
  kgsl:kgsl_irq
  kgsl:kgsl_issueibcmds
  kgsl:kgsl_mem_add_bind_range
  kgsl:kgsl_mem_alloc
  kgsl:kgsl_mem_free
  kgsl:kgsl_mem_map
  kgsl:kgsl_mem_mmap
  kgsl:kgsl_mem_remove_bind_range
  kgsl:kgsl_mem_sync_cache
  kgsl:kgsl_mem_sync_full_cache
  kgsl:kgsl_mem_timestamp_free
  kgsl:kgsl_mem_timestamp_queue
  kgsl:kgsl_mem_unmapped_area_collision
  kgsl:kgsl_mmu_pagefault
  kgsl:kgsl_msg
  kgsl:kgsl_pagetable_destroy
  kgsl:kgsl_pool_add_page
  kgsl:kgsl_pool_alloc_page_system
  kgsl:kgsl_pool_free_page
  kgsl:kgsl_pool_get_page
  kgsl:kgsl_pool_try_page_lower
  kgsl:kgsl_pwr_request_state
  kgsl:kgsl_pwr_set_state
  kgsl:kgsl_pwrlevel
  kgsl:kgsl_pwrstats
  kgsl:kgsl_rail
  kgsl:kgsl_readtimestamp
  kgsl:kgsl_reclaim_memdesc
  kgsl:kgsl_reclaim_process
  kgsl:kgsl_register_event
  kgsl:kgsl_regwrite
  kgsl:kgsl_thermal_constraint
  kgsl:kgsl_timeline_alloc
  kgsl:kgsl_timeline_destroy
  kgsl:kgsl_timeline_fence_alloc
  kgsl:kgsl_timeline_fence_release
  kgsl:kgsl_timeline_signal
  kgsl:kgsl_timeline_wait
  kgsl:kgsl_user_pwrlevel_constraint
  kgsl:kgsl_waittimestamp_entry
  kgsl:kgsl_waittimestamp_exit
  kgsl:perfmgr_gpu_status
  kgsl:syncpoint_fence
  kgsl:syncpoint_fence_expire
  kgsl:syncpoint_timestamp
  kgsl:syncpoint_timestamp_expire
  kmem:kfree
  kmem:kmalloc
  kmem:kmem_cache_alloc
  kmem:kmem_cache_free
  kmem:mm_alloc_contig_migrate_range_info
  kmem:mm_page_alloc
  kmem:mm_page_alloc_extfrag
  kmem:mm_page_alloc_zone_locked
  kmem:mm_page_free
  kmem:mm_page_free_batched
  kmem:mm_page_pcpu_drain
  kmem:rss_stat
  kprobes:dma_buf_file_release_miuibpf_bcc_2126
  kprobes:dma_buf_stats_setup_miuibpf_bcc_2126
  kprobes:enter_fd_install_miuibpf_bcc_2126
  kvm:kvm_access_fault
  kvm:kvm_ack_irq
  kvm:kvm_age_hva
  kvm:kvm_arm_clear_debug
  kvm:kvm_arm_set_dreg32
  kvm:kvm_arm_set_regset
  kvm:kvm_arm_setup_debug
  kvm:kvm_dirty_ring_exit
  kvm:kvm_dirty_ring_push
  kvm:kvm_dirty_ring_reset
  kvm:kvm_entry
  kvm:kvm_exit
  kvm:kvm_fpu
  kvm:kvm_get_timer_map
  kvm:kvm_guest_fault
  kvm:kvm_halt_poll_ns
  kvm:kvm_handle_sys_reg
  kvm:kvm_hvc_arm64
  kvm:kvm_irq_line
  kvm:kvm_mmio
  kvm:kvm_mmio_emulate
  kvm:kvm_set_guest_debug
  kvm:kvm_set_irq
  kvm:kvm_set_spte_hva
  kvm:kvm_set_way_flush
  kvm:kvm_sys_access
  kvm:kvm_test_age_hva
  kvm:kvm_timer_emulate
  kvm:kvm_timer_hrtimer_expire
  kvm:kvm_timer_restore_state
  kvm:kvm_timer_save_state
  kvm:kvm_timer_update_irq
  kvm:kvm_toggle_cache
  kvm:kvm_unmap_hva_range
  kvm:kvm_userspace_exit
  kvm:kvm_vcpu_wakeup
  kvm:kvm_wfx_arm64
  kvm:trap_reg
  kvm:vgic_update_irq_pending
  kyber:kyber_adjust
  kyber:kyber_latency
  kyber:kyber_throttled
  l2tp:delete_session
  l2tp:delete_tunnel
  l2tp:free_session
  l2tp:free_tunnel
  l2tp:register_session
  l2tp:register_tunnel
  l2tp:session_pkt_expired
  l2tp:session_pkt_oos
  l2tp:session_pkt_outside_rx_window
  l2tp:session_seqnum_lns_disable
  l2tp:session_seqnum_lns_enable
  l2tp:session_seqnum_reset
  l2tp:session_seqnum_update
  lock:contention_begin
  lock:contention_end
  lock:lock_acquire
  lock:lock_acquired
  lock:lock_contended
  lock:lock_release
  mac80211:api_beacon_loss
  mac80211:api_chswitch_done
  mac80211:api_connection_loss
  mac80211:api_cqm_beacon_loss_notify
  mac80211:api_cqm_rssi_notify
  mac80211:api_disconnect
  mac80211:api_enable_rssi_reports
  mac80211:api_eosp
  mac80211:api_gtk_rekey_notify
  mac80211:api_radar_detected
  mac80211:api_ready_on_channel
  mac80211:api_remain_on_channel_expired
  mac80211:api_restart_hw
  mac80211:api_scan_completed
  mac80211:api_sched_scan_results
  mac80211:api_sched_scan_stopped
  mac80211:api_send_eosp_nullfunc
  mac80211:api_sta_block_awake
  mac80211:api_sta_set_buffered
  mac80211:api_start_tx_ba_cb
  mac80211:api_start_tx_ba_session
  mac80211:api_stop_tx_ba_cb
  mac80211:api_stop_tx_ba_session
  mac80211:drv_abort_channel_switch
  mac80211:drv_abort_pmsr
  mac80211:drv_add_chanctx
  mac80211:drv_add_interface
  mac80211:drv_add_nan_func
  mac80211:drv_add_twt_setup
  mac80211:drv_allow_buffered_frames
  mac80211:drv_ampdu_action
  mac80211:drv_assign_vif_chanctx
  mac80211:drv_cancel_hw_scan
  mac80211:drv_cancel_remain_on_channel
  mac80211:drv_change_chanctx
  mac80211:drv_change_interface
  mac80211:drv_change_sta_links
  mac80211:drv_change_vif_links
  mac80211:drv_channel_switch
  mac80211:drv_channel_switch_beacon
  mac80211:drv_channel_switch_rx_beacon
  mac80211:drv_conf_tx
  mac80211:drv_config
  mac80211:drv_config_iface_filter
  mac80211:drv_configure_filter
  mac80211:drv_del_nan_func
  mac80211:drv_event_callback
  mac80211:drv_flush
  mac80211:drv_get_antenna
  mac80211:drv_get_et_sset_count
  mac80211:drv_get_et_stats
  mac80211:drv_get_et_strings
  mac80211:drv_get_expected_throughput
  mac80211:drv_get_ftm_responder_stats
  mac80211:drv_get_key_seq
  mac80211:drv_get_ringparam
  mac80211:drv_get_stats
  mac80211:drv_get_survey
  mac80211:drv_get_tsf
  mac80211:drv_get_txpower
  mac80211:drv_hw_scan
  mac80211:drv_ipv6_addr_change
  mac80211:drv_join_ibss
  mac80211:drv_leave_ibss
  mac80211:drv_link_info_changed
  mac80211:drv_mgd_complete_tx
  mac80211:drv_mgd_prepare_tx
  mac80211:drv_mgd_protect_tdls_discover
  mac80211:drv_nan_change_conf
  mac80211:drv_net_fill_forward_path
  mac80211:drv_offchannel_tx_cancel_wait
  mac80211:drv_offset_tsf
  mac80211:drv_post_channel_switch
  mac80211:drv_pre_channel_switch
  mac80211:drv_prepare_multicast
  mac80211:drv_reconfig_complete
  mac80211:drv_release_buffered_frames
  mac80211:drv_remain_on_channel
  mac80211:drv_remove_chanctx
  mac80211:drv_remove_interface
  mac80211:drv_reset_tsf
  mac80211:drv_resume
  mac80211:drv_return_bool
  mac80211:drv_return_int
  mac80211:drv_return_u32
  mac80211:drv_return_u64
  mac80211:drv_return_void
  mac80211:drv_sched_scan_start
  mac80211:drv_sched_scan_stop
  mac80211:drv_set_antenna
  mac80211:drv_set_bitrate_mask
  mac80211:drv_set_coverage_class
  mac80211:drv_set_default_unicast_key
  mac80211:drv_set_frag_threshold
  mac80211:drv_set_key
  mac80211:drv_set_rekey_data
  mac80211:drv_set_ringparam
  mac80211:drv_set_rts_threshold
  mac80211:drv_set_tim
  mac80211:drv_set_tsf
  mac80211:drv_set_wakeup
  mac80211:drv_sta_add
  mac80211:drv_sta_notify
  mac80211:drv_sta_pre_rcu_remove
  mac80211:drv_sta_rate_tbl_update
  mac80211:drv_sta_rc_update
  mac80211:drv_sta_remove
  mac80211:drv_sta_set_4addr
  mac80211:drv_sta_set_decap_offload
  mac80211:drv_sta_set_txpwr
  mac80211:drv_sta_state
  mac80211:drv_sta_statistics
  mac80211:drv_start
  mac80211:drv_start_ap
  mac80211:drv_start_nan
  mac80211:drv_start_pmsr
  mac80211:drv_stop
  mac80211:drv_stop_ap
  mac80211:drv_stop_nan
  mac80211:drv_suspend
  mac80211:drv_sw_scan_complete
  mac80211:drv_sw_scan_start
  mac80211:drv_switch_vif_chanctx
  mac80211:drv_sync_rx_queues
  mac80211:drv_tdls_cancel_channel_switch
  mac80211:drv_tdls_channel_switch
  mac80211:drv_tdls_recv_channel_switch
  mac80211:drv_twt_teardown_request
  mac80211:drv_tx_frames_pending
  mac80211:drv_tx_last_beacon
  mac80211:drv_unassign_vif_chanctx
  mac80211:drv_update_tkip_key
  mac80211:drv_update_vif_offload
  mac80211:drv_vif_cfg_changed
  mac80211:drv_wake_tx_queue
  mac80211:stop_queue
  mac80211:wake_queue
  mac802154:802154_drv_return_int
  mac802154:802154_drv_return_void
  mac802154:802154_drv_set_cca_ed_level
  mac802154:802154_drv_set_cca_mode
  mac802154:802154_drv_set_channel
  mac802154:802154_drv_set_csma_params
  mac802154:802154_drv_set_extended_addr
  mac802154:802154_drv_set_lbt_mode
  mac802154:802154_drv_set_max_frame_retries
  mac802154:802154_drv_set_pan_coord
  mac802154:802154_drv_set_pan_id
  mac802154:802154_drv_set_promiscuous_mode
  mac802154:802154_drv_set_short_addr
  mac802154:802154_drv_set_tx_power
  mac802154:802154_drv_start
  mac802154:802154_drv_stop
  maple_tree:ma_op
  maple_tree:ma_read
  maple_tree:ma_write
  mdio:mdio_access
  mem_buf:lookup_sgl
  mem_buf:map_mem_s2
  mem_buf:mem_buf_alloc_info
  mem_buf:receive_alloc_req
  mem_buf:receive_alloc_resp_msg
  mem_buf:receive_relinquish_msg
  mem_buf:receive_relinquish_resp_msg
  mem_buf:send_alloc_req
  mem_buf:send_alloc_resp_msg
  mem_buf:send_relinquish_msg
  mem_buf:send_relinquish_resp_msg
  metis:lock_spin_dur
  metis:metis_log
  metis:mi_rwsem_blocked_info
  metis:tracing_mark_write
  migrate:mm_migrate_pages
  migrate:mm_migrate_pages_start
  migrate:remove_migration_pte
  migrate:set_migration_pte
  mmap:exit_mmap
  mmap:vm_unmapped_area
  mmap:vma_mas_szero
  mmap:vma_store
  mmap_lock:mmap_lock_acquire_returned
  mmap_lock:mmap_lock_released
  mmap_lock:mmap_lock_start_locking
  mmc:mmc_request_done
  mmc:mmc_request_start
  module:module_free
  module:module_get
  module:module_load
  module:module_put
  module:module_request
  msm_vidc_events:msm_v4l2_vidc_buffer_event_log
  msm_vidc_events:msm_v4l2_vidc_close
  msm_vidc_events:msm_v4l2_vidc_fw_load
  msm_vidc_events:msm_v4l2_vidc_open
  msm_vidc_events:msm_vidc_common_state_change
  msm_vidc_events:msm_vidc_dma_buffer
  msm_vidc_events:msm_vidc_perf_power_scale
  msm_vidc_events:venus_hfi_var_done
  napi:napi_poll
  neigh:neigh_cleanup_and_release
  neigh:neigh_create
  neigh:neigh_event_send_dead
  neigh:neigh_event_send_done
  neigh:neigh_timer_handler
  neigh:neigh_update
  neigh:neigh_update_done
  net:napi_gro_frags_entry
  net:napi_gro_frags_exit
  net:napi_gro_receive_entry
  net:napi_gro_receive_exit
  net:net_dev_queue
  net:net_dev_start_xmit
  net:net_dev_xmit
  net:net_dev_xmit_timeout
  net:netif_receive_skb
  net:netif_receive_skb_entry
  net:netif_receive_skb_exit
  net:netif_receive_skb_list_entry
  net:netif_receive_skb_list_exit
  net:netif_rx
  net:netif_rx_entry
  net:netif_rx_exit
  netlink:netlink_extack
  nvme:nvme_async_event
  nvme:nvme_complete_rq
  nvme:nvme_setup_cmd
  nvme:nvme_sq
  oom:compact_retry
  oom:finish_task_reaping
  oom:mark_victim
  oom:oom_score_adj_update
  oom:reclaim_retry_zone
  oom:skip_task_reaping
  oom:start_task_reaping
  oom:wake_reaper
  page_isolation:test_pages_isolated
  page_pool:page_pool_release
  page_pool:page_pool_state_hold
  page_pool:page_pool_state_release
  page_pool:page_pool_update_nid
  pagemap:mm_lru_activate
  pagemap:mm_lru_insertion
  percpu:percpu_alloc_percpu
  percpu:percpu_alloc_percpu_fail
  percpu:percpu_create_chunk
  percpu:percpu_destroy_chunk
  percpu:percpu_free_percpu
  perf_trace_counters:sched_switch_ctrs_cfg
  perf_trace_counters:sched_switch_with_ctrs
  power:clock_disable
  power:clock_enable
  power:clock_set_rate
  power:cpu_frequency
  power:cpu_frequency_limits
  power:cpu_idle
  power:cpu_idle_miss
  power:dev_pm_qos_add_request
  power:dev_pm_qos_remove_request
  power:dev_pm_qos_update_request
  power:device_pm_callback_end
  power:device_pm_callback_start
  power:gpu_work_period
  power:guest_halt_poll_ns
  power:pm_qos_add_request
  power:pm_qos_remove_request
  power:pm_qos_update_flags
  power:pm_qos_update_request
  power:pm_qos_update_target
  power:power_domain_target
  power:powernv_throttle
  power:pstate_sample
  power:suspend_resume
  power:wakeup_source_activate
  power:wakeup_source_deactivate
  preemptirq:irq_disable
  preemptirq:irq_enable
  preemptirq:preempt_disable
  preemptirq:preempt_enable
  preemptirq_long:irq_disable_long
  preemptirq_long:preempt_disable_long
  printk:console
  pwm:pwm_apply
  pwm:pwm_get
  qcom_haptics:qcom_haptics_fifo_hw_status
  qcom_haptics:qcom_haptics_fifo_prgm_status
  qcom_haptics:qcom_haptics_play
  qcom_haptics:qcom_haptics_status
  qcom_lpm:gov_pred_hist
  qcom_lpm:gov_pred_select
  qcom_lpm:lpm_gov_select
  qdisc:qdisc_create
  qdisc:qdisc_dequeue
  qdisc:qdisc_destroy
  qdisc:qdisc_enqueue
  qdisc:qdisc_reset
  qrtr:qrtr_ns_message
  qrtr:qrtr_ns_server_add
  qrtr:qrtr_ns_service_announce_del
  qrtr:qrtr_ns_service_announce_new
  qup_i2c_trace:i2c_log_info
  qup_i3c_trace:i3c_log_info
  qup_spi_trace:spi_log_info
  ras:aer_event
  ras:arm_event
  ras:mc_event
  ras:non_standard_event
  raw_syscalls:sys_enter
  raw_syscalls:sys_exit
  rcu:rcu_barrier
  rcu:rcu_batch_end
  rcu:rcu_batch_start
  rcu:rcu_callback
  rcu:rcu_dyntick
  rcu:rcu_exp_funnel_lock
  rcu:rcu_exp_grace_period
  rcu:rcu_fqs
  rcu:rcu_future_grace_period
  rcu:rcu_grace_period
  rcu:rcu_grace_period_init
  rcu:rcu_invoke_callback
  rcu:rcu_invoke_kfree_bulk_callback
  rcu:rcu_invoke_kvfree_callback
  rcu:rcu_kvfree_callback
  rcu:rcu_nocb_wake
  rcu:rcu_preempt_task
  rcu:rcu_quiescent_state_report
  rcu:rcu_segcb_stats
  rcu:rcu_stall_warning
  rcu:rcu_torture_read
  rcu:rcu_unlock_preempted_task
  rcu:rcu_utilization
  regmap:regcache_drop_region
  regmap:regcache_sync
  regmap:regmap_async_complete_done
  regmap:regmap_async_complete_start
  regmap:regmap_async_io_complete
  regmap:regmap_async_write_start
  regmap:regmap_bulk_read
  regmap:regmap_bulk_write
  regmap:regmap_cache_bypass
  regmap:regmap_cache_only
  regmap:regmap_hw_read_done
  regmap:regmap_hw_read_start
  regmap:regmap_hw_write_done
  regmap:regmap_hw_write_start
  regmap:regmap_reg_read
  regmap:regmap_reg_read_cache
  regmap:regmap_reg_write
  regulator:regulator_bypass_disable
  regulator:regulator_bypass_disable_complete
  regulator:regulator_bypass_enable
  regulator:regulator_bypass_enable_complete
  regulator:regulator_disable
  regulator:regulator_disable_complete
  regulator:regulator_enable
  regulator:regulator_enable_complete
  regulator:regulator_enable_delay
  regulator:regulator_set_voltage
  regulator:regulator_set_voltage_complete
  rmnet:print_icmp_rx
  rmnet:print_icmp_tx
  rmnet:print_pfn
  rmnet:print_skb_gso
  rmnet:print_tcp_rx
  rmnet:print_tcp_tx
  rmnet:print_udp_rx
  rmnet:print_udp_tx
  rmnet:rmnet_err
  rmnet:rmnet_freq_boost
  rmnet:rmnet_freq_reset
  rmnet:rmnet_freq_update
  rmnet:rmnet_high
  rmnet:rmnet_low
  rmnet:rmnet_perf_err
  rmnet:rmnet_perf_high
  rmnet:rmnet_perf_low
  rmnet:rmnet_shs_err
  rmnet:rmnet_shs_high
  rmnet:rmnet_shs_low
  rmnet:rmnet_shs_wq_err
  rmnet:rmnet_shs_wq_high
  rmnet:rmnet_shs_wq_low
  rmnet:rmnet_xmit_skb
  rndis_ipa:rndis_netif_ni
  rndis_ipa:rndis_status_rcvd
  rndis_ipa:rndis_tx_dp
  rpm:rpm_idle
  rpm:rpm_resume
  rpm:rpm_return_int
  rpm:rpm_status
  rpm:rpm_suspend
  rpm:rpm_usage
  rpmh:rpmh_drv_enable
  rpmh:rpmh_send_msg
  rpmh:rpmh_solver_set
  rpmh:rpmh_switch_channel
  rpmh:rpmh_tx_done
  rproc_qcom:rproc_qcom_event
  rtc:rtc_alarm_irq_enable
  rtc:rtc_irq_set_freq
  rtc:rtc_irq_set_state
  rtc:rtc_read_alarm
  rtc:rtc_read_offset
  rtc:rtc_read_time
  rtc:rtc_set_alarm
  rtc:rtc_set_offset
  rtc:rtc_set_time
  rtc:rtc_timer_dequeue
  rtc:rtc_timer_enqueue
  rtc:rtc_timer_fired
  rwmmio:rwmmio_post_read
  rwmmio:rwmmio_post_write
  rwmmio:rwmmio_read
  rwmmio:rwmmio_write
  sched:sched_blocked_reason
  sched:sched_kthread_stop
  sched:sched_kthread_stop_ret
  sched:sched_kthread_work_execute_end
  sched:sched_kthread_work_execute_start
  sched:sched_kthread_work_queue_work
  sched:sched_migrate_task
  sched:sched_move_numa
  sched:sched_pi_setprio
  sched:sched_process_exec
  sched:sched_process_exit
  sched:sched_process_fork
  sched:sched_process_free
  sched:sched_process_hang
  sched:sched_process_wait
  sched:sched_stat_blocked
  sched:sched_stat_iowait
  sched:sched_stat_runtime
  sched:sched_stat_sleep
  sched:sched_stat_wait
  sched:sched_stick_numa
  sched:sched_swap_numa
  sched:sched_switch
  sched:sched_wait_task
  sched:sched_wake_idle_without_ipi
  sched:sched_wakeup
  sched:sched_wakeup_new
  sched:sched_waking
  schedwalt:core_ctl_eval_need
  schedwalt:core_ctl_notif_data
  schedwalt:core_ctl_sbt
  schedwalt:core_ctl_set_boost
  schedwalt:core_ctl_update_nr_need
  schedwalt:halt_cpus
  schedwalt:halt_cpus_start
  schedwalt:sched_busy_hyst_time
  schedwalt:sched_cgroup_attach
  schedwalt:sched_compute_energy
  schedwalt:sched_cpu_util
  schedwalt:sched_enq_deq_task
  schedwalt:sched_find_best_target
  schedwalt:sched_fmax_uncap
  schedwalt:sched_get_nr_running_avg
  schedwalt:sched_get_task_cpu_cycles
  schedwalt:sched_load_to_gov
  schedwalt:sched_migration_update_sum
  schedwalt:sched_overutilized
  schedwalt:sched_penalty
  schedwalt:sched_pipeline_swapped
  schedwalt:sched_pipeline_tasks
  schedwalt:sched_ravg_window_change
  schedwalt:sched_rt_find_lowest_rq
  schedwalt:sched_select_task_rt
  schedwalt:sched_set_boost
  schedwalt:sched_set_preferred_cluster
  schedwalt:sched_task_handler
  schedwalt:sched_task_util
  schedwalt:sched_update_history
  schedwalt:sched_update_pred_demand
  schedwalt:sched_update_task_ravg
  schedwalt:sched_update_task_ravg_mini
  schedwalt:sched_update_updown_early_migrate_values
  schedwalt:sched_update_updown_migrate_values
  schedwalt:update_cpu_capacity
  schedwalt:walt_active_load_balance
  schedwalt:walt_cfs_deactivate_mvp_task
  schedwalt:walt_cfs_mvp_pick_next
  schedwalt:walt_cfs_mvp_wakeup_nopreempt
  schedwalt:walt_cfs_mvp_wakeup_preempt
  schedwalt:walt_find_busiest_queue
  schedwalt:walt_lb_cpu_util
  schedwalt:walt_newidle_balance
  schedwalt:walt_nohz_balance_kick
  schedwalt:walt_window_rollover
  schedwalt:waltgov_next_freq
  schedwalt:waltgov_util_update
  scmi:scmi_fc_call
  scmi:scmi_msg_dump
  scmi:scmi_rx_done
  scmi:scmi_xfer_begin
  scmi:scmi_xfer_end
  scmi:scmi_xfer_response_wait
  scsi:scsi_dispatch_cmd_done
  scsi:scsi_dispatch_cmd_error
  scsi:scsi_dispatch_cmd_start
  scsi:scsi_dispatch_cmd_timeout
  scsi:scsi_eh_wakeup
  sde:sde_cmd_release_bw
  sde:sde_encoder_underrun
  sde:sde_evtlog
  sde:sde_hw_fence_status
  sde:sde_perf_calc_crtc
  sde:sde_perf_crtc_update
  sde:sde_perf_set_ot
  sde:sde_perf_set_qos_luts
  sde:sde_perf_uidle_cntr
  sde:sde_perf_uidle_status
  sde:sde_perf_update_bus
  sde:tracing_mark_write
  secure_buffer:hyp_assign_batch_end
  secure_buffer:hyp_assign_batch_start
  secure_buffer:hyp_assign_end
  secure_buffer:hyp_assign_info
  serial:serial_info
  signal:signal_deliver
  signal:signal_generate
  skb:consume_skb
  skb:kfree_skb
  skb:skb_copy_datagram_iovec
  slimbus:slimbus_dbg
  smbus:smbus_read
  smbus:smbus_reply
  smbus:smbus_result
  smbus:smbus_write
  smcinvoke:invoke_cmd_handler
  smcinvoke:marshal_in_invoke_req
  smcinvoke:marshal_in_tzcb_req_fd
  smcinvoke:marshal_in_tzcb_req_handle
  smcinvoke:marshal_out_invoke_req
  smcinvoke:marshal_out_tzcb_req
  smcinvoke:prepare_send_scm_msg
  smcinvoke:process_accept_req_has_response
  smcinvoke:process_accept_req_placed
  smcinvoke:process_accept_req_ret
  smcinvoke:process_invoke_req_result
  smcinvoke:process_invoke_req_tzhandle
  smcinvoke:process_invoke_request_from_kernel_client
  smcinvoke:process_log_info
  smcinvoke:process_tzcb_req_handle
  smcinvoke:process_tzcb_req_result
  smcinvoke:process_tzcb_req_wait
  smcinvoke:put_pending_cbobj_locked
  smcinvoke:release_mem_obj_locked
  smcinvoke:smcinvoke_create_bridge
  smcinvoke:smcinvoke_ioctl
  smcinvoke:smcinvoke_release
  smcinvoke:smcinvoke_release_filp
  smcinvoke:smcinvoke_release_from_kernel_client
  smcinvoke:status
  sock:inet_sk_error_report
  sock:inet_sock_set_state
  sock:sock_exceed_buf_limit
  sock:sock_rcvqueue_full
  spi:spi_controller_busy
  spi:spi_controller_idle
  spi:spi_message_done
  spi:spi_message_start
  spi:spi_message_submit
  spi:spi_set_cs
  spi:spi_setup
  spi:spi_transfer_start
  spi:spi_transfer_stop
  spmi:spmi_cmd
  spmi:spmi_read_begin
  spmi:spmi_read_end
  spmi:spmi_write_begin
  spmi:spmi_write_end
  swiotlb:swiotlb_bounced
  synthetic:rss_stat_throttled
  synthetic:suspend_resume_minimal
  task:task_newtask
  task:task_rename
  tcp:tcp_bad_csum
  tcp:tcp_cong_state_set
  tcp:tcp_destroy_sock
  tcp:tcp_probe
  tcp:tcp_rcv_space_adjust
  tcp:tcp_receive_reset
  tcp:tcp_retransmit_skb
  tcp:tcp_retransmit_synack
  tcp:tcp_send_reset
  thermal:cdev_update
  thermal:thermal_power_cpu_get_power_simple
  thermal:thermal_power_cpu_limit
  thermal:thermal_power_devfreq_get_power
  thermal:thermal_power_devfreq_limit
  thermal:thermal_temperature
  thermal:thermal_zone_trip
  thermal_power_allocator:thermal_power_allocator
  thermal_power_allocator:thermal_power_allocator_pid
  thermal_pressure:thermal_pressure_update
  thp:hugepage_set_pmd
  thp:hugepage_update
  thp:remove_migration_pmd
  thp:set_migration_pmd
  timer:hrtimer_cancel
  timer:hrtimer_expire_entry
  timer:hrtimer_expire_exit
  timer:hrtimer_init
  timer:hrtimer_start
  timer:itimer_expire
  timer:itimer_state
  timer:tick_stop
  timer:timer_cancel
  timer:timer_expire_entry
  timer:timer_expire_exit
  timer:timer_init
  timer:timer_start
  tipc:tipc_l2_device_event
  tipc:tipc_link_bc_ack
  tipc:tipc_link_conges
  tipc:tipc_link_dump
  tipc:tipc_link_fsm
  tipc:tipc_link_reset
  tipc:tipc_link_retrans
  tipc:tipc_link_timeout
  tipc:tipc_link_too_silent
  tipc:tipc_list_dump
  tipc:tipc_node_check_state
  tipc:tipc_node_create
  tipc:tipc_node_delete
  tipc:tipc_node_dump
  tipc:tipc_node_fsm
  tipc:tipc_node_link_down
  tipc:tipc_node_link_up
  tipc:tipc_node_lost_contact
  tipc:tipc_node_reset_links
  tipc:tipc_node_timeout
  tipc:tipc_proto_build
  tipc:tipc_proto_rcv
  tipc:tipc_sk_advance_rx
  tipc:tipc_sk_create
  tipc:tipc_sk_drop_msg
  tipc:tipc_sk_dump
  tipc:tipc_sk_filter_rcv
  tipc:tipc_sk_overlimit1
  tipc:tipc_sk_overlimit2
  tipc:tipc_sk_poll
  tipc:tipc_sk_rej_msg
  tipc:tipc_sk_release
  tipc:tipc_sk_sendmcast
  tipc:tipc_sk_sendmsg
  tipc:tipc_sk_sendstream
  tipc:tipc_sk_shutdown
  tipc:tipc_skb_dump
  tlb:tlb_flush
  ubwcp:ubwcp_dma_sync_single_for_cpu_end
  ubwcp:ubwcp_dma_sync_single_for_cpu_start
  ubwcp:ubwcp_dma_sync_single_for_device_end
  ubwcp:ubwcp_dma_sync_single_for_device_start
  ubwcp:ubwcp_free_buffer_end
  ubwcp:ubwcp_free_buffer_start
  ubwcp:ubwcp_hw_flush_end
  ubwcp:ubwcp_hw_flush_start
  ubwcp:ubwcp_init_buffer_end
  ubwcp:ubwcp_init_buffer_start
  ubwcp:ubwcp_lock_end
  ubwcp:ubwcp_lock_start
  ubwcp:ubwcp_memremap_pages_end
  ubwcp:ubwcp_memremap_pages_start
  ubwcp:ubwcp_memunmap_pages_end
  ubwcp:ubwcp_memunmap_pages_start
  ubwcp:ubwcp_offline_sync_end
  ubwcp:ubwcp_offline_sync_start
  ubwcp:ubwcp_prefetch_tgt_end
  ubwcp:ubwcp_prefetch_tgt_start
  ubwcp:ubwcp_probe
  ubwcp:ubwcp_remove
  ubwcp:ubwcp_set_buf_attrs_end
  ubwcp:ubwcp_set_buf_attrs_start
  ubwcp:ubwcp_set_direct_map_range_uncached_end
  ubwcp:ubwcp_set_direct_map_range_uncached_start
  ubwcp:ubwcp_unlock_end
  ubwcp:ubwcp_unlock_start
  ucsi:ucsi_connector_change
  ucsi:ucsi_register_altmode
  ucsi:ucsi_register_port
  ucsi:ucsi_reset_ppm
  ucsi:ucsi_run_command
  udp:udp_fail_queue_rcv_skb
  ufs:ufshcd_auto_bkops_state
  ufs:ufshcd_clk_gating
  ufs:ufshcd_clk_scaling
  ufs:ufshcd_command
  ufs:ufshcd_exception_event
  ufs:ufshcd_init
  ufs:ufshcd_profile_clk_gating
  ufs:ufshcd_profile_clk_scaling
  ufs:ufshcd_profile_hibern8
  ufs:ufshcd_runtime_resume
  ufs:ufshcd_runtime_suspend
  ufs:ufshcd_system_resume
  ufs:ufshcd_system_suspend
  ufs:ufshcd_uic_command
  ufs:ufshcd_upiu
  ufs:ufshcd_wl_resume
  ufs:ufshcd_wl_runtime_resume
  ufs:ufshcd_wl_runtime_suspend
  ufs:ufshcd_wl_suspend
  ufsqcom:ufs_qcom_clk_scale_notify
  ufsqcom:ufs_qcom_command
  ufsqcom:ufs_qcom_hce_enable_notify
  ufsqcom:ufs_qcom_hook_check_int_errors
  ufsqcom:ufs_qcom_link_startup_notify
  ufsqcom:ufs_qcom_pwr_change_notify
  ufsqcom:ufs_qcom_resume
  ufsqcom:ufs_qcom_setup_clocks
  ufsqcom:ufs_qcom_shutdown
  ufsqcom:ufs_qcom_suspend
  ufsqcom:ufs_qcom_uic
  usb_gadget:p_config_usb_cfg_link_0
  usb_gadget:p_config_usb_cfg_unlink_0
  usb_gadget:p_gadget_dev_desc_UDC_store_0
  usb_gadget:p_unregister_gadget_item_0
  usb_gadget:r_config_usb_cfg_link_0
  usb_gadget:r_gadget_dev_desc_UDC_store_0
  v4l2:v4l2_dqbuf
  v4l2:v4l2_qbuf
  v4l2:vb2_v4l2_buf_done
  v4l2:vb2_v4l2_buf_queue
  v4l2:vb2_v4l2_dqbuf
  v4l2:vb2_v4l2_qbuf
  vb2:vb2_buf_done
  vb2:vb2_buf_queue
  vb2:vb2_dqbuf
  vb2:vb2_qbuf
  vmscan:mm_shrink_slab_end
  vmscan:mm_shrink_slab_start
  vmscan:mm_vmscan_direct_reclaim_begin
  vmscan:mm_vmscan_direct_reclaim_end
  vmscan:mm_vmscan_kswapd_sleep
  vmscan:mm_vmscan_kswapd_wake
  vmscan:mm_vmscan_lru_isolate
  vmscan:mm_vmscan_lru_shrink_active
  vmscan:mm_vmscan_lru_shrink_inactive
  vmscan:mm_vmscan_memcg_reclaim_begin
  vmscan:mm_vmscan_memcg_reclaim_end
  vmscan:mm_vmscan_memcg_softlimit_reclaim_begin
  vmscan:mm_vmscan_memcg_softlimit_reclaim_end
  vmscan:mm_vmscan_node_reclaim_begin
  vmscan:mm_vmscan_node_reclaim_end
  vmscan:mm_vmscan_throttled
  vmscan:mm_vmscan_wakeup_kswapd
  vmscan:mm_vmscan_write_folio
  vsock:virtio_transport_alloc_pkt
  vsock:virtio_transport_recv_pkt
  watchdog:watchdog_ping
  watchdog:watchdog_set_timeout
  watchdog:watchdog_start
  watchdog:watchdog_stop
  wda:wda_client_state_down
  wda:wda_client_state_up
  wda:wda_set_powersave_mode
  wlan:dp_ce_tasklet_sched_latency
  wlan:dp_del_reg_write
  wlan:dp_rx_generic_ip_pkt
  wlan:dp_rx_pkt
  wlan:dp_rx_tcp_pkt
  wlan:dp_rx_udp_pkt
  wlan:dp_tx_comp_generic_ip_pkt
  wlan:dp_tx_comp_pkt
  wlan:dp_tx_comp_tcp_pkt
  wlan:dp_tx_comp_udp_pkt
  workqueue:workqueue_activate_work
  workqueue:workqueue_execute_end
  workqueue:workqueue_execute_start
  workqueue:workqueue_queue_work
  writeback:balance_dirty_pages
  writeback:bdi_dirty_ratelimit
  writeback:flush_foreign
  writeback:folio_wait_writeback
  writeback:global_dirty_state
  writeback:inode_foreign_history
  writeback:inode_switch_wbs
  writeback:sb_clear_inode_writeback
  writeback:sb_mark_inode_writeback
  writeback:track_foreign_dirty
  writeback:wbc_writepage
  writeback:writeback_bdi_register
  writeback:writeback_dirty_folio
  writeback:writeback_dirty_inode
  writeback:writeback_dirty_inode_enqueue
  writeback:writeback_dirty_inode_start
  writeback:writeback_exec
  writeback:writeback_lazytime
  writeback:writeback_lazytime_iput
  writeback:writeback_mark_inode_dirty
  writeback:writeback_pages_written
  writeback:writeback_queue
  writeback:writeback_queue_io
  writeback:writeback_sb_inodes_requeue
  writeback:writeback_single_inode
  writeback:writeback_single_inode_start
  writeback:writeback_start
  writeback:writeback_wait
  writeback:writeback_wake_background
  writeback:writeback_write_inode
  writeback:writeback_write_inode_start
  writeback:writeback_written
  xdp:mem_connect
  xdp:mem_disconnect
  xdp:mem_return_failed
  xdp:xdp_bulk_tx
  xdp:xdp_cpumap_enqueue
  xdp:xdp_cpumap_kthread
  xdp:xdp_devmap_xmit
  xdp:xdp_exception
  xdp:xdp_redirect
  xdp:xdp_redirect_err
  xdp:xdp_redirect_map
  xdp:xdp_redirect_map_err
  xhci-hcd:xhci_add_endpoint
  xhci-hcd:xhci_address_ctrl_ctx
  xhci-hcd:xhci_address_ctx
  xhci-hcd:xhci_alloc_dev
  xhci-hcd:xhci_alloc_virt_device
  xhci-hcd:xhci_configure_endpoint
  xhci-hcd:xhci_configure_endpoint_ctrl_ctx
  xhci-hcd:xhci_dbc_alloc_request
  xhci-hcd:xhci_dbc_free_request
  xhci-hcd:xhci_dbc_gadget_ep_queue
  xhci-hcd:xhci_dbc_giveback_request
  xhci-hcd:xhci_dbc_handle_event
  xhci-hcd:xhci_dbc_handle_transfer
  xhci-hcd:xhci_dbc_queue_request
  xhci-hcd:xhci_dbg_address
  xhci-hcd:xhci_dbg_cancel_urb
  xhci-hcd:xhci_dbg_context_change
  xhci-hcd:xhci_dbg_init
  xhci-hcd:xhci_dbg_quirks
  xhci-hcd:xhci_dbg_reset_ep
  xhci-hcd:xhci_dbg_ring_expansion
  xhci-hcd:xhci_discover_or_reset_device
  xhci-hcd:xhci_free_dev
  xhci-hcd:xhci_free_virt_device
  xhci-hcd:xhci_get_port_status
  xhci-hcd:xhci_handle_cmd_addr_dev
  xhci-hcd:xhci_handle_cmd_config_ep
  xhci-hcd:xhci_handle_cmd_disable_slot
  xhci-hcd:xhci_handle_cmd_reset_dev
  xhci-hcd:xhci_handle_cmd_reset_ep
  xhci-hcd:xhci_handle_cmd_set_deq
  xhci-hcd:xhci_handle_cmd_set_deq_ep
  xhci-hcd:xhci_handle_cmd_stop_ep
  xhci-hcd:xhci_handle_command
  xhci-hcd:xhci_handle_event
  xhci-hcd:xhci_handle_port_status
  xhci-hcd:xhci_handle_transfer
  xhci-hcd:xhci_hub_status_data
  xhci-hcd:xhci_inc_deq
  xhci-hcd:xhci_inc_enq
  xhci-hcd:xhci_queue_trb
  xhci-hcd:xhci_ring_alloc
  xhci-hcd:xhci_ring_ep_doorbell
  xhci-hcd:xhci_ring_expansion
  xhci-hcd:xhci_ring_free
  xhci-hcd:xhci_ring_host_doorbell
  xhci-hcd:xhci_setup_addressable_virt_device
  xhci-hcd:xhci_setup_device
  xhci-hcd:xhci_setup_device_slot
  xhci-hcd:xhci_stop_device
  xhci-hcd:xhci_urb_dequeue
  xhci-hcd:xhci_urb_enqueue
  xhci-hcd:xhci_urb_giveback

# simpleperf使用示例

```bash
simpleperf stat -a - e raw-mem-access -e stalled-cycles-backend -e cpu-cycles -e raw-mem-access-rd -e raw-mem-access-wr                                                <
^CPerformance counter statistics:

#           count  event_name               # count / runtime
   67,033,690,649  raw-mem-access           # 548.437 M/sec
   99,126,447,630  stalled-cycles-backend   # 811.009 M/sec
  240,358,000,494  cpu-cycles               # 1.966515 GHz
   43,618,981,179  raw-mem-access-rd        # 356.874 M/sec
   23,414,482,774  raw-mem-access-wr        # 191.567 M/sec

Total test time: 15.275519 seconds.
```