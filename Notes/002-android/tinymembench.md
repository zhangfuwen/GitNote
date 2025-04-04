

```
tinymembench v0.3.9 (simple benchmark for memory throughput and latency)

==========================================================================
== Memory bandwidth tests                                               ==
==                                                                      ==
== Note 1: 1MB = 1000000 bytes                                          ==
== Note 2: Results for 'copy' tests show how many bytes can be          ==
==         copied per second (adding together read and writen           ==
==         bytes would have provided twice higher numbers)              ==
== Note 3: 2-pass copy means that we are using a small temporary buffer ==
==         to first fetch data into it, and only then write it to the   ==
==         destination (source -> L1 cache, L1 cache -> destination)    ==
== Note 4: If sample standard deviation exceeds 0.1%, it is shown in    ==
==         brackets                                                     ==
==========================================================================

 C copy backwards                                     :   8063.0 MB/s (17.3%)
 C copy                                               :   9183.1 MB/s (4.2%)
 C copy prefetched (32 bytes step)                    :   9130.7 MB/s (2.8%)
 C copy prefetched (64 bytes step)                    :   9509.1 MB/s (1.6%)
 C 2-pass copy                                        :   9596.6 MB/s (1.8%)
 C 2-pass copy prefetched (32 bytes step)             :   9620.6 MB/s (1.9%)
 C 2-pass copy prefetched (64 bytes step)             :   9603.1 MB/s (2.1%)
 C fill                                               :  19885.9 MB/s (2.2%)
 ---
 standard memcpy                                      :   9132.4 MB/s (2.2%)
 standard memset                                      :  19937.4 MB/s (1.5%)

==========================================================================
== Memory latency test                                                  ==
==                                                                      ==
== Average time is measured for random memory accesses in the buffers   ==
== of different sizes. The larger is the buffer, the more significant   ==
== are relative contributions of TLB, L1/L2 cache misses and SDRAM      ==
== accesses. For extremely large buffer sizes we are expecting to see   ==
== page table walk with several requests to SDRAM for almost every      ==
== memory access (though 64MiB is not nearly large enough to experience ==
== this effect to its fullest).                                         ==
==                                                                      ==
== Note 1: All the numbers are representing extra time, which needs to  ==
==         be added to L1 cache latency. The cycle timings for L1 cache ==
==         latency can be usually found in the processor documentation. ==
== Note 2: Dual random read means that we are simultaneously performing ==
==         two independent memory accesses at a time. In the case if    ==
==         the memory subsystem can't handle multiple outstanding       ==
==         requests, dual random read has the same timings as two       ==
==         single reads performed one after another.                    ==
==========================================================================

block size : single random read / dual random read, [MADV_NOHUGEPAGE]
      1024 :    0.0 ns          /     0.0 ns
      2048 :    0.0 ns          /     0.0 ns
      4096 :    0.0 ns          /     0.0 ns
      8192 :    0.0 ns          /     0.0 ns
     16384 :    0.0 ns          /     0.0 ns
     32768 :    0.0 ns          /     0.0 ns
     65536 :    1.1 ns          /     1.6 ns
    131072 :    1.6 ns          /     2.3 ns
    262144 :    2.4 ns          /     3.4 ns
    524288 :    5.5 ns          /     7.4 ns
   1048576 :   12.7 ns          /    16.9 ns
   2097152 :   27.1 ns          /    35.5 ns
   4194304 :   83.5 ns          /   112.3 ns
   8388608 :  113.4 ns          /   134.0 ns
  16777216 :  134.6 ns          /   153.9 ns
  33554432 :  148.1 ns          /   164.2 ns
  67108864 :  158.6 ns          /   174.7 ns

block size : single random read / dual random read, [MADV_HUGEPAGE]
      1024 :    0.0 ns          /     0.0 ns
      2048 :    0.0 ns          /     0.0 ns
      4096 :    0.0 ns          /     0.0 ns
      8192 :    0.0 ns          /     0.0 ns
     16384 :    0.0 ns          /     0.0 ns
     32768 :    0.0 ns          /     0.0 ns
     65536 :    1.0 ns          /     1.6 ns
    131072 :    1.5 ns          /     2.3 ns
    262144 :    2.4 ns          /     3.4 ns
    524288 :    5.9 ns          /     8.0 ns
   1048576 :   12.6 ns          /    17.1 ns
   2097152 :   28.5 ns          /    36.4 ns
   4194304 :   84.8 ns          /   112.0 ns
   8388608 :  114.1 ns          /   134.5 ns
  16777216 :  134.6 ns          /   152.9 ns
  33554432 :  147.3 ns          /   165.4 ns
  67108864 :  157.8 ns          /   174.2 ns
```