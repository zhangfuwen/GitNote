# 简介

zram就是将内存中的部分数据压缩存在内存里，压缩率有一倍以上。如淘宝的页面已最近最少使用的，原来占用100M的匿名页会压缩起来存放，压缩后占用40M，通过这个压缩就节省了60M内存。
![](assets/Pasted%20image%2020250226100914.png)
zram设计为linux内核中的一个块设备，代码在：

![](assets/Pasted%20image%2020250226112827.png)
从[Konfig文件](https://github.com/torvalds/linux/blob/v5.16/drivers/block/zram/Kconfig)看，它支持zstd, lzo等多种压缩算法。
![](assets/Pasted%20image%2020250226113025.png)

	Set disk size by writing the value to sysfs node 'disksize'. The value can be either in bytes or you can use mem suffixes. Examples:
	
	```bash
	# Initialize /dev/zram0 with 50MB disksize
	echo $((50*1024*1024)) > /sys/block/zram0/disksize
	
	# Using mem suffixes
	echo 256K > /sys/block/zram0/disksize
	echo 512M > /sys/block/zram0/disksize
	echo 1G > /sys/block/zram0/disksize
	```
	
	Note: There is little point creating a zram of greater than twice the size of memory since we expect a 2:1 compression ratio. Note that zram uses about 0.1% of the size of the disk when not in use so a huge zram is wasteful.

最简单的使用方法是[在zran块设备上建立swap分区](https://segmentfault.com/a/1190000041578292)：

```bash
mkswap /dev/zram0
swapon /dev/zram0
```

## zram的内存管理

参考：https://zhuanlan.zhihu.com/p/631300401
https://github.com/huataihuang/cloud-atlas/blob/master/source/linux/redhat_linux/kernel/zram.rst
  
zram 使用了 [Zsmalloc](https://zhida.zhihu.com/search?content_id=228420307&content_type=Article&match_order=1&q=Zsmalloc&zhida_source=entity) 分配器来管理它的内存空间，Zsmalloc 分配器尝试将多个相同大小的对象存放在组合页（称为 zspage）中，这个组合页不要求物理连续，从而提高内存的使用率。  
首先会根据 zram 的内存中页面的个数，创建相应个数的 zram table，每个 zram table 都对应一个页面；然后会调用 zs_create_pool 创建一个 zsmalloc 的内存池，以后所有的页面申请和释放都是通过 zs_malloc 和 zs_free 来分配和释放相对应的对象。