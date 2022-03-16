# openwrt的各种玩法

## 1. 想要所有设备科学上网，不需要有wifi功能的新硬件

你只需要一个能运行openwrt的硬件，然后设置你设备的默认网关为openwrt设备即可。

## 2. openwrt也可以的容器中运行

```
$ docker import http://downloads.openwrt.org/attitude_adjustment/12.09/x86/generic/openwrt-x86-generic-rootfs.tar.gz openwrt-x86-generic-rootfs
$ docker images
REPOSITORY                           TAG                   IMAGE ID            CREATED             VIRTUAL SIZE
openwrt-x86-generic-rootfs           latest                2cebd16f086c        6 minutes ago       5.283 MB

$ docker run -i openwrt-x86-generic-rootfs -name openwrt /sbin/init
$ docker exec -it openwrt /bin/ash


```

ref: https://openwrt.org/docs/guide-user/virtualization/docker_openwrt_image

raspberry pi 3B :https://archive.openwrt.org/releases/21.02.2/targets/bcm27xx/bcm2710/openwrt-21.02.2-bcm27xx-bcm2710-rpi-3-ext4-factory.img.gz

### docker install

```
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh ./get-docker.sh
```

## 3. openwrt的树莓派镜像可以用rpi-imager安装

https://firmware-selector.openwrt.org/?version=21.02.0&target=bcm27xx%2Fbcm2710&id=rpi-3


## 4. rpi-imager在linux下失败可以去windows下试试

Linux下的USB可能有些问题，可能是缓冲区太大了。
