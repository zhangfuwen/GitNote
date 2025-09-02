---

title: grub

---


# change resolution


Changing gfxmode variable and temporary switching to different terminal_output (console or vga_text) should change screen resolution.


```
terminal_output console
set gfxmode=1280x1024
terminal_output gfxterm
```

videoinfo command shows available resolutions

You can also list multiple resolutions separated by either commas or semicolons, and GRUB will pick the first resolution the hardware can support.

If you want the Linux kernel to maintain the resolution that was set by GRUB, you'll also need:

```
set gfxpayload=keep
```

# rescue command

```
grub rescue> set prefix=(hd0,1)/boot/grub
grub rescue> set root=(hd0,1)
grub rescue> insmod normal
grub rescue> normal
grub rescue> insmod linux
grub rescue> linux /boot/vmlinuz-3.13.0-29-generic root=/dev/sda1
grub rescue> initrd /boot/initrd.img-3.13.0-29-generic
grub rescue> boot
```


# chroot 

```bash
mount -n --bind /root/dev /dev/.static/dev
mount -n --move /dev /root/dev
mount -n --move /proc /root/proc
mount -n --move /sys /root/sys

exec chroot /root
exec systemd --system
```

## in conclusion

in initramfs:

```bash
mount -t ext4 /dev/sda4 /root
cd /root
exec ./initramfs_run.sh
```

`initramfs_run.sh`:

```bash
mount -t proc proc ./proc
mount -t sysfs sysfs ./sys
mount --bind /dev ./dev
mount --bind /dev/pts ./dev/pts
exec chroot . sh -c "exec /bin/systemd --system"
```

# mount failed

```
rootfstype=ext4 add after root=
```

problem is `root=(hd0,4)` is wrong, should be `root=/dev/sda4`


# mount img partition 

```bash

function mount_img_part() {
    if [[ $# !=2 ]]; then 
        echo "error: usage $1 img_file dir"
        return
    fi

    img_file=$1
    dir=$2

    if [[ ! -f $img_file]]; then
        echo "error: image file $img_file does not exist!"
        return
    fi

    start_sector=$(sfdisk -l $img_file | grep img2 | awk '{print $2}')
    start_byte=$(expr $start_sector \* 512)

    if [[ -d $dir ]]; then
        umount $dir > /dev/null 2>&1
    else
        mkdir -p $dir
    fi
    mkdir $dir
    mount-o loop,offset=$start_byte $img_file $dir
}

mount ./oceanosxxxx.img /tmp/ocean_mount

 ```


# make usb disk

```bash
DEVNODE=/dev/sde

echo "partioning.."
sudo sfdisk $DEVNODE < << EOF
label: gpt
label-id: 05C39BC1-E9C4-4B46-A128-5139DFF4909F
device: /dev/sde
unit: sectors
first-lba: 2048
last-lba: 120831966
sector-size: 512

/dev/sde1 : start=        2048, size=     2097152, type=C12A7328-F81F-11D2-BA4B-00A0C93EC93B, uuid=1F8AAE56-AD2C-764A-840D-4AE5A5182646, name="EFI-SYSTEM"
/dev/sde2 : start=     2099200, size=     2097152, type=0FC63DAF-8483-4772-8E79-3D69D8477DE4, uuid=06216244-E7B8-1242-B5E4-8DB10A84E603, name="BOOT"
/dev/sde3 : start=     4196352, size=     2097152, type=0FC63DAF-8483-4772-8E79-3D69D8477DE4, uuid=9EA2813E-AC7A-1F41-881C-93FAAF889FE4, name="usb_kernel"
/dev/sde4 : start=     6293504, size=    42149888, type=0FC63DAF-8483-4772-8E79-3D69D8477DE4, uuid=D6B3D08F-03A4-C442-B299-FF53DDA920F9, name="usb_rootfs"
/dev/sde5 : start=    48443392, size=    62148608, type=0FC63DAF-8483-4772-8E79-3D69D8477DE4, uuid=7CC46A0C-FDE7-4CA8-B692-6200BABE6833, name="usb_home"
EOF
echo "partioning done"

echo "dd efi..."
sudo dd if=./larkos-efi.img of=${DEVNODE}1 status=progress oflag=direct,sync bs=4K
echo "dd efi done"

echo "dd boot..."
sudo dd if=./larkos-boot.img of=${DEVNODE}2 status=progress oflag=direct,sync bs=1M
echo "dd boot done"

echo "dd kernel..."
sudo dd if=./larkos-kernel.img of=${DEVNODE}3 status=progress oflag=direct,sync bs=1M
echo "dd kernel done"

echo "dd rootfs..."
sudo dd if=./larkos-rootfs.img of=${DEVNODE}4 status=progress oflag=direct,sync bs=1M
echo "dd rootfs done"

echo "resize2fs ..."
sudo resize2fs ${DEVNODE}4
echo "resize2fs done"

echo "mkfs home..."
sudo mkfs.ext4 ${DEVNODE}5
echo "mkfs home done"

```

# loop  back

`loopback loop12345 ./rootfs.img` 不过这个过程需要把整个文件加载到内存，而efi的驱动效率不高，文件大点就搞不定。


# grub.cfg vs menu.lst

```
/boot/grub/menu.lst = old
/boot/grub/grub.conf = new
```

# EFI引导过程

efi从内置碰盘引导是参考NVRAM来列出引导项的（efibootmgr)。
efi在NVRAM之外会有fallback模式，这个模式是让你选一个盘，然后它去盘上找esp分区，然后在esp分区里找自己认识的.efi文件，这个过程一定会找的有efi/BOOT/BOOTX64.efi，剩下的就全凭
efi firmware的开发者个人爱好了，比如说会找找grubx64.efi之类的。

efi找到grubx64.efi引导之后，通过当前文件夹下包含一下grub.cfg文件，它的内容是：
```ini
search.fs_uuid eaac9bfe-14d5-437a-ad5a-e17c13010d4c root hd0,gpt5 
set prefix=($root)'/boot/grub'
configfile $prefix/grub.cfg
```

三行，意思分别是：
1. 找到一个uuid为`eaac9bfe-14d5-437a-ad5a-e17c13010d4c`的分区，取名为hd0,gpt5，并且设置变量root的值为这个分区，即(hd0,gpt5)。
2. 设置prefix这个变量值为(hd0,gpt5)/boot/grub，即把好个分区的/boot/grub文件夹做为grub后续操作的根文件夹。
3. 加载另一个grub.cfg文件，即那个分区的/boot/grub/grub.cfg文件。

通常如果我们知道哪个分区下面的linux操作系统的话，一句就能加载它了，即config_file (hdx,gpty)/boot/grub/grub.cfg，然后你就会看到grub正常的选OS界面了。

(hd0,gpt5)这样的碰盘命名规则：
磁盘从0开始，分区从1开始。

/dev/sda5这样的命令规则：
磁盘从a开始，分区从1开始。

# 参考链接

http://www.jinbuguo.com/linux/grub.cfg.html 中文的
https://www.happyassassin.net/posts/2014/01/25/uefi-boot-how-does-that-actually-work-then/ efiboot的方法，没有读完。

