---

title:grub

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

grub rescue> set prefix=(hd0,1)/boot/grub
grub rescue> set root=(hd0,1)
grub rescue> insmod normal
grub rescue> normal
grub rescue> insmod linux
grub rescue> linux /boot/vmlinuz-3.13.0-29-generic root=/dev/sda1
grub rescue> initrd /boot/initrd.img-3.13.0-29-generic
grub rescue> boot


# chroot 

mount -n --bind /root/dev /dev/.static/dev
mount -n --move /dev /root/dev
mount -n --move /proc /root/proc
mount -n --move /sys /root/sys

# mount failed

rootfstype=ext4 add after root=