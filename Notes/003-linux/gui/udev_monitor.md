---

title: udev monitor

---


# 外部显示器关闭了，系统关闭或休眠了

```bash
sudo udevadm monitor

monitor will print the received events for:
UDEV - the event which udev sends out after rule processing
KERNEL - the kernel uevent

KERNEL[65468.657336] change   /devices/pci0000:00/0000:00:01.0/0000:01:00.0/drm/card0 (drm)
UDEV  [65468.662010] change   /devices/pci0000:00/0000:00:01.0/0000:01:00.0/drm/card0 (drm)
KERNEL[65468.727467] change   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.0/0003:046D:C339.0015/input/input53/input53::numlock (leds)
KERNEL[65468.727537] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.0/0003:046D:C339.0015/input/input53/input53::numlock (leds)
KERNEL[65468.727571] change   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.0/0003:046D:C339.0015/input/input53/input53::capslock (leds)
KERNEL[65468.727615] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.0/0003:046D:C339.0015/input/input53/input53::capslock (leds)
KERNEL[65468.727662] change   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.0/0003:046D:C339.0015/input/input53/input53::scrolllock (leds)
KERNEL[65468.727700] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.0/0003:046D:C339.0015/input/input53/input53::scrolllock (leds)
UDEV  [65468.730748] change   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.0/0003:046D:C339.0015/input/input53/input53::numlock (leds)
UDEV  [65468.732685] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.0/0003:046D:C339.0015/input/input53/input53::numlock (leds)
UDEV  [65468.733043] change   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.0/0003:046D:C339.0015/input/input53/input53::capslock (leds)
UDEV  [65468.734227] change   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.0/0003:046D:C339.0015/input/input53/input53::scrolllock (leds)
UDEV  [65468.734918] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.0/0003:046D:C339.0015/input/input53/input53::capslock (leds)
UDEV  [65468.736209] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.0/0003:046D:C339.0015/input/input53/input53::scrolllock (leds)
KERNEL[65468.751543] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.0/0003:046D:C339.0015/input/input53/event19 (input)
UDEV  [65468.754058] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.0/0003:046D:C339.0015/input/input53/event19 (input)
KERNEL[65468.771397] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.0/0003:046D:C339.0015/input/input53 (input)
KERNEL[65468.771426] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.0/0003:046D:C339.0015/hidraw/hidraw0 (hidraw)
KERNEL[65468.771445] unbind   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.0/0003:046D:C339.0015 (hid)
KERNEL[65468.771462] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.0/0003:046D:C339.0015 (hid)
KERNEL[65468.771476] unbind   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.0 (usb)
KERNEL[65468.771492] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.0 (usb)
UDEV  [65468.772228] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.0/0003:046D:C339.0015/input/input53 (input)
UDEV  [65468.772265] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.0/0003:046D:C339.0015/hidraw/hidraw0 (hidraw)
UDEV  [65468.772883] unbind   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.0/0003:046D:C339.0015 (hid)
UDEV  [65468.773357] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.0/0003:046D:C339.0015 (hid)
UDEV  [65468.774082] unbind   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.0 (usb)
UDEV  [65468.774568] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.0 (usb)
KERNEL[65468.819364] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.1/0003:046D:C339.0016/input/input54/event20 (input)
UDEV  [65468.820327] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.1/0003:046D:C339.0016/input/input54/event20 (input)
KERNEL[65468.843365] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.1/0003:046D:C339.0016/input/input54 (input)
UDEV  [65468.844278] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.1/0003:046D:C339.0016/input/input54 (input)
KERNEL[65468.859356] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.1/0003:046D:C339.0016/input/input55/event21 (input)
UDEV  [65468.860192] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.1/0003:046D:C339.0016/input/input55/event21 (input)
KERNEL[65468.891532] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.1/0003:046D:C339.0016/input/input55 (input)
KERNEL[65468.891568] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.1/usbmisc/hiddev0 (usbmisc)
KERNEL[65468.891585] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.1/0003:046D:C339.0016/hidraw/hidraw1 (hidraw)
KERNEL[65468.891609] unbind   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.1/0003:046D:C339.0016 (hid)
KERNEL[65468.891627] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.1/0003:046D:C339.0016 (hid)
KERNEL[65468.891644] unbind   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.1 (usb)
KERNEL[65468.891663] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.1 (usb)
KERNEL[65468.892012] unbind   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2 (usb)
KERNEL[65468.892032] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/wakeup/wakeup63 (wakeup)
KERNEL[65468.892055] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2 (usb)
KERNEL[65468.892178] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.0/0003:046D:C52F.0017/hidraw/hidraw2 (hidraw)
UDEV  [65468.892402] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.1/0003:046D:C339.0016/input/input55 (input)
UDEV  [65468.893142] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.1/usbmisc/hiddev0 (usbmisc)
UDEV  [65468.893169] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.1/0003:046D:C339.0016/hidraw/hidraw1 (hidraw)
UDEV  [65468.893836] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.0/0003:046D:C52F.0017/hidraw/hidraw2 (hidraw)
UDEV  [65468.895154] unbind   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.1/0003:046D:C339.0016 (hid)
UDEV  [65468.895708] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.1/0003:046D:C339.0016 (hid)
UDEV  [65468.896387] unbind   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.1 (usb)
UDEV  [65468.897204] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.1 (usb)
UDEV  [65468.898766] unbind   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2 (usb)
UDEV  [65468.899646] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/wakeup/wakeup63 (wakeup)
UDEV  [65468.900204] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2 (usb)
KERNEL[65468.955378] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.1/0003:046D:C52F.0018/0003:046D:4057.0019/input/input58/mouse2 (input)
UDEV  [65468.956449] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.1/0003:046D:C52F.0018/0003:046D:4057.0019/input/input58/mouse2 (input)
KERNEL[65469.003404] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.1/0003:046D:C52F.0018/0003:046D:4057.0019/input/input58/event22 (input)
UDEV  [65469.004486] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.1/0003:046D:C52F.0018/0003:046D:4057.0019/input/input58/event22 (input)
KERNEL[65469.063343] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.1/0003:046D:C52F.0018/0003:046D:4057.0019/input/input58 (input)
KERNEL[65469.063434] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.1/0003:046D:C52F.0018/0003:046D:4057.0019/hidraw/hidraw4 (hidraw)
KERNEL[65469.063447] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.1/0003:046D:C52F.0018/0003:046D:4057.0019/power_supply/hidpp_battery_4/hwmon11 (hwmon)
KERNEL[65469.063455] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.1/0003:046D:C52F.0018/0003:046D:4057.0019/power_supply/hidpp_battery_4/wakeup64 (wakeup)
KERNEL[65469.063467] change   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.1/0003:046D:C52F.0018/0003:046D:4057.0019/power_supply/hidpp_battery_4 (power_supply)
KERNEL[65469.063489] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.1/0003:046D:C52F.0018/0003:046D:4057.0019/power_supply/hidpp_battery_4 (power_supply)
KERNEL[65469.063506] unbind   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.1/0003:046D:C52F.0018/0003:046D:4057.0019 (hid)
KERNEL[65469.063522] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.1/0003:046D:C52F.0018/0003:046D:4057.0019 (hid)
KERNEL[65469.063537] unbind   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.0/0003:046D:C52F.0017 (hid)
KERNEL[65469.063554] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.0/0003:046D:C52F.0017 (hid)
KERNEL[65469.063570] unbind   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.0 (usb)
KERNEL[65469.063586] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.0 (usb)
KERNEL[65469.063648] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.1/usbmisc/hiddev1 (usbmisc)
KERNEL[65469.063659] remove   /class/usbmisc (class)
KERNEL[65469.063717] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.1/0003:046D:C52F.0018/hidraw/hidraw3 (hidraw)
KERNEL[65469.063734] unbind   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.1/0003:046D:C52F.0018 (hid)
KERNEL[65469.063752] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.1/0003:046D:C52F.0018 (hid)
KERNEL[65469.063768] unbind   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.1 (usb)
KERNEL[65469.063784] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.1 (usb)
KERNEL[65469.064093] unbind   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3 (usb)
KERNEL[65469.064116] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3 (usb)
UDEV  [65469.064168] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.1/0003:046D:C52F.0018/0003:046D:4057.0019/input/input58 (input)
UDEV  [65469.064188] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.1/0003:046D:C52F.0018/0003:046D:4057.0019/hidraw/hidraw4 (hidraw)
UDEV  [65469.064200] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.1/0003:046D:C52F.0018/0003:046D:4057.0019/power_supply/hidpp_battery_4/hwmon11 (hwmon)
UDEV  [65469.064210] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.1/0003:046D:C52F.0018/0003:046D:4057.0019/power_supply/hidpp_battery_4/wakeup64 (wakeup)
KERNEL[65469.064377] unbind   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1:1.0 (usb)
KERNEL[65469.064397] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1:1.0 (usb)
UDEV  [65469.064661] remove   /class/usbmisc (class)
KERNEL[65469.064738] unbind   /devices/pci0000:00/0000:00:14.0/usb1/1-1 (usb)
KERNEL[65469.064761] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1 (usb)
UDEV  [65469.064917] unbind   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.0/0003:046D:C52F.0017 (hid)
UDEV  [65469.064966] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.1/0003:046D:C52F.0018/hidraw/hidraw3 (hidraw)
UDEV  [65469.065086] change   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.1/0003:046D:C52F.0018/0003:046D:4057.0019/power_supply/hidpp_battery_4 (power_supply)
UDEV  [65469.065183] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.1/usbmisc/hiddev1 (usbmisc)
UDEV  [65469.065248] unbind   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1:1.0 (usb)
UDEV  [65469.065356] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.0/0003:046D:C52F.0017 (hid)
UDEV  [65469.065524] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.1/0003:046D:C52F.0018/0003:046D:4057.0019/power_supply/hidpp_battery_4 (power_supply)
UDEV  [65469.065632] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1:1.0 (usb)
UDEV  [65469.065954] unbind   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.0 (usb)
UDEV  [65469.066298] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.0 (usb)
UDEV  [65469.066563] unbind   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.1/0003:046D:C52F.0018/0003:046D:4057.0019 (hid)
UDEV  [65469.066907] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.1/0003:046D:C52F.0018/0003:046D:4057.0019 (hid)
UDEV  [65469.067373] unbind   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.1/0003:046D:C52F.0018 (hid)
UDEV  [65469.067692] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.1/0003:046D:C52F.0018 (hid)
UDEV  [65469.068168] unbind   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.1 (usb)
UDEV  [65469.068459] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.1 (usb)
UDEV  [65469.068926] unbind   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3 (usb)
UDEV  [65469.069226] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3 (usb)
UDEV  [65469.069693] unbind   /devices/pci0000:00/0000:00:14.0/usb1/1-1 (usb)
UDEV  [65469.069983] remove   /devices/pci0000:00/0000:00:14.0/usb1/1-1 (usb)
KERNEL[65482.433200] unbind   /devices/pci0000:00/0000:00:16.0/0000:00:16.0-b638ab7e-94e2-4ea2-a552-d1c54b627f04 (mei)
KERNEL[65482.433294] remove   /devices/pci0000:00/0000:00:16.0/0000:00:16.0-b638ab7e-94e2-4ea2-a552-d1c54b627f04 (mei)
KERNEL[65482.433353] remove   /devices/pci0000:00/0000:00:16.0/0000:00:16.0-3c4852d6-d47b-4f46-b05e-b5edc1aa440e (mei)
KERNEL[65482.433404] remove   /devices/pci0000:00/0000:00:16.0/0000:00:16.0-082ee5a7-7c25-470a-9643-0c06f0466ea1 (mei)
KERNEL[65482.433451] remove   /devices/pci0000:00/0000:00:16.0/0000:00:16.0-5565a099-7fe2-45c1-a22b-d7e9dfea9a2e (mei)
KERNEL[65482.433501] remove   /devices/pci0000:00/0000:00:16.0/0000:00:16.0-dba4d603-d7ed-4931-8823-17ad585705d5 (mei)
KERNEL[65482.433550] remove   /devices/pci0000:00/0000:00:16.0/0000:00:16.0-8e6a6715-9abc-4043-88ef-9e39c6f63e0f (mei)
KERNEL[65482.433602] remove   /devices/pci0000:00/0000:00:16.0/0000:00:16.0-f908627d-13bf-4a04-b91f-a64e9245323d (mei)
KERNEL[65482.433656] remove   /devices/pci0000:00/0000:00:16.0/0000:00:16.0-dd17041c-09ea-4b17-a271-5b989867ec65 (mei)
KERNEL[65482.433705] remove   /devices/pci0000:00/0000:00:16.0/0000:00:16.0-8c2f4425-77d6-4755-aca3-891fdbc66a58 (mei)
KERNEL[65482.433753] remove   /devices/pci0000:00/0000:00:16.0/0000:00:16.0-309dcde8-ccb1-4062-8f78-600115a34327 (mei)
KERNEL[65482.433802] remove   /devices/pci0000:00/0000:00:16.0/0000:00:16.0-6861ec7b-d07a-4673-856c-7f22b4d55769 (mei)
KERNEL[65482.433849] remove   /devices/pci0000:00/0000:00:16.0/0000:00:16.0-42b3ce2f-bd9f-485a-96ae-26406230b1ff (mei)
KERNEL[65482.433897] remove   /devices/pci0000:00/0000:00:16.0/0000:00:16.0-55213584-9a29-4916-badf-0fb7ed682aeb (mei)
KERNEL[65482.433946] add      /devices/pci0000:00/0000:00:16.0/0000:00:16.0-b638ab7e-94e2-4ea2-a552-d1c54b627f04 (mei)
KERNEL[65482.433997] add      /devices/pci0000:00/0000:00:16.0/0000:00:16.0-3c4852d6-d47b-4f46-b05e-b5edc1aa440e (mei)
KERNEL[65482.434042] add      /devices/pci0000:00/0000:00:16.0/0000:00:16.0-082ee5a7-7c25-470a-9643-0c06f0466ea1 (mei)
KERNEL[65482.434088] add      /devices/pci0000:00/0000:00:16.0/0000:00:16.0-5565a099-7fe2-45c1-a22b-d7e9dfea9a2e (mei)
KERNEL[65482.434135] add      /devices/pci0000:00/0000:00:16.0/0000:00:16.0-dba4d603-d7ed-4931-8823-17ad585705d5 (mei)
KERNEL[65482.434179] add      /devices/pci0000:00/0000:00:16.0/0000:00:16.0-8e6a6715-9abc-4043-88ef-9e39c6f63e0f (mei)
KERNEL[65482.434224] add      /devices/pci0000:00/0000:00:16.0/0000:00:16.0-f908627d-13bf-4a04-b91f-a64e9245323d (mei)
KERNEL[65482.434269] add      /devices/pci0000:00/0000:00:16.0/0000:00:16.0-dd17041c-09ea-4b17-a271-5b989867ec65 (mei)
KERNEL[65482.434314] add      /devices/pci0000:00/0000:00:16.0/0000:00:16.0-8c2f4425-77d6-4755-aca3-891fdbc66a58 (mei)
KERNEL[65482.434363] add      /devices/pci0000:00/0000:00:16.0/0000:00:16.0-309dcde8-ccb1-4062-8f78-600115a34327 (mei)
KERNEL[65482.434411] add      /devices/pci0000:00/0000:00:16.0/0000:00:16.0-6861ec7b-d07a-4673-856c-7f22b4d55769 (mei)
KERNEL[65482.434458] add      /devices/pci0000:00/0000:00:16.0/0000:00:16.0-42b3ce2f-bd9f-485a-96ae-26406230b1ff (mei)
KERNEL[65482.434500] add      /devices/pci0000:00/0000:00:16.0/0000:00:16.0-55213584-9a29-4916-badf-0fb7ed682aeb (mei)
UDEV  [65482.439261] unbind   /devices/pci0000:00/0000:00:16.0/0000:00:16.0-b638ab7e-94e2-4ea2-a552-d1c54b627f04 (mei)
UDEV  [65482.439292] remove   /devices/pci0000:00/0000:00:16.0/0000:00:16.0-b638ab7e-94e2-4ea2-a552-d1c54b627f04 (mei)
UDEV  [65482.439314] remove   /devices/pci0000:00/0000:00:16.0/0000:00:16.0-dd17041c-09ea-4b17-a271-5b989867ec65 (mei)
UDEV  [65482.439333] remove   /devices/pci0000:00/0000:00:16.0/0000:00:16.0-8e6a6715-9abc-4043-88ef-9e39c6f63e0f (mei)
UDEV  [65482.439350] remove   /devices/pci0000:00/0000:00:16.0/0000:00:16.0-6861ec7b-d07a-4673-856c-7f22b4d55769 (mei)
UDEV  [65482.439367] remove   /devices/pci0000:00/0000:00:16.0/0000:00:16.0-dba4d603-d7ed-4931-8823-17ad585705d5 (mei)
UDEV  [65482.439384] remove   /devices/pci0000:00/0000:00:16.0/0000:00:16.0-8c2f4425-77d6-4755-aca3-891fdbc66a58 (mei)
UDEV  [65482.439401] remove   /devices/pci0000:00/0000:00:16.0/0000:00:16.0-309dcde8-ccb1-4062-8f78-600115a34327 (mei)
UDEV  [65482.439816] add      /devices/pci0000:00/0000:00:16.0/0000:00:16.0-b638ab7e-94e2-4ea2-a552-d1c54b627f04 (mei)
UDEV  [65482.439873] add      /devices/pci0000:00/0000:00:16.0/0000:00:16.0-8e6a6715-9abc-4043-88ef-9e39c6f63e0f (mei)
UDEV  [65482.440163] add      /devices/pci0000:00/0000:00:16.0/0000:00:16.0-309dcde8-ccb1-4062-8f78-600115a34327 (mei)
UDEV  [65482.440524] add      /devices/pci0000:00/0000:00:16.0/0000:00:16.0-dba4d603-d7ed-4931-8823-17ad585705d5 (mei)
UDEV  [65482.440823] add      /devices/pci0000:00/0000:00:16.0/0000:00:16.0-8c2f4425-77d6-4755-aca3-891fdbc66a58 (mei)
UDEV  [65482.441127] remove   /devices/pci0000:00/0000:00:16.0/0000:00:16.0-5565a099-7fe2-45c1-a22b-d7e9dfea9a2e (mei)
UDEV  [65482.441253] remove   /devices/pci0000:00/0000:00:16.0/0000:00:16.0-082ee5a7-7c25-470a-9643-0c06f0466ea1 (mei)
UDEV  [65482.441579] remove   /devices/pci0000:00/0000:00:16.0/0000:00:16.0-f908627d-13bf-4a04-b91f-a64e9245323d (mei)
UDEV  [65482.441859] add      /devices/pci0000:00/0000:00:16.0/0000:00:16.0-5565a099-7fe2-45c1-a22b-d7e9dfea9a2e (mei)
UDEV  [65482.441887] remove   /devices/pci0000:00/0000:00:16.0/0000:00:16.0-3c4852d6-d47b-4f46-b05e-b5edc1aa440e (mei)
UDEV  [65482.443424] add      /devices/pci0000:00/0000:00:16.0/0000:00:16.0-6861ec7b-d07a-4673-856c-7f22b4d55769 (mei)
UDEV  [65482.443458] remove   /devices/pci0000:00/0000:00:16.0/0000:00:16.0-55213584-9a29-4916-badf-0fb7ed682aeb (mei)
UDEV  [65482.443478] add      /devices/pci0000:00/0000:00:16.0/0000:00:16.0-f908627d-13bf-4a04-b91f-a64e9245323d (mei)
UDEV  [65482.443497] add      /devices/pci0000:00/0000:00:16.0/0000:00:16.0-082ee5a7-7c25-470a-9643-0c06f0466ea1 (mei)
UDEV  [65482.443514] add      /devices/pci0000:00/0000:00:16.0/0000:00:16.0-3c4852d6-d47b-4f46-b05e-b5edc1aa440e (mei)
KERNEL[65482.443529] change   /devices/virtual/thermal/thermal_zone2 (thermal)
UDEV  [65482.443595] add      /devices/pci0000:00/0000:00:16.0/0000:00:16.0-55213584-9a29-4916-badf-0fb7ed682aeb (mei)
UDEV  [65482.443667] remove   /devices/pci0000:00/0000:00:16.0/0000:00:16.0-42b3ce2f-bd9f-485a-96ae-26406230b1ff (mei)
UDEV  [65482.443947] change   /devices/virtual/thermal/thermal_zone2 (thermal)
UDEV  [65482.444286] add      /devices/pci0000:00/0000:00:16.0/0000:00:16.0-42b3ce2f-bd9f-485a-96ae-26406230b1ff (mei)
KERNEL[65482.444462] bind     /devices/pci0000:00/0000:00:16.0/0000:00:16.0-b638ab7e-94e2-4ea2-a552-d1c54b627f04 (mei)
KERNEL[65482.444614] change   /devices/virtual/thermal/thermal_zone3 (thermal)
UDEV  [65482.445355] bind     /devices/pci0000:00/0000:00:16.0/0000:00:16.0-b638ab7e-94e2-4ea2-a552-d1c54b627f04 (mei)
UDEV  [65482.445381] change   /devices/virtual/thermal/thermal_zone3 (thermal)
KERNEL[65482.445775] change   /devices/virtual/thermal/thermal_zone4 (thermal)
UDEV  [65482.446026] change   /devices/virtual/thermal/thermal_zone4 (thermal)
KERNEL[65482.446284] change   /devices/virtual/thermal/thermal_zone5 (thermal)
UDEV  [65482.446534] change   /devices/virtual/thermal/thermal_zone5 (thermal)
KERNEL[65482.447894] change   /devices/virtual/thermal/thermal_zone6 (thermal)
UDEV  [65482.447922] change   /devices/virtual/thermal/thermal_zone6 (thermal)
UDEV  [65482.448467] add      /devices/pci0000:00/0000:00:16.0/0000:00:16.0-dd17041c-09ea-4b17-a271-5b989867ec65 (mei)
KERNEL[65482.448619] change   /devices/virtual/thermal/thermal_zone7 (thermal)
UDEV  [65482.449395] change   /devices/virtual/thermal/thermal_zone7 (thermal)
KERNEL[65482.449541] change   /devices/virtual/thermal/thermal_zone8 (thermal)
KERNEL[65482.450276] change   /devices/virtual/thermal/thermal_zone9 (thermal)
UDEV  [65482.450304] change   /devices/virtual/thermal/thermal_zone8 (thermal)
UDEV  [65482.450582] change   /devices/virtual/thermal/thermal_zone9 (thermal)
KERNEL[65482.452290] change   /devices/virtual/thermal/thermal_zone11 (thermal)
UDEV  [65482.452606] change   /devices/virtual/thermal/thermal_zone11 (thermal)
KERNEL[65482.548135] change   /devices/pci0000:00/0000:00:01.0/0000:01:00.0/drm/card0 (drm)
UDEV  [65482.548829] change   /devices/pci0000:00/0000:00:01.0/0000:01:00.0/drm/card0 (drm)
KERNEL[65482.711898] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1 (usb)
KERNEL[65482.712332] change   /devices/pci0000:00/0000:00:14.0/usb1/1-1 (usb)
KERNEL[65482.712377] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1:1.0 (usb)
KERNEL[65482.712922] bind     /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1:1.0 (usb)
KERNEL[65482.712971] bind     /devices/pci0000:00/0000:00:14.0/usb1/1-1 (usb)
UDEV  [65482.715314] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1 (usb)
UDEV  [65482.716447] change   /devices/pci0000:00/0000:00:14.0/usb1/1-1 (usb)
UDEV  [65482.717229] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1:1.0 (usb)
UDEV  [65482.717921] bind     /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1:1.0 (usb)
UDEV  [65482.719094] bind     /devices/pci0000:00/0000:00:14.0/usb1/1-1 (usb)
KERNEL[65483.309313] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2 (usb)
KERNEL[65483.314420] change   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2 (usb)
KERNEL[65483.314617] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.0 (usb)
KERNEL[65483.315201] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.0/0003:046D:C339.001A (hid)
KERNEL[65483.315220] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/wakeup/wakeup63 (wakeup)
KERNEL[65483.315326] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.0/0003:046D:C339.001A/input/input59 (input)
KERNEL[65483.375316] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.0/0003:046D:C339.001A/input/input59/input59::numlock (leds)
KERNEL[65483.375331] change   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.0/0003:046D:C339.001A/input/input59/input59::numlock (leds)
KERNEL[65483.375337] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.0/0003:046D:C339.001A/input/input59/input59::capslock (leds)
KERNEL[65483.375344] change   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.0/0003:046D:C339.001A/input/input59/input59::capslock (leds)
KERNEL[65483.375349] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.0/0003:046D:C339.001A/input/input59/input59::scrolllock (leds)
KERNEL[65483.375356] change   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.0/0003:046D:C339.001A/input/input59/input59::scrolllock (leds)
KERNEL[65483.375367] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.0/0003:046D:C339.001A/input/input59/event19 (input)
KERNEL[65483.375376] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.0/0003:046D:C339.001A/hidraw/hidraw0 (hidraw)
KERNEL[65483.375389] bind     /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.0/0003:046D:C339.001A (hid)
KERNEL[65483.375401] bind     /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.0 (usb)
KERNEL[65483.375412] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.1 (usb)
KERNEL[65483.376261] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.1/0003:046D:C339.001B (hid)
KERNEL[65483.376361] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.1/0003:046D:C339.001B/input/input60 (input)
KERNEL[65483.435251] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.1/0003:046D:C339.001B/input/input60/event20 (input)
KERNEL[65483.435274] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.1/0003:046D:C339.001B/input/input61 (input)
KERNEL[65483.435285] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.1/0003:046D:C339.001B/input/input61/event21 (input)
KERNEL[65483.435293] add      /class/usbmisc (class)
KERNEL[65483.435307] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.1/usbmisc/hiddev0 (usbmisc)
KERNEL[65483.435324] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.1/0003:046D:C339.001B/hidraw/hidraw1 (hidraw)
KERNEL[65483.435336] bind     /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.1/0003:046D:C339.001B (hid)
KERNEL[65483.435348] bind     /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.1 (usb)
KERNEL[65483.435361] bind     /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2 (usb)
UDEV  [65483.435672] add      /class/usbmisc (class)
KERNEL[65483.870279] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3 (usb)
KERNEL[65483.875098] change   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3 (usb)
KERNEL[65483.875495] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.0 (usb)
KERNEL[65483.876595] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.0/0003:046D:C52F.001C (hid)
KERNEL[65483.876687] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.0/0003:046D:C52F.001C/hidraw/hidraw2 (hidraw)
KERNEL[65483.935322] bind     /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.0/0003:046D:C52F.001C (hid)
KERNEL[65483.935341] bind     /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.0 (usb)
KERNEL[65483.935353] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.1 (usb)
KERNEL[65483.936549] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.1/0003:046D:C52F.001D (hid)
KERNEL[65483.936645] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.1/usbmisc/hiddev1 (usbmisc)
KERNEL[65483.936663] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.1/0003:046D:C52F.001D/hidraw/hidraw3 (hidraw)
KERNEL[65483.995523] bind     /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.1/0003:046D:C52F.001D (hid)
KERNEL[65483.995542] bind     /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.1 (usb)
KERNEL[65483.995556] bind     /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3 (usb)
KERNEL[65483.997109] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.1/0003:046D:C52F.001D/0003:046D:4057.001E (hid)
UDEV  [65483.998275] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2 (usb)
UDEV  [65483.998419] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3 (usb)
UDEV  [65483.999481] change   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3 (usb)
UDEV  [65483.999789] change   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2 (usb)
UDEV  [65484.000449] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/wakeup/wakeup63 (wakeup)
UDEV  [65484.000767] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.0 (usb)
UDEV  [65484.001059] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.1 (usb)
UDEV  [65484.001110] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.1 (usb)
UDEV  [65484.001600] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.0/0003:046D:C52F.001C (hid)
UDEV  [65484.001880] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.1/0003:046D:C52F.001D (hid)
UDEV  [65484.002268] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.1/0003:046D:C339.001B (hid)
UDEV  [65484.002286] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.1/usbmisc/hiddev1 (usbmisc)
UDEV  [65484.002375] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.1/usbmisc/hiddev0 (usbmisc)
UDEV  [65484.003197] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.0 (usb)
UDEV  [65484.004025] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.0/0003:046D:C52F.001C/hidraw/hidraw2 (hidraw)
UDEV  [65484.004084] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.1/0003:046D:C339.001B/input/input60 (input)
UDEV  [65484.004135] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.1/0003:046D:C339.001B/input/input61 (input)
UDEV  [65484.004487] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.1/0003:046D:C52F.001D/hidraw/hidraw3 (hidraw)
UDEV  [65484.004628] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.0/0003:046D:C339.001A (hid)
UDEV  [65484.004966] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.1/0003:046D:C339.001B/hidraw/hidraw1 (hidraw)
UDEV  [65484.005173] bind     /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.0/0003:046D:C52F.001C (hid)
KERNEL[65484.005333] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.1/0003:046D:C52F.001D/0003:046D:4057.001E/input/input64 (input)
KERNEL[65484.005394] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.1/0003:046D:C52F.001D/0003:046D:4057.001E/input/input64/mouse2 (input)
KERNEL[65484.005449] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.1/0003:046D:C52F.001D/0003:046D:4057.001E/input/input64/event22 (input)
UDEV  [65484.005475] bind     /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.1/0003:046D:C52F.001D (hid)
KERNEL[65484.005527] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.1/0003:046D:C52F.001D/0003:046D:4057.001E/hidraw/hidraw4 (hidraw)
KERNEL[65484.005548] bind     /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.1/0003:046D:C52F.001D/0003:046D:4057.001E (hid)
UDEV  [65484.006319] bind     /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.0 (usb)
UDEV  [65484.006382] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.0/0003:046D:C339.001A/input/input59 (input)
UDEV  [65484.006693] bind     /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.1 (usb)
UDEV  [65484.006879] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.0/0003:046D:C339.001A/hidraw/hidraw0 (hidraw)
UDEV  [65484.008145] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.0/0003:046D:C339.001A/input/input59/input59::numlock (leds)
UDEV  [65484.008172] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.0/0003:046D:C339.001A/input/input59/input59::capslock (leds)
UDEV  [65484.008214] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.0/0003:046D:C339.001A/input/input59/input59::scrolllock (leds)
UDEV  [65484.009175] change   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.0/0003:046D:C339.001A/input/input59/input59::numlock (leds)
UDEV  [65484.009204] change   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.0/0003:046D:C339.001A/input/input59/input59::capslock (leds)
UDEV  [65484.009257] change   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.0/0003:046D:C339.001A/input/input59/input59::scrolllock (leds)
UDEV  [65484.010025] bind     /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3 (usb)
UDEV  [65484.010802] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.1/0003:046D:C52F.001D/0003:046D:4057.001E (hid)
UDEV  [65484.012128] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.1/0003:046D:C52F.001D/0003:046D:4057.001E/input/input64 (input)
UDEV  [65484.013325] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.1/0003:046D:C52F.001D/0003:046D:4057.001E/hidraw/hidraw4 (hidraw)
UDEV  [65484.013724] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.1/0003:046D:C52F.001D/0003:046D:4057.001E/input/input64/mouse2 (input)
UDEV  [65484.341586] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.1/0003:046D:C339.001B/input/input60/event20 (input)
UDEV  [65484.341647] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.1/0003:046D:C339.001B/input/input61/event21 (input)
UDEV  [65484.341816] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.1/0003:046D:C52F.001D/0003:046D:4057.001E/input/input64/event22 (input)
UDEV  [65484.342324] bind     /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.1/0003:046D:C339.001B (hid)
UDEV  [65484.342479] bind     /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.1/0003:046D:C52F.001D/0003:046D:4057.001E (hid)
UDEV  [65484.343023] bind     /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.1 (usb)
UDEV  [65484.361394] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.0/0003:046D:C339.001A/input/input59/event19 (input)
UDEV  [65484.361903] bind     /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.0/0003:046D:C339.001A (hid)
UDEV  [65484.362612] bind     /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2/1-1.2:1.0 (usb)
UDEV  [65484.364978] bind     /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.2 (usb)
KERNEL[65485.754110] change   /devices/pci0000:00/0000:00:01.0/0000:01:00.0/drm/card0 (drm)
UDEV  [65485.757174] change   /devices/pci0000:00/0000:00:01.0/0000:01:00.0/drm/card0 (drm)
KERNEL[65485.989520] change   /devices/pci0000:00/0000:00:01.0/0000:01:00.0/drm/card0 (drm)
UDEV  [65485.990176] change   /devices/pci0000:00/0000:00:01.0/0000:01:00.0/drm/card0 (drm)
KERNEL[65488.891066] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.1/0003:046D:C52F.001D/0003:046D:4057.001E/power_supply/hidpp_battery_5/wakeup64 (wakeup)
KERNEL[65488.891090] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.1/0003:046D:C52F.001D/0003:046D:4057.001E/power_supply/hidpp_battery_5/hwmon11 (hwmon)
UDEV  [65488.892108] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.1/0003:046D:C52F.001D/0003:046D:4057.001E/power_supply/hidpp_battery_5/wakeup64 (wakeup)
UDEV  [65488.892129] add      /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.1/0003:046D:C52F.001D/0003:046D:4057.001E/power_supply/hidpp_battery_5/hwmon11 (hwmon)
KERNEL[65488.907437] change   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.1/0003:046D:C52F.001D/0003:046D:4057.001E/power_supply/hidpp_battery_5 (power_supply)
UDEV  [65488.913843] change   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.1/0003:046D:C52F.001D/0003:046D:4057.001E/power_supply/hidpp_battery_5 (power_supply)
KERNEL[65488.923048] change   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.1/0003:046D:C52F.001D/0003:046D:4057.001E/power_supply/hidpp_battery_5 (power_supply)
UDEV  [65488.923920] change   /devices/pci0000:00/0000:00:14.0/usb1/1-1/1-1.3/1-1.3:1.1/0003:046D:C52F.001D/0003:046D:4057.001E/power_supply/hidpp_battery_5 (power_supply)

```

```asm
➜  GitNote git:(master) ✗ sudo udevadm info -a -n /dev/sda 

Udevadm info starts with the device specified by the devpath and then
walks up the chain of parent devices. It prints for every device
found, all possible attributes in the udev rules key format.
A rule to match, can be composed by the attributes of the device
and the attributes from one single parent device.

  looking at device '/devices/pci0000:00/0000:00:14.0/usb2/2-6/2-6:1.0/host0/target0:0:0/0:0:0:0/block/sda':
    KERNEL=="sda"
    SUBSYSTEM=="block"
    DRIVER==""
    ATTR{alignment_offset}=="0"
    ATTR{capability}=="51"
    ATTR{discard_alignment}=="0"
    ATTR{events}=="media_change"
    ATTR{events_async}==""
    ATTR{events_poll_msecs}=="-1"
    ATTR{ext_range}=="256"
    ATTR{hidden}=="0"
    ATTR{inflight}=="       0        0"
    ATTR{integrity/device_is_integrity_capable}=="0"
    ATTR{integrity/format}=="none"
    ATTR{integrity/protection_interval_bytes}=="0"
    ATTR{integrity/read_verify}=="0"
    ATTR{integrity/tag_size}=="0"
    ATTR{integrity/write_generate}=="0"
    ATTR{mq/0/cpu_list}=="0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11"
    ATTR{mq/0/nr_reserved_tags}=="0"
    ATTR{mq/0/nr_tags}=="1"
    ATTR{power/async}=="disabled"
    ATTR{power/control}=="auto"
    ATTR{power/runtime_active_kids}=="0"
    ATTR{power/runtime_active_time}=="0"
    ATTR{power/runtime_enabled}=="disabled"
    ATTR{power/runtime_status}=="unsupported"
    ATTR{power/runtime_suspended_time}=="0"
    ATTR{power/runtime_usage}=="0"
    ATTR{queue/add_random}=="1"
    ATTR{queue/chunk_sectors}=="0"
    ATTR{queue/dax}=="0"
    ATTR{queue/discard_granularity}=="0"
    ATTR{queue/discard_max_bytes}=="0"
    ATTR{queue/discard_max_hw_bytes}=="0"
    ATTR{queue/discard_zeroes_data}=="0"
    ATTR{queue/fua}=="0"
    ATTR{queue/hw_sector_size}=="512"
    ATTR{queue/io_poll}=="0"
    ATTR{queue/io_poll_delay}=="-1"
    ATTR{queue/io_timeout}=="30000"
    ATTR{queue/iosched/fifo_batch}=="16"
    ATTR{queue/iosched/front_merges}=="1"
    ATTR{queue/iosched/read_expire}=="500"
    ATTR{queue/iosched/write_expire}=="5000"
    ATTR{queue/iosched/writes_starved}=="2"
    ATTR{queue/iostats}=="1"
    ATTR{queue/logical_block_size}=="512"
    ATTR{queue/max_discard_segments}=="1"
    ATTR{queue/max_hw_sectors_kb}=="32"
    ATTR{queue/max_integrity_segments}=="0"
    ATTR{queue/max_sectors_kb}=="32"
    ATTR{queue/max_segment_size}=="65536"
    ATTR{queue/max_segments}=="2048"
    ATTR{queue/minimum_io_size}=="512"
    ATTR{queue/nomerges}=="0"
    ATTR{queue/nr_requests}=="2"
    ATTR{queue/nr_zones}=="0"
    ATTR{queue/optimal_io_size}=="0"
    ATTR{queue/physical_block_size}=="512"
    ATTR{queue/read_ahead_kb}=="128"
    ATTR{queue/rotational}=="1"
    ATTR{queue/rq_affinity}=="1"
    ATTR{queue/scheduler}=="[mq-deadline] none"
    ATTR{queue/stable_writes}=="0"
    ATTR{queue/wbt_lat_usec}=="75000"
    ATTR{queue/write_cache}=="write back"
    ATTR{queue/write_same_max_bytes}=="0"
    ATTR{queue/write_zeroes_max_bytes}=="0"
    ATTR{queue/zone_append_max_bytes}=="0"
    ATTR{queue/zoned}=="none"
    ATTR{range}=="16"
    ATTR{removable}=="1"
    ATTR{ro}=="0"
    ATTR{size}=="120832000"
    ATTR{stat}=="     402       55    14144      807       12        0       16      743        0      984     1558        0        0        0        0       12        7"
    ATTR{trace/act_mask}=="disabled"
    ATTR{trace/enable}=="0"
    ATTR{trace/end_lba}=="disabled"
    ATTR{trace/pid}=="disabled"
    ATTR{trace/start_lba}=="disabled"

  looking at parent device '/devices/pci0000:00/0000:00:14.0/usb2/2-6/2-6:1.0/host0/target0:0:0/0:0:0:0':
    KERNELS=="0:0:0:0"
    SUBSYSTEMS=="scsi"
    DRIVERS=="sd"
    ATTRS{blacklist}==""
    ATTRS{device_blocked}=="0"
    ATTRS{device_busy}=="1"
    ATTRS{dh_state}=="detached"
    ATTRS{eh_timeout}=="10"
    ATTRS{evt_capacity_change_reported}=="0"
    ATTRS{evt_inquiry_change_reported}=="0"
    ATTRS{evt_lun_change_reported}=="0"
    ATTRS{evt_media_change}=="0"
    ATTRS{evt_mode_parameter_change_reported}=="0"
    ATTRS{evt_soft_threshold_reached}=="0"
    ATTRS{inquiry}==""
    ATTRS{iocounterbits}=="32"
    ATTRS{iodone_cnt}=="0x1eb"
    ATTRS{ioerr_cnt}=="0x1"
    ATTRS{iorequest_cnt}=="0x1ec"
    ATTRS{max_sectors}=="64"
    ATTRS{model}=="U358            "
    ATTRS{power/async}=="enabled"
    ATTRS{power/autosuspend_delay_ms}=="-1"
    ATTRS{power/control}=="on"
    ATTRS{power/runtime_active_kids}=="0"
    ATTRS{power/runtime_active_time}=="1133"
    ATTRS{power/runtime_enabled}=="forbidden"
    ATTRS{power/runtime_status}=="active"
    ATTRS{power/runtime_suspended_time}=="0"
    ATTRS{power/runtime_usage}=="2"
    ATTRS{queue_depth}=="1"
    ATTRS{queue_type}=="none"
    ATTRS{rev}=="1100"
    ATTRS{scsi_level}=="7"
    ATTRS{state}=="running"
    ATTRS{timeout}=="30"
    ATTRS{type}=="0"
    ATTRS{vendor}=="aigo    "

  looking at parent device '/devices/pci0000:00/0000:00:14.0/usb2/2-6/2-6:1.0/host0/target0:0:0':
    KERNELS=="target0:0:0"
    SUBSYSTEMS=="scsi"
    DRIVERS==""
    ATTRS{power/async}=="enabled"
    ATTRS{power/control}=="auto"
    ATTRS{power/runtime_active_kids}=="1"
    ATTRS{power/runtime_active_time}=="1133"
    ATTRS{power/runtime_enabled}=="enabled"
    ATTRS{power/runtime_status}=="active"
    ATTRS{power/runtime_suspended_time}=="0"
    ATTRS{power/runtime_usage}=="0"

  looking at parent device '/devices/pci0000:00/0000:00:14.0/usb2/2-6/2-6:1.0/host0':
    KERNELS=="host0"
    SUBSYSTEMS=="scsi"
    DRIVERS==""
    ATTRS{power/async}=="enabled"
    ATTRS{power/control}=="auto"
    ATTRS{power/runtime_active_kids}=="1"
    ATTRS{power/runtime_active_time}=="1185"
    ATTRS{power/runtime_enabled}=="enabled"
    ATTRS{power/runtime_status}=="active"
    ATTRS{power/runtime_suspended_time}=="1023"
    ATTRS{power/runtime_usage}=="0"

  looking at parent device '/devices/pci0000:00/0000:00:14.0/usb2/2-6/2-6:1.0':
    KERNELS=="2-6:1.0"
    SUBSYSTEMS=="usb"
    DRIVERS=="usb-storage"
    ATTRS{authorized}=="1"
    ATTRS{bAlternateSetting}==" 0"
    ATTRS{bInterfaceClass}=="08"
    ATTRS{bInterfaceNumber}=="00"
    ATTRS{bInterfaceProtocol}=="50"
    ATTRS{bInterfaceSubClass}=="06"
    ATTRS{bNumEndpoints}=="02"
    ATTRS{power/async}=="enabled"
    ATTRS{power/runtime_active_kids}=="1"
    ATTRS{power/runtime_enabled}=="enabled"
    ATTRS{power/runtime_status}=="active"
    ATTRS{power/runtime_usage}=="0"
    ATTRS{supports_autosuspend}=="1"

  looking at parent device '/devices/pci0000:00/0000:00:14.0/usb2/2-6':
    KERNELS=="2-6"
    SUBSYSTEMS=="usb"
    DRIVERS=="usb"
    ATTRS{authorized}=="1"
    ATTRS{avoid_reset_quirk}=="0"
    ATTRS{bConfigurationValue}=="1"
    ATTRS{bDeviceClass}=="00"
    ATTRS{bDeviceProtocol}=="00"
    ATTRS{bDeviceSubClass}=="00"
    ATTRS{bMaxPacketSize0}=="9"
    ATTRS{bMaxPower}=="504mA"
    ATTRS{bNumConfigurations}=="1"
    ATTRS{bNumInterfaces}==" 1"
    ATTRS{bcdDevice}=="1100"
    ATTRS{bmAttributes}=="80"
    ATTRS{busnum}=="2"
    ATTRS{configuration}==""
    ATTRS{devnum}=="3"
    ATTRS{devpath}=="6"
    ATTRS{idProduct}=="1000"
    ATTRS{idVendor}=="090c"
    ATTRS{ltm_capable}=="no"
    ATTRS{manufacturer}=="aigo"
    ATTRS{maxchild}=="0"
    ATTRS{power/active_duration}=="2368"
    ATTRS{power/async}=="enabled"
    ATTRS{power/autosuspend}=="2"
    ATTRS{power/autosuspend_delay_ms}=="2000"
    ATTRS{power/connected_duration}=="2368"
    ATTRS{power/control}=="on"
    ATTRS{power/level}=="on"
    ATTRS{power/persist}=="1"
    ATTRS{power/runtime_active_kids}=="1"
    ATTRS{power/runtime_active_time}=="2219"
    ATTRS{power/runtime_enabled}=="forbidden"
    ATTRS{power/runtime_status}=="active"
    ATTRS{power/runtime_suspended_time}=="0"
    ATTRS{power/runtime_usage}=="1"
    ATTRS{power/usb3_hardware_lpm_u1}=="enabled"
    ATTRS{power/usb3_hardware_lpm_u2}=="enabled"
    ATTRS{product}=="MiniKing"
    ATTRS{quirks}=="0x0"
    ATTRS{removable}=="removable"
    ATTRS{rx_lanes}=="1"
    ATTRS{serial}=="AA0000000079"
    ATTRS{speed}=="5000"
    ATTRS{tx_lanes}=="1"
    ATTRS{urbnum}=="1432"
    ATTRS{version}==" 3.20"

  looking at parent device '/devices/pci0000:00/0000:00:14.0/usb2':
    KERNELS=="usb2"
    SUBSYSTEMS=="usb"
    DRIVERS=="usb"
    ATTRS{authorized}=="1"
    ATTRS{authorized_default}=="1"
    ATTRS{avoid_reset_quirk}=="0"
    ATTRS{bConfigurationValue}=="1"
    ATTRS{bDeviceClass}=="09"
    ATTRS{bDeviceProtocol}=="03"
    ATTRS{bDeviceSubClass}=="00"
    ATTRS{bMaxPacketSize0}=="9"
    ATTRS{bMaxPower}=="0mA"
    ATTRS{bNumConfigurations}=="1"
    ATTRS{bNumInterfaces}==" 1"
    ATTRS{bcdDevice}=="0510"
    ATTRS{bmAttributes}=="e0"
    ATTRS{busnum}=="2"
    ATTRS{configuration}==""
    ATTRS{devnum}=="1"
    ATTRS{devpath}=="0"
    ATTRS{idProduct}=="0003"
    ATTRS{idVendor}=="1d6b"
    ATTRS{interface_authorized_default}=="1"
    ATTRS{ltm_capable}=="yes"
    ATTRS{manufacturer}=="Linux 5.10.0-13-amd64 xhci-hcd"
    ATTRS{maxchild}=="10"
    ATTRS{power/active_duration}=="15372936"
    ATTRS{power/async}=="enabled"
    ATTRS{power/autosuspend}=="0"
    ATTRS{power/autosuspend_delay_ms}=="0"
    ATTRS{power/connected_duration}=="65971152"
    ATTRS{power/control}=="auto"
    ATTRS{power/level}=="auto"
    ATTRS{power/runtime_active_kids}=="1"
    ATTRS{power/runtime_active_time}=="15389260"
    ATTRS{power/runtime_enabled}=="enabled"
    ATTRS{power/runtime_status}=="active"
    ATTRS{power/runtime_suspended_time}=="50552256"
    ATTRS{power/runtime_usage}=="0"
    ATTRS{power/usb3_hardware_lpm_u1}=="disabled"
    ATTRS{power/usb3_hardware_lpm_u2}=="disabled"
    ATTRS{power/wakeup}=="disabled"
    ATTRS{power/wakeup_abort_count}==""
    ATTRS{power/wakeup_active}==""
    ATTRS{power/wakeup_active_count}==""
    ATTRS{power/wakeup_count}==""
    ATTRS{power/wakeup_expire_count}==""
    ATTRS{power/wakeup_last_time_ms}==""
    ATTRS{power/wakeup_max_time_ms}==""
    ATTRS{power/wakeup_total_time_ms}==""
    ATTRS{product}=="xHCI Host Controller"
    ATTRS{quirks}=="0x0"
    ATTRS{removable}=="unknown"
    ATTRS{rx_lanes}=="1"
    ATTRS{serial}=="0000:00:14.0"
    ATTRS{speed}=="10000"
    ATTRS{tx_lanes}=="1"
    ATTRS{urbnum}=="415"
    ATTRS{version}==" 3.10"

  looking at parent device '/devices/pci0000:00/0000:00:14.0':
    KERNELS=="0000:00:14.0"
    SUBSYSTEMS=="pci"
    DRIVERS=="xhci_hcd"
    ATTRS{ari_enabled}=="0"
    ATTRS{broken_parity_status}=="0"
    ATTRS{class}=="0x0c0330"
    ATTRS{consistent_dma_mask_bits}=="64"
    ATTRS{d3cold_allowed}=="1"
    ATTRS{device}=="0x06ed"
    ATTRS{dma_mask_bits}=="64"
    ATTRS{driver_override}=="(null)"
    ATTRS{enable}=="1"
    ATTRS{irq}=="147"
    ATTRS{local_cpulist}=="0-11"
    ATTRS{local_cpus}=="fff"
    ATTRS{msi_bus}=="1"
    ATTRS{msi_irqs/147}=="msi"
    ATTRS{numa_node}=="-1"
    ATTRS{power/async}=="enabled"
    ATTRS{power/control}=="on"
    ATTRS{power/runtime_active_kids}=="2"
    ATTRS{power/runtime_active_time}=="65942303"
    ATTRS{power/runtime_enabled}=="forbidden"
    ATTRS{power/runtime_status}=="active"
    ATTRS{power/runtime_suspended_time}=="0"
    ATTRS{power/runtime_usage}=="1"
    ATTRS{power/wakeup}=="enabled"
    ATTRS{power/wakeup_abort_count}=="0"
    ATTRS{power/wakeup_active}=="0"
    ATTRS{power/wakeup_active_count}=="8"
    ATTRS{power/wakeup_count}=="0"
    ATTRS{power/wakeup_expire_count}=="8"
    ATTRS{power/wakeup_last_time_ms}=="65085547"
    ATTRS{power/wakeup_max_time_ms}=="105"
    ATTRS{power/wakeup_total_time_ms}=="825"
    ATTRS{revision}=="0x00"
    ATTRS{subsystem_device}=="0x22c0"
    ATTRS{subsystem_vendor}=="0x17aa"
    ATTRS{vendor}=="0x8086"

  looking at parent device '/devices/pci0000:00':
    KERNELS=="pci0000:00"
    SUBSYSTEMS==""
    DRIVERS==""
    ATTRS{power/async}=="enabled"
    ATTRS{power/control}=="auto"
    ATTRS{power/runtime_active_kids}=="12"
    ATTRS{power/runtime_active_time}=="0"
    ATTRS{power/runtime_enabled}=="disabled"
    ATTRS{power/runtime_status}=="unsupported"
    ATTRS{power/runtime_suspended_time}=="0"
    ATTRS{power/runtime_usage}=="0"

```

```asm
➜  GitNote git:(master) ✗ sudo udevadm info -a -n /dev/dri/card0

Udevadm info starts with the device specified by the devpath and then
walks up the chain of parent devices. It prints for every device
found, all possible attributes in the udev rules key format.
A rule to match, can be composed by the attributes of the device
and the attributes from one single parent device.

  looking at device '/devices/pci0000:00/0000:00:01.0/0000:01:00.0/drm/card0':
    KERNEL=="card0"
    SUBSYSTEM=="drm"
    DRIVER==""
    ATTR{power/async}=="disabled"
    ATTR{power/control}=="auto"
    ATTR{power/runtime_active_kids}=="0"
    ATTR{power/runtime_active_time}=="0"
    ATTR{power/runtime_enabled}=="disabled"
    ATTR{power/runtime_status}=="unsupported"
    ATTR{power/runtime_suspended_time}=="0"
    ATTR{power/runtime_usage}=="0"

  looking at parent device '/devices/pci0000:00/0000:00:01.0/0000:01:00.0':
    KERNELS=="0000:01:00.0"
    SUBSYSTEMS=="pci"
    DRIVERS=="nouveau"
    ATTRS{ari_enabled}=="0"
    ATTRS{boot_vga}=="1"
    ATTRS{broken_parity_status}=="0"
    ATTRS{class}=="0x030000"
    ATTRS{consistent_dma_mask_bits}=="47"
    ATTRS{current_link_speed}=="8.0 GT/s PCIe"
    ATTRS{current_link_width}=="16"
    ATTRS{d3cold_allowed}=="1"
    ATTRS{device}=="0x1f95"
    ATTRS{dma_mask_bits}=="47"
    ATTRS{driver_override}=="(null)"
    ATTRS{enable}=="2"
    ATTRS{irq}=="161"
    ATTRS{local_cpulist}=="0-11"
    ATTRS{local_cpus}=="fff"
    ATTRS{max_link_speed}=="8.0 GT/s PCIe"
    ATTRS{max_link_width}=="16"
    ATTRS{msi_bus}=="1"
    ATTRS{msi_irqs/161}=="msi"
    ATTRS{numa_node}=="-1"
    ATTRS{power/async}=="enabled"
    ATTRS{power/autosuspend_delay_ms}=="5000"
    ATTRS{power/control}=="auto"
    ATTRS{power/runtime_active_kids}=="0"
    ATTRS{power/runtime_active_time}=="66034295"
    ATTRS{power/runtime_enabled}=="enabled"
    ATTRS{power/runtime_status}=="active"
    ATTRS{power/runtime_suspended_time}=="0"
    ATTRS{power/runtime_usage}=="2"
    ATTRS{power/wakeup}=="disabled"
    ATTRS{power/wakeup_abort_count}==""
    ATTRS{power/wakeup_active}==""
    ATTRS{power/wakeup_active_count}==""
    ATTRS{power/wakeup_count}==""
    ATTRS{power/wakeup_expire_count}==""
    ATTRS{power/wakeup_last_time_ms}==""
    ATTRS{power/wakeup_max_time_ms}==""
    ATTRS{power/wakeup_total_time_ms}==""
    ATTRS{revision}=="0xa1"
    ATTRS{subsystem_device}=="0x22c0"
    ATTRS{subsystem_vendor}=="0x17aa"
    ATTRS{vendor}=="0x10de"

  looking at parent device '/devices/pci0000:00/0000:00:01.0':
    KERNELS=="0000:00:01.0"
    SUBSYSTEMS=="pci"
    DRIVERS=="pcieport"
    ATTRS{ari_enabled}=="0"
    ATTRS{broken_parity_status}=="0"
    ATTRS{class}=="0x060400"
    ATTRS{consistent_dma_mask_bits}=="32"
    ATTRS{current_link_speed}=="8.0 GT/s PCIe"
    ATTRS{current_link_width}=="16"
    ATTRS{d3cold_allowed}=="1"
    ATTRS{device}=="0x1901"
    ATTRS{dma_mask_bits}=="32"
    ATTRS{driver_override}=="(null)"
    ATTRS{enable}=="1"
    ATTRS{irq}=="121"
    ATTRS{local_cpulist}=="0-11"
    ATTRS{local_cpus}=="fff"
    ATTRS{max_link_speed}=="8.0 GT/s PCIe"
    ATTRS{max_link_width}=="16"
    ATTRS{msi_bus}=="1"
    ATTRS{msi_irqs/121}=="msi"
    ATTRS{numa_node}=="-1"
    ATTRS{power/async}=="enabled"
    ATTRS{power/autosuspend_delay_ms}=="100"
    ATTRS{power/control}=="auto"
    ATTRS{power/runtime_active_kids}=="1"
    ATTRS{power/runtime_active_time}=="66034311"
    ATTRS{power/runtime_enabled}=="enabled"
    ATTRS{power/runtime_status}=="active"
    ATTRS{power/runtime_suspended_time}=="0"
    ATTRS{power/runtime_usage}=="0"
    ATTRS{power/wakeup}=="enabled"
    ATTRS{power/wakeup_abort_count}=="0"
    ATTRS{power/wakeup_active}=="0"
    ATTRS{power/wakeup_active_count}=="0"
    ATTRS{power/wakeup_count}=="0"
    ATTRS{power/wakeup_expire_count}=="0"
    ATTRS{power/wakeup_last_time_ms}=="0"
    ATTRS{power/wakeup_max_time_ms}=="0"
    ATTRS{power/wakeup_total_time_ms}=="0"
    ATTRS{revision}=="0x02"
    ATTRS{secondary_bus_number}=="1"
    ATTRS{subordinate_bus_number}=="1"
    ATTRS{subsystem_device}=="0x22c0"
    ATTRS{subsystem_vendor}=="0x17aa"
    ATTRS{vendor}=="0x8086"

  looking at parent device '/devices/pci0000:00':
    KERNELS=="pci0000:00"
    SUBSYSTEMS==""
    DRIVERS==""
    ATTRS{power/async}=="enabled"
    ATTRS{power/control}=="auto"
    ATTRS{power/runtime_active_kids}=="11"
    ATTRS{power/runtime_active_time}=="0"
    ATTRS{power/runtime_enabled}=="disabled"
    ATTRS{power/runtime_status}=="unsupported"
    ATTRS{power/runtime_suspended_time}=="0"
    ATTRS{power/runtime_usage}=="0"

```



Open a file called `80-local.rules` in `/etc/udev/rules.d` and enter this code:

```asm
SUBSYSTEM=="block", ACTION=="add", RUN+="/usr/local/bin/trigger.sh"
```

Save the file, unplug your test thumb drive, and reboot.

Wait, reboot on a Linux machine?

Theoretically, you can just issue `udevadm control --reload`, which should load all rules,

```asm
SUBSYSTEM=="block", ATTRS{idVendor}=="03f0", ACTION=="add", SYMLINK+="safety%n"
SUBSYSTEM=="block", ATTRS{idVendor}=="03f0", ACTION=="add", RUN+="/usr/local/bin/trigger.sh"
```

# 参考资料

1. [net.hadness.PowerProfiles](https://hadess.fedorapeople.org/power-profiles-daemon-docs/gdbus-net.hadess.PowerProfiles.html)