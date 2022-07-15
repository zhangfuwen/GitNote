---

title: linux声音问题

---

# 从下往上的视角

## 内核有没有识别到设备、是否加载了内存态驱动

```bash
lspci -v | grep -A7 -i "audio"
```

```commandline
00:1f.3 Multimedia audio controller: Intel Corporation Comet Lake PCH cAVS
Subsystem: Lenovo Comet Lake PCH cAVS
Flags: bus master, fast devsel, latency 64, IRQ 177, IOMMU group 11
Memory at 4040118000 (64-bit, non-prefetchable) [size=16K]
Memory at 4040000000 (64-bit, non-prefetchable) [size=1M]
Capabilities: <access denied>
    Kernel driver in use: sof-audio-pci
    Kernel modules: snd_hda_intel, snd_sof_pci

    00:1f.4 SMBus: Intel Corporation Comet Lake PCH SMBus Controller
    Subsystem: Lenovo Comet Lake PCH SMBus Controller
    Flags: medium devsel, IRQ 16, IOMMU group 11
    Memory at 4040122000 (64-bit, non-prefetchable) [size=256]
    I/O ports at efa0 [size=32]
    --
    01:00.1 Audio device: NVIDIA Corporation Device 10fa (rev a1)
    Subsystem: Lenovo Device 22c0
    Flags: bus master, fast devsel, latency 0, IRQ 17, IOMMU group 1
    Memory at ee000000 (32-bit, non-prefetchable) [size=16K]
    Capabilities: <access denied>
    Kernel driver in use: snd_hda_intel
    Kernel modules: snd_hda_intel
```

可以看到Kernel modules字段，已经加载了内核态驱动。

## 内核是否识别到相应设备

```bash
cat /proc/asound/cards
```

```commandline
 0 [NVidia         ]: HDA-Intel - HDA NVidia
                      HDA NVidia at 0xee000000 irq 17
 1 [sofhdadsp      ]: sof-hda-dsp - sof-hda-dsp
                      sof-hda-dsp
```

```bash
cat /proc/asound/devices 
```

```commandline
  2: [ 0- 3]: digital audio playback
  3: [ 0- 7]: digital audio playback
  4: [ 0- 8]: digital audio playback
  5: [ 0- 9]: digital audio playback
  6: [ 0-10]: digital audio playback
  7: [ 0-11]: digital audio playback
  8: [ 0- 0]: hardware dependent
  9: [ 0]   : control
 10: [ 1- 6]: digital audio capture
 11: [ 1- 7]: digital audio capture
 12: [ 1- 0]: digital audio playback
 13: [ 1- 0]: digital audio capture
 14: [ 1- 1]: digital audio playback
 15: [ 1- 1]: digital audio capture
 16: [ 1- 3]: digital audio playback
 17: [ 1- 4]: digital audio playback
 18: [ 1- 5]: digital audio playback
 19: [ 1- 0]: hardware dependent
 20: [ 1]   : control
 33:        : timer
```

```bash
cat /proc/asound/card0/pcm3p/info # 这是外接显示器上的音频口 
```

```commandline
card: 0
device: 3
subdevice: 0
stream: PLAYBACK
id: HDMI 0
name: HDMI 0
subname: subdevice #0
class: 0
subclass: 0
subdevices_count: 1
subdevices_avail: 1

```

## alsa 用户态能否识别到设备

```bash
sudo aplay -l
```
```commandline
**** List of PLAYBACK Hardware Devices ****
card 0: NVidia [HDA NVidia], device 3: HDMI 0 [HDMI 0]
  Subdevices: 1/1
  Subdevice #0: subdevice #0
card 0: NVidia [HDA NVidia], device 7: HDMI 1 [HDMI 1]
  Subdevices: 1/1
  Subdevice #0: subdevice #0
card 0: NVidia [HDA NVidia], device 8: HDMI 2 [HDMI 2]
  Subdevices: 1/1
  Subdevice #0: subdevice #0
card 0: NVidia [HDA NVidia], device 9: HDMI 3 [HDMI 3]
  Subdevices: 1/1
  Subdevice #0: subdevice #0
card 0: NVidia [HDA NVidia], device 10: HDMI 4 [HDMI 4]
  Subdevices: 1/1
  Subdevice #0: subdevice #0
card 0: NVidia [HDA NVidia], device 11: HDMI 5 [HDMI 5]
  Subdevices: 1/1
  Subdevice #0: subdevice #0
card 1: sofhdadsp [sof-hda-dsp], device 0: HDA Analog (*) []
  Subdevices: 1/1
  Subdevice #0: subdevice #0
card 1: sofhdadsp [sof-hda-dsp], device 1: HDA Digital (*) []
  Subdevices: 1/1
  Subdevice #0: subdevice #0
card 1: sofhdadsp [sof-hda-dsp], device 3: HDMI1 (*) []
  Subdevices: 1/1
  Subdevice #0: subdevice #0
card 1: sofhdadsp [sof-hda-dsp], device 4: HDMI2 (*) []
  Subdevices: 1/1
  Subdevice #0: subdevice #0
card 1: sofhdadsp [sof-hda-dsp], device 5: HDMI3 (*) []
  Subdevices: 1/1
  Subdevice #0: subdevice #0
```

## 测试alsa能否跑音频

```bash
sudo aplay /usr/share/sounds/alsa/Front_Center.wav
```

## 测试pulsa audio

```bash
pacmd

list-sources
```