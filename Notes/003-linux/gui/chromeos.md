---

title: chromeos

---

# build

## 安装依赖

```bash
sudo apt -y install binutils bison bzip2 cdbs curl dbus-x11 dpkg-dev elfutils devscripts fakeroot flex git-core gperf libasound2-dev libatspi2.0-dev libbrlapi-dev libbz2-dev libcairo2-dev libcap-dev libc6-dev libcups2-dev libcurl4-gnutls-dev libdrm-dev libelf-dev libevdev-dev libffi-dev libgbm-dev libglib2.0-dev libglu1-mesa-dev libgtk-3-dev libkrb5-dev libnspr4-dev libnss3-dev libpam0g-dev libpci-dev libpulse-dev libsctp-dev libspeechd-dev libsqlite3-dev libssl-dev libudev-dev libva-dev libwww-perl libxshmfence-dev libxslt1-dev libxss-dev libxt-dev libxtst-dev locales openbox p7zip patch perl pkg-config rpm ruby subversion uuid-dev wdiff x11-utils xcompmgr xz-utils zip libbluetooth-dev libxkbcommon-dev mesa-common-dev zstd
```

## 安装 depot_tools

```bash
git clone https://chromium.googlesource.com/chromium/tools/depot_tools.git
export PATH="$PATH:${HOME}/ChromeOS/depot_tools"
```

## 下载安装gn

```bash
git clone https://gn.googlesource.com/gn
cd gn
python3 build/gen.py
ninja -C out 
sudo cp ./gn /usr/bin
```

## 下载chromium源码

```bash
git clone https://chromium.googlesource.com/chromium
```

## 构建chrome

```bash
cd chromium
fetch --no-history --nohooks chromium
gclient runhooks

echo "target_os = ['chromeos']" >> .gclient

cd src
gn gen out/Default --args='target_os="chromeos"'
autoninja -C out/Default chrome
```

编译过程大约需要 2 小时

## 运行chrome os

```bash
./out/Default/chrome --enable-wayland-server --wayland-server-socket=wayland-exo
```

## release build

```bash
gn gen out_amd64-generic/Release
autoninja -C out/Release chrome
```

## run chrome os non-windowed mode


