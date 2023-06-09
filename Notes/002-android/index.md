# Android


## 安装sdkmanager和avdmanager

linux:
```bash
export ANDROID_CMDLINE_VERSION=9.0

sudo apt-get install google-android-cmdline-tools-${ANDROID_CMDLINE_VERSION}-installer

export ANDROID_BUILD_TOOL_VERSION=33.0.2
sdkmanager --install "tools" \
  "platform-tools" \
  "build-tools;${ANDROID_BUILD_TOOL_VERSION}" \
  "ndk-bundle" \
  "platforms;andnroid-33"

export ANDROID_SDK=/usr/lib/android-sdk
export ANDROID_SDK_HOME=ANDROID_SDK
export ANDROID_NDK=/usr/lib/android-sdk/ndk-bundle
export ANDROID_NDK_HOME=ANDROID_NDK
export PATH=$ANDROID_SDK/cmdline-tools/$ANDROID_CMDLINE_VERSION/bin:\
          $ANDROID_SDK/build-tools/$ANDROID_BUILD_TOOL_VERSION:\
          $ANDROID_SDK/tools/:\
          $ANDROID_SDK/platform-tools/:\
          $PATH
```

```bash

echo "export ANDROID_CMDLINE_VERSION=9.0" >> ~/.profile
echo "export ANDROID_BUILD_TOOL_VERSION=33.0.2" >> ~/.profile
echo "export ANDROID_SDK=${ANDROID_SDK}" >> ~/.profile
echo "export ANDROID_SDK_HOME=${ANDROID_SDK_HOME}" >> ~/.profile
echo "export ANDROID_NDK=${ANDROID_NDK}" >> ~/.profile
echo "export ANDROID_NDK_HOME=${ANDROID_NDK_HOME}" >> ~/.profile
echo 'export PATH=$ANDROID_SDK/cmdline-tools/$ANDROID_CMDLINE_VERSION/bin:\
          $ANDROID_SDK/build-tools/$ANDROID_BUILD_TOOL_VERSION:\
          $ANDROID_SDK/tools/:\
          $ANDROID_SDK/platform-tools/:\
          $PATH' >> ~/.profile

```

mac os:
```bash
brew install android-sdk
```


```bash
sdkmanager --install "platforms;android-33" \
  "system-images;android-33;google_apis_playstore;arm64-v8a" \
  platform-tools \
  ndk-bundle
  
export ANDROID_SDK=/usr/local/share/android-sdk
export ANDROID_SDK_HOME=ANDROID_SDK
export ANDROID_NDK=/usr/local/share/android-sdk/ndk-bundle
export ANDROID_NDK_HOME=ANDROID_NDK
export PATH=$ANDROID_SDK/cmdline-tools/$ANDROID_CMDLINE_VERSION/bin:\
          $ANDROID_SDK/build-tools/$ANDROID_BUILD_TOOL_VERSION:\
          $ANDROID_SDK/tools/:\
          $ANDROID_SDK/platform-tools/:\
          $PATH

avdmanager create avd --name android13-arm64 -d 18 -k 'system-images;android-33;google_apis_playstore;arm64-v8a'
/usr/local/share/android-sdk/emulator/emulator @android13-arm64

```

```bash

echo "export ANDROID_CMDLINE_VERSION=9.0" >> ~/.profile
echo "export ANDROID_BUILD_TOOL_VERSION=33.0.2" >> ~/.profile
echo "export ANDROID_SDK=${ANDROID_SDK}" >> ~/.profile
echo "export ANDROID_SDK_HOME=${ANDROID_SDK_HOME}" >> ~/.profile
echo "export ANDROID_NDK=${ANDROID_NDK}" >> ~/.profile
echo "export ANDROID_NDK_HOME=${ANDROID_NDK_HOME}" >> ~/.profile
echo 'export PATH=$ANDROID_SDK/cmdline-tools/$ANDROID_CMDLINE_VERSION/bin:\
          $ANDROID_SDK/build-tools/$ANDROID_BUILD_TOOL_VERSION:\
          $ANDROID_SDK/tools/:\
          $ANDROID_SDK/platform-tools/:\
          $PATH' >> ~/.profile

```