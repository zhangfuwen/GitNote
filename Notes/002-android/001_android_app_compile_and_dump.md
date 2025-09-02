
# Android Apk构建与拆解

## 总体流程

总体流程涉及五个工具，也即5个步骤：

```bash
javac – compile Java code
d8 – convert .class to .dex (Google's DEX compiler)
aapt2 – compile and link resources
zipalign – align the APK
apksigner – sign the APK
```


## 工程准备

```
MyApp/
├── src/
│   └── com/example/myapp/MainActivity.java
├── res/
│   └── values/
│       └── strings.xml
│   └── layout/
│       └── activity_main.xml
├── AndroidManifest.xml
└── build/
```


## Compile java code

