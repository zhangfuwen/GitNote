---

title: vcpkg 使用

---

# 安装vcpkg

```bash
git clone https://github.com/Microsoft/vcpkg.git
./vcpkg/bootstrap-vcpkg.sh
```

# 安装包

```bash
./vcpkg install cpp-httplib
```

# 创建包

```bash
vcpkg create handycpp https://github.com/zhangfuwen/handycpp
vcpkg edit handycpp # and then edit
vcpkg install handycpp
```
