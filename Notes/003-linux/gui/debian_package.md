---

title: debian 打包

---


# change log

要用debchange来生成changelog

```bash
sudo apt install devscripts
```

```bash
debchange "initial commit"
```


# 反向查找依赖该包的其他包

```bash
if dpkg -s apt-rdepends > /dev/null 2>&1; then
  echo  "apt-rdepends already exists"
else
  sudo apt install apt-rdepends
fi

apt-rdepends -r qtcreator 
```


# 查看当前包依赖的包

```bash
dpkg -s qtcreator
```

# 列出已安装的包

```bash
apt --installed list
```


