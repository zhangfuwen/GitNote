---

title: gnome-sushi customizattion

---

```bash

cd ~/Code
mkdir gnome-sushi
cd gnome-sushi
sudo apt source gnome-sushi
sudo chown -R zhangfuwen:zhangfuwen ./*
cd gnome-sushi
# modify javascript source code
meson --prefix /usr build
cd build 
ninja
sudo ninja install

```
