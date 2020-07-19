## go 1.13 使用goproxy.cn访问github
```bash
go env -w GOPROXY=https://goproxy.cn,direct
```

## 较旧版本使用goproxy.cn

使用方法：

```bash 
export GOPROXY=https://goproxy.cn
```

或

```bash
export GO111MODULE=on
export GOPROXY=https://goproxy.cn
```

或直接写入：～/.bash_profile


## buffalo 安装和生成项目

install node and npm first:
```bash
sudo apt-get install node npm
```

```bash
wget https://github.com/gobuffalo/buffalo/releases/download/v0.15.5/buffalo_0.15.5_Linux_x86_64.tar.gz

## untar and put it under $PATH

buffalo new coke --db-type mysql
vi database.yaml
buffalo pop create # create db base
ADDR=0.0.0.0 buffalo dev
```
