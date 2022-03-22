---

title: 用docker提供mysql数据库服务

---

# 用docker提供mysql数据库服务

最近我的ubuntu一关机，重启后mysqld就无法启动了，现在决定使用docker来管理mysql服务。希望可以稳定下来。

# 安装docker及mysql

```bash
sudo apt-get install docker.io
service docker start
sudo docker pull mysql/mysql-server
```

# 运行mysql

`docker run --expose 3306 --name my-container-name -e MYSQL_ROOT_PASSWORD=my-secret-pw -d mysql/mysql-server:tag`

my-container-name 是怎么随便取的名字

my-secret-pw 是自已设置mysql的root密码

tag是要运行的mysql版本号，可以用latest，也可以用一个真正的版本号，支持的版本列表在

https://hub.docker.com/r/mysql/mysql-server/tags/

我的命令是：

`sudo docker run --name mysql_docker -e MYSQL_ROOT_PASSWORD=123456 -d mysql/mysql-server:latest`

创建数据库：

```
sudo docker exec -it mysql_docker bash
mysql -uroot -p
create table zhangfuwen;
exit
exit
```
## 把docker的3306端口与主机的3306端口绑定

By default Docker containers can make connections to the outside world, but the outside world cannot connect to containers. Each outgoing connection will appear to originate from one of the host machine’s own IP addresses thanks to an iptables masquerading rule on the host machine that the Docker server creates when it starts:
```bash
$ sudo iptables -t nat -L -n
```
...
Chain POSTROUTING (policy ACCEPT)
target     prot opt source               destination
MASQUERADE  all  --  172.17.0.0/16       0.0.0.0/0
...
The Docker server creates a masquerade rule that let containers connect to IP addresses in the outside world.

If you want containers to accept incoming connections, you will need to provide special options when invoking docker run. There are two approaches.

First, you can supply -P or --publish-all=true|false to docker run which is a blanket operation that identifies every port with an EXPOSE line in the image’s Dockerfile or --expose <port> commandline flag and maps it to a host port somewhere within an ephemeral port range. The docker port command then needs to be used to inspect created mapping. The ephemeral port range is configured by /proc/sys/net/ipv4/ip_local_port_range kernel parameter, typically ranging from 32768 to 61000.

Mapping can be specified explicitly using -p SPEC or --publish=SPEC option. It allows you to particularize which port on docker server - which can be any port at all, not just one within the ephemeral port range – you want mapped to which port in the container.

Either way, you should be able to peek at what Docker has accomplished in your network stack by examining your NAT tables.

What your NAT rules might look like when Docker
is finished setting up a -P forward:

$ iptables -t nat -L -n

...
Chain DOCKER (2 references)
target     prot opt source               destination
DNAT       tcp  --  0.0.0.0/0            0.0.0.0/0            tcp dpt:49153 to:172.17.0.2:80

```bash
# What your NAT rules might look like when Docker
# is finished setting up a -p 80:80 forward:
```
Chain DOCKER (2 references)
target     prot opt source               destination
DNAT       tcp  --  0.0.0.0/0            0.0.0.0/0            tcp dpt:80 to:172.17.0.2:80
You can see that Docker has exposed these container ports on 0.0.0.0, the wildcard IP address that will match any possible incoming port on the host machine. If you want to be more restrictive and only allow container services to be contacted through a specific external interface on the host machine, you have two choices. When you invoke docker run you can use either -p IP:host_port:container_port or -p IP::port to specify the external interface for one particular binding.

Or if you always want Docker port forwards to bind to one specific IP address, you can edit your system-wide Docker server settings and add the option --ip=IP_ADDRESS. Remember to restart your Docker server after editing this setting.

## 保存状态


```
dean@dean-Aspire-4740:~$ sudo docker ps
CONTAINER ID        IMAGE                       COMMAND                  CREATED             STATUS              PORTS                 NAMES
406dfe63d729        mysql/mysql-server:latest   "/entrypoint.sh mysql"   11 minutes ago      Up 11 minutes       3306/tcp, 33060/tcp   mysql_docker

dean@dean-Aspire-4740:~$ sudo docker commit 406dfe63d729 mysql/mysql-server 
sha256:89860fe30160148d5305a529996fd8bda1d57c20325b0129fb0f9e5b60e0f4a8
dean@dean-Aspire-4740:~$ sudo docker ps
CONTAINER ID        IMAGE               COMMAND                  CREATED             STATUS              PORTS                 NAMES
406dfe63d729        65f6760a2eef        "/entrypoint.sh mysql"   13 minutes ago      Up 13 minutes       3306/tcp, 33060/tcp   mysql_docker

```
commit并加tag


```
dean@dean-Aspire-4740:~$ sudo docker commit 406dfe63d729 mysql/mysql-server:version1 
sha256:0ca5485937accdc4a90037427ea2ab7feb38f22d3898e6a2b823c0b1e6536cc2
```

## 关闭docker


```
dean@dean-Aspire-4740:~$ sudo docker stop mysql_docker
```

## 运行修改后的docker


```
dean@dean-Aspire-4740:~$ sudo docker run -p 3306:3306 --name mysql_docker1 -e MYSQL_ROOT_PASSWORD=123456 -d mysql/mysql-server:version1
dean@dean-Aspire-4740:~$ sudo docker stop mysql_docker1
mysql_docker1
dean@dean-Aspire-4740:~$ sudo docker start mysql_docker1
mysql_docker1
```
查看是否已有暴出这个端口：


```
sudo docker port mysql_docker1 3306
```
或


```
sudo docker inspect mysql_docker1
```

## grant access:


```
sudo docker exec -it mysql_docker1 bash
mysql -uroot -p
GRANT ALL PRIVILEGES
ON *.*
TO 'root'@'%'
IDENTIFIED BY '123456';
```


## 登入：


```bash
 mysql -uroot -p -h127.0.0.1 -P 3306 
```

## docker 使用宿主机网络

```bash
docker run -d --name nginx --network host nginx
```

上面的命令中，没有必要像前面一样使用`-p 80:80 -p 443:443`来映射端口，是因为本身与宿主机共用了网络，容器中暴露端口等同于宿主机暴露端口。

## docker使用国内源
```bash
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json <<-'EOF'
{
  "registry-mirrors": ["https://f3y3atkw.mirror.aliyuncs.com"]
}
EOF
```
其他源：

```json
{
  "registry-mirrors" : [
    "http://ovfftd6p.mirror.aliyuncs.com",
    "http://registry.docker-cn.com",
    "http://docker.mirrors.ustc.edu.cn",
    "http://hub-mirror.c.163.com"
  ],
  "insecure-registries" : [
    "registry.docker-cn.com",
    "docker.mirrors.ustc.edu.cn"
  ],
  "debug" : true,
  "experimental" : true
}
```

然后重启docker后台进程，在使用systemctl的机器上是:
```bash
 sudo systemctl daemon-reload
 sudo systemctl docker restart
```
在没有systemctl的机器上是:
```bash
sudo service docker restart
```

## docker 将容器保存为镜像



### 语法

```
docker commit [OPTIONS] CONTAINER [REPOSITORY[:TAG]]
```

OPTIONS说明：

- **-a :**提交的镜像作者；

  

- **-c :**使用Dockerfile指令来创建镜像；

  

- **-m :**提交时的说明文字；

  

- **-p :**在commit时，将容器暂停。

  docker commit -a "runoob.com" -m "my apache" a404c6c174a2  mymysql:v1 
  
  