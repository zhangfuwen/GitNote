# nginx
如下配置可以让nginx响应静态文件请求。

```conf
    server {
        root /www/data;

        location / {
        }
    
        location /images/ {
        }
    
        location ~ \.(mp3|mp4) {
            root /www/media;
        }
    }
```

要求是通过root指向根文件夹，通过location指定url上的哪些文件夹是可以方问静态文件的。
