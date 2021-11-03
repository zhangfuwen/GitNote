我们通过repo init初始化一个工程，通过repo sync把各个project都下载下来。其实其余的工作都可以通过git完成了。比如说project的commit、push、pull。

然而还是有一些场景使用repo命令会比较方便。

# 在所有的project中启动一个新分支
``repo start newbranchname``
# 对所有project执行命令


```
repo forall -c git checkout master
repo forall -c git git reset --hard HEAD
```

