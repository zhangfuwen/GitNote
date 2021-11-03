# 漂亮的git lg

默认的git log显示很难看，所以在一台新的电脑上做的第一件事就是美化一下。填加一个git lg命令，显示得好看一点。  
![](/assets/git-lg.png)  
做法：

```bash
git config --global alias.lg "log --color --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit"
```

# 漂亮的textui tig

在gitg命令窗口中可以使用gitk或gitg命令打开一个GUI来方便地操作。命令行一个一个命令打字不适合所有场景。如果你不喜欢或是没有gui的话，也可以试试tig，vim一样的操作手感。  
![](/assets/tig.png)  
安装：

```bash
sudo apt install tig
```

tig使用很简单，但未必浅显，后面我可以搞成专题。

# 常用的git命令

```
1.到本地仓库 cd
2.查看状态：git status
3.添加文件：git  add .
4.提交 git commit -m”备注”
5.查看日志：git log
6.查看当前分支：git branch
7.拉取最新代码：git pull origin 分支名
8.推送代码：git push origin 分支名
9.删除远程分支：git push origin :分支名 
10.新建分支，并切换到新建的分支:git checkout -b 新分支名
11.将新建的分支推送到服务器：git push origin 新建的分支名
12.删除本地分支：git branch -D 分支名
13.合并某个分支到当前分支：git merge 需要合并到当前分支的分支名
14.强制回撤到某次提交的版本：git reset —hard 版本号的前6位(如：abe75e)
15.添加tag：git tag -a “标签名” -m”备注”
16.将添加的标签推送到远程服务器：git push —tag
17.进入到某哥tag:git checkout 标签名
18.强制回撤到某个标签：git reset —hard 标签名
19.删除本地tag：git tag -d 标签名
20.删除远程的tag：git push origin -–delete tag 标签名
21.删除git而不删除文件：find . -name “.git | xargs rm -Rf
22.查看git远程仓库地址：git remote -v
23.移除远程的git地址：git remote rm origin
24.将本地修改强制推送到服务器  git push -f -u origin master1.到本地仓库 cd
2.查看状态：git status
3.添加文件：git  add .
4.提交 git commit -m”备注”
5.查看日志：git log
6.查看当前分支：git branch
7.拉取最新代码：git pull origin 分支名
8.推送代码：git push origin 分支名
9.删除远程分支：git push origin :分支名 
10.新建分支，并切换到新建的分支:git checkout -b 新分支名
11.将新建的分支推送到服务器：git push origin 新建的分支名
12.删除本地分支：git branch -D 分支名
13.合并某个分支到当前分支：git merge 需要合并到当前分支的分支名
14.强制回撤到某次提交的版本：git reset —hard 版本号的前6位(如：abe75e)
15.添加tag：git tag -a “标签名” -m”备注”
16.将添加的标签推送到远程服务器：git push —tag
17.进入到某哥tag:git checkout 标签名
18.强制回撤到某个标签：git reset —hard 标签名
19.删除本地tag：git tag -d 标签名
20.删除远程的tag：git push origin -–delete tag 标签名
21.删除git而不删除文件：find . -name “.git | xargs rm -Rf
22.查看git远程仓库地址：git remote -v
23.移除远程的git地址：git remote rm origin
24.将本地修改强制推送到服务器  git push -f -u origin master
```

<div id="gitalk-container"></div>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/gitalk@1/dist/gitalk.css">
<script src="https://cdn.jsdelivr.net/npm/gitalk@1/dist/gitalk.min.js"></script>
<script >
  const gitalk = new Gitalk({
    clientID: '{{ site.gittalk.clientID }}',
    clientSecret: "{{ site.gittalk.clientSecret }}",
    repo: "{{ site.gittalk.repository }}",
    owner: "{{ site.gittalk.owner }}",
    admin: ["{{ site.gittalk.admin | split: ','  | join: '","'}}"],
    id: "{{ page.url }}",    // Ensure uniqueness and length less than 50 
    distractionFreeMode: false,  // Facebook-like distraction free mode
    perPage: 100
});

gitalk.render('gitalk-container')