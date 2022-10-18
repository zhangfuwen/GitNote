# 漂亮的git lg

默认的git log显示很难看，所以在一台新的电脑上做的第一件事就是美化一下。填加一个git lg命令，显示得好看一点。
![](/assets/git-lg.png)
做法：

``` bash
git config --global alias.lg "log --color --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit"
```

# 漂亮的textui tig

在gitg命令窗口中可以使用gitk或gitg命令打开一个GUI来方便地操作。命令行一个一个命令打字不适合所有场景。如果你不喜欢或是没有gui的话，也可以试试tig，vim一样的操作手感。
![](/assets/tig.png)
安装：

``` bash
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

# 修改upstream对齐

``` bash
git branch --set-upstream-to=origin/remote_branch  your_branch
```

比如从gitee切换到github：

``` bash
➜  GitNote git:(master) ✗ git branch -alvv 
* master               2a8f498 [gitee/master: 领先 61] Update _config.yml
  remotes/gh/master    2a8f498 Update _config.yml
  remotes/gitee/master 59a767d xx
➜  GitNote git:(master) ✗ git branch --set-upstream-to=gh/master master
分支 'master' 设置为跟踪来自 'gh' 的远程分支 'master'。
➜  GitNote git:(master) ✗ git branch -alvv
* master               2a8f498 [gh/master] Update _config.yml
  remotes/gh/master    2a8f498 Update _config.yml
  remotes/gitee/master 59a767d xx
```

# git branch 以类 less 的界面显示问题

git branch现在不输到的当然terminal了，而是以类似less的方式显示，一按q就没有了，有时你想对照着git branch的输出敲多个命令现在没法办到了。

解决办法：

``` bash
git config --global pager.branch false
```

具体参考：https://www.codenong.com/48341920/

# commands

## default editor

```bash
git config --global core.editor "vim"

```

## git hash

```bash
git rev-parse --short HEAD
```

## config

git config分 --system --global --local等。

--system是使用/etc/gitconfig文件的。
--global使用提~/.gitconfig文件
--local使用的是当前git仓的.git/config文件。

所以你可以进行一些本地化的配置。比如说git config --local user.email "xx"，可以让你在不同的仓使用不同的身份提交代码。

### 具体常用配置

```bash
git config --global user.name "zhangfuwen"
git config --global user.email "zhangfuwen@xxx.com"
git config --global core.editor vim
git config --global pager.branch false
git config --global alias.ll 'log --oneline'
```


# git commit message 与 issue navigation

## udacity的git message

[link](https://udacity.github.io/git-styleguide/)

overall picture:

```tip
type: Subject

body

footer
```

footer是元数据，用于指向issue id等。

### type

`feat`: A new feature
`fix`: A bug fix
`docs`: Changes to documentation
`style`: Formatting, missing semi colons, etc; no code change
`refactor`: Refactoring production code
`test`: Adding tests, refactoring test; no production code change
`chore`: Updating build tasks, package manager configs, etc; no production code change

### example

```note
feat: Summarize changes in around 50 characters or less

More detailed explanatory text, if necessary. Wrap it to about 72
characters or so. In some contexts, the first line is treated as the
subject of the commit and the rest of the text as the body. The
blank line separating the summary from the body is critical (unless
you omit the body entirely); various tools like `log`, `shortlog`
and `rebase` can get confused if you run the two together.

Explain the problem that this commit is solving. Focus on why you
are making this change as opposed to how (the code explains that).
Are there side effects or other unintuitive consequences of this
change? Here's the place to explain them.

Further paragraphs come after blank lines.

 - Bullet points are okay, too

 - Typically a hyphen or asterisk is used for the bullet, preceded
   by a single space, with blank lines in between, but conventions
   vary here

If you use an issue tracker, put references to them at the bottom,
like this:

Resolves: #123
See also: #456, #789
```

# git alias

```bash
git config --global alias.st 'status -sb'
git config --global alias.ll 'log --oneline'
git config --global alias.last 'log -1 HEAD --stat'
git config --global alias.se '!git rev-list --all | xargs git grep -F'  # search commit
```

# git submodule

```bash
cd thirdparty
git submodule add https://xxx
git submodule init
git submodule update
```

