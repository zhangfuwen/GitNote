#!/bin/bash

curl -fLo ~/.vimrc --create-dirs https://gitee.com/zhangfuwen/GitNote/raw/master/vim/vimrc
curl -fLo ~/.vim/autoload/plug.vim --create-dirs \
    https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim

vim -E -c PlugInstall -c q

sudo apt-get install -y ripgrep cppman ctags || sudo yum install -y ripgrep cppman ctags || sudo dnf install -y ripgrep cppman ctags