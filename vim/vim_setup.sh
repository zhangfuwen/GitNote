#!/bin/bash

$SUDO apt install vim
curl -fLo ~/.vimrc --create-dirs https://gitee.com/zhangfuwen/GitNote/raw/master/vim/vimrc
curl -fLo ~/.vim/autoload/plug.vim --create-dirs \
    https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim

vim -E -c PlugInstall -c q

$SUDO apt-get install -y ripgrep cppman ctags || $SUDO yum install -y ripgrep cppman ctags || $SUDO dnf install -y ripgrep cppman ctags
