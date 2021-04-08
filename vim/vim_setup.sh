#!/bin/bash

function download_nvim_x86
{
    echo "try install neovim"
    mkdir ~/bin/
    wget -O ~/bin/nvim-nightly.tar.gz https://github.com/neovim/neovim/releases/download/nightly/nvim-linux64.tar.gz 
    tar xvzf ~/bin/nvim-nightly.tar.gz -C ~/bin/
    echo "export PATH=~/bin/nvim-linux64/bin:$PATH" >> ~/.bashrc
    source ~/.bashrc 
    rm nvim-linux64.tar.gz  -C ~/bin/
    $VIM=nvim
}

function download_nvim_arm64
{
    echo "try install vim"
    $SUDO apt-get install -y vim || $SUDO yum install -y vim
    $VIM=vim
}

if [[ $architecture == "aarch64" ]]; then
    download_nvim_arm64
elif [[ $architecture == "x86_64" ]]; then
    download_nvim_x86
fi

echo "init nvim config file"
mkdir -p ~/.config/nvim
cat > ~/.config/nvim/init.vim << EOF
set runtimepath^=~/.vim runtimepath+=~/.vim/after
let &packpath=&runtimepath
source ~/.vimrc
EOF

# $SUDO apt install vim

echo "installing vim-plug"
curl -fLo ~/.vimrc --create-dirs https://gitee.com/zhangfuwen/GitNote/raw/master/vim/vimrc
curl -fLo ~/.vim/autoload/plug.vim --create-dirs \
    https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim

echo "install vim pluggins"
$VIM -E -c PlugInstall -c q

echo "install ctags"
$SUDO apt-get install -y ripgrep cppman ctags || $SUDO yum install -y ripgrep cppman ctags || $SUDO dnf install -y ripgrep cppman ctags


