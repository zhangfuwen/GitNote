#!/bin/bash
script_dir=$(basename $0)
source $script_dir/env_setup.sh

VIM=vim
function download_nvim_x86()
{
    echo "try install neovim"
    if [[ ! $(command -v nvim) ]];then
        mkdir ~/bin/
        wget -O ~/bin/nvim-nightly.tar.gz https://github.com/neovim/neovim/releases/download/nightly/nvim-linux64.tar.gz 
        tar xvzf ~/bin/nvim-nightly.tar.gz -C ~/bin/
        echo "export PATH=~/bin/nvim-linux64/bin:$PATH" >> ~/.bashrc
        source ~/.bashrc 
        rm nvim-linux64.tar.gz  -C ~/bin/
    fi
    echo "nvim installed"
    VIM=nvim
}

function download_nvim_arm64()
{
    echo "try install vim"
    if [[ ! $(command -v vim) ]]; then
        $SUDO $pkgman install -y vim
    fi
    echo "vim installed"
    VIM=vim
}

if [[ "$architecture" == "aarch64" ]]; then
    download_nvim_arm64
elif [[ "$architecture" == "x86_64" ]]; then
    download_nvim_x86
fi

echo "init nvim config file"
mkdir -p ~/.config/nvim
if [[ ! -f ~/.config/nvim/init.vim ]] || $(user_confirm "~/.config/nvim/init.vim exists, do you want to update?"); then
    cat > ~/.config/nvim/init.vim << EOF
" updated by GitNote
set runtimepath^=~/.vim runtimepath+=~/.vim/after
let &packpath=&runtimepath
source ~/.vimrc
EOF
fi

# $SUDO apt install vim

echo "installing vim-plug"
if [[ ! -f ~/.vimrc ]] || $(user_confirm "~/.vimrc exists, do you want us to append new contents to it?") ; then
    set -x
    curl -fLo ~/.vimrc --create-dirs https://gitee.com/zhangfuwen/GitNote/raw/master/vim/vimrc
    set +x
fi

curl -fLo ~/.vim/autoload/plug.vim --create-dirs \
    https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim

echo "install vim pluggins"
$VIM +PlugInstall +qall 

echo "install ctags"
$SUDO $pkgman install -y ripgrep cppman ctags


