#!/bin/bash

SUDO=""
if [ "$EUID" -ne 0 ]; then
	SUDO=sudo
fi

function download_and_run()
{
    local url=$1
    sh -c "$(wget $url -O -)"
}

function install_gh()
{
    if [[ ! $(command -v gh) ]]; then
        echo "gh not found, installing"
        $SUDO apt install software-properties-common
        $SUDO apt-key adv --keyserver keyserver.ubuntu.com --recv-key C99B11DEB97541F0
        $SUDO apt-add-repository https://cli.github.com/packages
        $SUDO apt update
        $SUDO apt install gh
    else
        echo "gh already installed"
    fi
}

function install_git()
{

    if [[ ! $(command -v git) ]]; then
        echo "git not found, installing"
        $SUDO apt install git
    else
        echo "git already installed"
    fi

    git config --global user.email zhangfuwen@bytedance.com
    git config --global user.name zhangfuwen
    git config --global alias.st status
    git config --global alias.co checkout
    git config --global alias.ci commit
    git config --global alias.br branch
    git config --global alias.unstage 'reset HEAD'
    git config --global alias.last 'log -1'
    git config --global alias.lg "log --color --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit"
}

function install_nerdfonts()
{
    mkdir -p ~/bin/src/nerdfonts
    wget -O ~/bin/src/nerdfonts/FireCode.zip \
        https://github.com/ryanoasis/nerd-fonts/releases/download/v2.1.0/FiraCode.zip
    mkdir ~/.fonts
    unzip ~/bin/src/nerdfonts/FireCode.zip -d ~/.fonts/
    echo "now you can set your terminal fonts to 'FiraCode Nerd Font Mono Regular'"
}

# aarch64 x86_64
architecture=$(lscpu | awk '/Architecture:/{print $2}') 

function setup_vim()
{
	download_and_run https://gitee.com/zhangfuwen/GitNote/raw/master/vim/vim_setup.sh
}
