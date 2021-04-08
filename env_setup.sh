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

# aarch64 x86_64
architecture=$(lscpu | awk '/Architecture:/{print $2}') 

function setup_vim()
{
	download_and_run https://gitee.com/zhangfuwen/GitNote/raw/master/vim/vim_setup.sh
}
