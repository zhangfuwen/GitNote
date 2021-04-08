#!/bin/bash

SUDO=""
if [ "$EUID" -ne 0 ]
	SUDO=sudo
fi

function download_and_run()
{
    local url=$1
    sh -c "$(wget $url -O -)"
}

function setup_vim()
{
	download_and_run https://gitee.com/zhangfuwen/GitNote/raw/master/vim/vim_setup.sh
}
