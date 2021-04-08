#!/bin/bash

function download_and_run()
{
    local url=$1
    which wget 1&>1;
    if [[ $? != 0]]; then
        echo "wget not installed, try using curl"
        which curl 1&>1;
        if [[ $? != 0]]; then 
            echo "curl not installed, please install wget or curl"
        else
            sh -c "$(curl -fsSL $url)"
        fi
    else
        sh -c "$(wget $url -O -)"
    fi
}

download_and_run https://gitee.com/zhangfuwen/GitNote/raw/master/vim/vim_setup.sh