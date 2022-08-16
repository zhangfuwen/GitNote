#!/bin/bash


function install_funim() {
    [[ ! -d ~/Code ]] && mkdir ~/Code
    pushd ~/Code
    git clone https://github.com/zhangfuwen/ibusfunim
    pushd ibusfunim
    mkdir build
    pushd build

    sudo apt install cmake pkg-config libibus-1.0-dev libgtkmm-3.0-dev libpulse-dev libpsl-dev libidn2-dev
    cmake .. -DCMAKE_BUILD_TYPE=Debug 
    make -j10
    sudo make install

    popd 
    popd 
    popd 

}

function setup_funim() {
    echo "reload ibus-daemon"
    ibus-daemon -drx

    echo "set current engine FunEngine"
    ibus engine FunEngine

    echo "set system default engine FunEngine"
    gsettings set org.freedesktop.ibus.general preload-engines "['FunEngine']"

    echo "current engine: $(ibus engine)"
}


