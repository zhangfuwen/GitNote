#!/bin/bash
script_dir=$(dirname "$0")
source $script_dir/env_setup.sh

# VIM=vim
# function download_nvim_x86()
# {
#     echo "try install neovim"
#     if [[ ! $(command -v nvim) ]];then
#         mkdir ~/bin/
#         wget -O ~/bin/nvim-nightly.tar.gz https://github.com/neovim/neovim/releases/download/nightly/nvim-linux64.tar.gz
#         tar xvzf ~/bin/nvim-nightly.tar.gz -C ~/bin/
#         [[ -f ~/.bashrc ]] && echo "export PATH=~/bin/nvim-linux64/bin:$PATH" >> ~/.bashrc && source ~/.bashrc
#         [[ -f ~/.zshrc ]] && echo "export PATH=~/bin/nvim-linux64/bin:$PATH" >> ~/.zshrc && source ~/.zshrc
#         rm nvim-linux64.tar.gz  -C ~/bin/
#     fi
#     export PATH=~/bin/nvim-linux64/bin:$PATH
#     echo "nvim installed"
#     VIM=nvim
# }
#
# function download_nvim_arm64()
# {
#     echo "try install vim"
#     if [[ ! $(command -v vim) ]]; then
#         $SUDO $pkgman install -y vim
#     fi
#     echo "vim installed"
#     VIM=vim
# }

# if [[ "$architecture" == "aarch64" ]]; then
#     download_nvim_arm64
# elif [[ "$architecture" == "x86_64" ]]; then
#     download_nvim_x86
# fi

# if [[ -d ~/bin/nvim-linux64 ]]; then
#     export PATH=~/bin/nvim-linux64/bin:$PATH
# fi

# echo "install vim python provider"
# $SUDO apt install python3 python3-pip
# pip install neovim



# echo "install vim pluggins"
# $VIM +PlugInstall +qall




