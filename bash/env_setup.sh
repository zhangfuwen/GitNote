#!/bin/bash

#ESC_SEQ='\x1b['
sec_seq='\033['

normal="0"

reset_bold="21"
reset_ul="24"

bold="1"
dim="2"
italic="3"
underline="4"
blink="5"
inverted="7"
strikethrough="9"

# Foreground colours
black="30"
red="31"
green="32"
yellow="33"
blue="34"
magenta="35"
cyan="36"
white="37"
# bright
br_black="90"
br_red="91"
br_green="92"
br_yello="93"
br_blue="94"
br_magenta="95"
br_cyan="96"
br_white="97"

# Background colours (optional)
bg_black="40"
bg_red="41"
bg_green="42"
bg_yellow="43"
bg_blue="44"
bg_magenta="45"
bg_cyan="46"
bg_white="47"
# light
bg_br_black="100"
bg_br_red="101"
bg_br_green="102"
bg_br_yello="103"
bg_br_blue="104"
bg_br_magenta="105"
bg_br_cyan="106"
bg_br_white="107"

function set_style() {
  if [[ $# == 5 ]]; then
    printf "%s" "\033[${!1};${!2};${!3};${!4};${!5}m"
  elif [[ $# == 4 ]]; then # fg, bg, font
    printf "%s" "\033${!1};${!2};${!3};${!4}m"
  elif [[ $# == 3 ]]; then # two functions
    printf "%s" "\033[${!1};${!2};${!3}m"
  elif [[ $# == 2 ]]; then
    printf "%s" "\033[${!1};${!2}m"
  elif [[ $# == 1 ]]; then
    printf "%s" "\033[${!1}m"
  fi
}

function style() {
  if [[ $# == 5 ]]; then
    printf "%s" "\033[${!1};${!2};${!3};${!4}m${5}\033[${normal}m"
  elif [[ $# == 4 ]]; then # fg, bg, font
    printf "%s" "\033${!1};${!2};${!3}m${4}\033[${normal}m"
  elif [[ $# == 3 ]]; then # two functions
    printf "%s" "\033[${!1};${!2}m${3}\033[${normal}m"
  elif [[ $# == 2 ]]; then
    printf "%s" "\033[${!1}m${2}\033[${normal}m"
  elif [[ $# == 1 ]]; then
    printf "%s" "$1"
  fi
}

function println() {
  if [[ $# == 1 ]]; then
    printf "%s\n" "$@"
  else
    first=$1
    shift
    rest="$@"
    printf "$first\n" $rest
  fi
}

function log_e() {
  printf "%s" "$(style inverted bold 'error:')"
  set_style red bold
  printf "%s" "$@"
  set_style normal
  printf "\n"
}

function log_w() {
  printf "%s" "$(style inverted bold 'warning:')"
  set_style yellow bold
  printf "%s" "$@"
  set_style normal
  printf "\n"
}

function log_i() {
  printf "%s" "$(style inverted bold 'info:')"
  set_style br_blue bold
  printf "%s" "$@"
  set_style normal
  printf "\n"
}

function log_d() {
  printf "%s" "$(style inverted bold 'debug:')"
  set_style dim bold
  printf "%s" "$@"
  set_style normal
  printf "\n"
}

function user_confirm() {
  local msg=$1
  while true; do
    read -p "${msg}[Yy/Nn]" yn
    case $yn in
    [Yy]*)
      return 0
      break
      ;;
    [Nn]*)
      return 1
      break
      ;;
    *) echo "Please answer yes[Yy] or no[Nn]." ;;
    esac
  done
}

SUDO=""

function get_sudo() {
  if [ "$EUID" -ne 0 ]; then
    SUDO=sudo
  fi
  return $SUDO
}
#get_sudo

if [[ ! $(command -v wget) || ! $(command -v curl) ]]; then
  echo "this script need wget and curl to work correctly, installing"
  $SUDO "$pkgman_install" wget curl
fi

function download_and_run() {
  local url=$1
  bash -c "$(wget ${url} -O -)"
}

function install_gh() {
  if [[ ! $(command -v gh) ]]; then
    echo "gh not found, installing"
    $SUDO apt install software-properties-common
    $SUDO apt-key adv --keyserver keyserver.ubuntu.com --recv-key C99B11DEB97541F0
    $SUDO apt-add-repository https://cli.github.com/packages
    $SUDO apt update
    $SUDO $pkgman_install gh
  else
    echo "gh already installed"
  fi
}

function install_git() {

  if [[ ! $(command -v git) ]]; then
    echo "git not found, installing"
    $SUDO $pkgman_install git
  else
    echo "git already installed"
  fi

}

function install_nerdfonts() {
  mkdir -p ~/bin/src/nerdfonts
  mkdir ~/.fonts

  wget -O ~/bin/src/nerdfonts/FireCode.zip \
    https://github.com/ryanoasis/nerd-fonts/releases/download/v2.1.0/FiraCode.zip
  unzip ~/bin/src/nerdfonts/FireCode.zip -d ~/.fonts/
  echo "now you can set your terminal fonts to 'FiraCode Nerd Font Mono Regular'"

  wget -O ~/bin/src/nerdfonts/RobotoMono.zip \
    https://github.com/ryanoasis/nerd-fonts/releases/download/v2.1.0/RobotoMono.zip
  unzip ~/bin/src/nerdfonts/RobotoMono.zip -d ~/.fonts/
  echo "now you can set your terminal fonts to 'RobotoMono Regular'"
}

# aarch64 x86_64
#architecture=""
#function get_architecture()
#{
#    local architecture=$(lscpu | awk '/Architecture:/{print $2}')
#    echo $architecture
#}
#architecture=$(get_architecture)

pkgman=""
pkgman_install=""
function get_pkg_manager() {
  if [[ $(command -v apt-get) ]]; then
    pkgman=apt-get
    pkgman_install="apt-get install"
  elif [[ $(command -v yum) ]]; then
    pkgman=yum
    pkgman_install="yum install"
  elif [[ $(command -v dnf) ]]; then
    pkgman=dnf
    pkgman_install="dnf install"
  elif [[ $(command -v pkg) ]]; then
    pkgman=pkg
    pkgman_install="pkg install"
  elif [ $(command -v apk) ]; then
    pkgman=apk
    pkgman_install="apk add"
  elif [ $(command -v brew) ]; then
    pkgman=brew
    pkgman_install="brew install"
  elif [[ $(command -v apt) ]]; then
    pkgman=apt
    pkgman_install="apt install"
  fi
  echo $pkgman
}
get_pkg_manager
echo "pkgman=$pkgman"
echo "pkgman_install=$pkgman_install"

function setup_vim() {
  echo "init nvim config file"
  mkdir -p ~/.config/nvim
  if test ! -f ~/.config/nvim/init.vim || user_confirm "$HOME/.config/nvim/init.vim exists, do you want to update?"; then
    cat >~/.config/nvim/init.vim <<EOF
" updated by GitNote
set runtimepath^=~/.vim runtimepath+=~/.vim/after
let &packpath=&runtimepath
source ~/.vimrc
EOF
  fi

  echo "installing vim-plug"
  if test ! -f ~/.vimrc || user_confirm "$HOME/.vimrc exists, do you want us to append new contents to it?"; then
    set -x
    curl -fLo ~/.vimrc --create-dirs https://github.com/zhangfuwen/GitNote/raw/master/vim/vimrc
    curl -fLo ~/.vim/coc.vim --create-dirs https://github.com/zhangfuwen/GitNote/raw/master/vim/coc.vim
    curl -fLo ~/.vim/plugins.vim --create-dirs https://github.com/zhangfuwen/GitNote/raw/master/vim/plugins.vim
    set +x
  fi

  echo "install ripgrep"
  $SUDO $pkgman_install  ripgrep
  echo "install cppman"
  $SUDO $pkgman_install  cppman
  echo "install universal-ctags"
  if not $SUDO $pkgman_install  universal-ctags; then
    $SUDO $pkgman_install ctags
  fi

  curl -fLo ~/.vim/autoload/plug.vim --create-dirs \
    https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim
}

function _adb() {
  if [[ $# -ge 2 ]] && [[ $1 == "-s" ]]; then
    serial=$2
    state=$(adb -s $serial get-state)
    if [[ $state != "device" ]]; then
      adb connect $serial
    fi
  else
    adb "$@"
  fi
}

function dir() {
  local dir=$(zenity --file-selection --filename="$pwd" --title="Select a File" --directory) && cd $dir
}

function gop() {
  if [[ $# == 0 ]]; then
    xdg-open .
  else
    xdg-open $1
  fi
}

function note_upload() {
  git add -A
  git commit -m "$(zenity --entry --text='commit msg')"
  git push
  git status
}

function setup_git() {
  git config --global user.email dean.sinaean@gmail.com
  git config --global user.name zhangfuwen
  git config --global alias.st status
  git config --global alias.co checkout
  git config --global alias.ci commit
  git config --global alias.br branch
  git config --global alias.unstage 'reset HEAD'
  git config --global mergetool.vimdiff.cmd "nvim -d $LOCAL $REMOTE $MERGED -c \'$wincmd w\' -c \'wincmd J\'"
  git config --global core.editor nvim
  git config --global core.excludesfile ~/.gitignore
  git config --global core.pager "less -R"
  git config --global url.ssh://git@github.com/.insteadOf https://github.com/
}

function install_zsh_and_oh_my_zsh() {
  $SUDO $pkgman_install zsh
  if test ! command -v curl; then
    $SUDO $pkgman_install curl
  fi
  sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
}

function wizard() {

  PS3="Enter a number: "

  select option in TryInstall_All \
    Install_Neovim \
    Setup_VimPlugins \
    Install_Github \
    Install_Git \
    Setup_Git \
    Install_NerdFonts \
    Github_AuthLogin \
    Install_OhMyZsh \
    Quit; do
    if test ${option} == "TryInstall_All"; then
      $SUDO ${pkgman_install} neovim
      setup_vim
      install_gh
      install_git
      setup_git
      install_nerdfonts
      gh auth login
    elif test ${option} == "Install_Neovim"; then
      $SUDO ${pkgman_install} neovim
    elif test ${option} == "Setup_VimPlugins"; then
      setup_vim
    elif test ${option} == "Install_Github"; then
      install_gh
    elif test ${option} == "Install_Git"; then
      install_git
    elif test ${option} == "Setup_Git"; then
      setup_git
    elif test ${option} == "Install_NerdFonts"; then
      install_nerdfonts
    elif test ${option} == "Github_AuthLogin"; then
      gh auth login
    elif test ${option} == "Install_OhMyZsh"; then
      install_zsh_and_oh_my_zsh
    elif test ${option} == "Quit"; then
      break
    fi
  done
}

#if [[ ! $(command -v zenity) ]]; then
#    $SUDO apt install zenity
#fi
#
#if [[ ! -d $HOME/bin ]]; then
#    mkdir $HOME/bin
#fi
#
#if [[ ! -d $HOME/bin/node-v16.17.1-linux-x64 ]]; then
#    wget https://nodejs.org/dist/v16.17.1/node-v16.17.1-linux-x64.tar.xz -O $HOME/Downloads/node.tar.xz
#    tar xvjf $HOME/Downloads/node.tar.xz -C $HOME/bin/
#fi
#export PATH=~/bin/node-v16.17.1-linux-x64/bin:$PATH
#
#
#export PATH=~/bin/nvim-linux64/bin:$PATH
#alias vi=nvim
