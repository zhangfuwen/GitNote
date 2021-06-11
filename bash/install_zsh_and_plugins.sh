# install zsh if not present
if ! which zsh; then
    sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
fi

# add zshmarks plugin
if [[ ! -d ~/.oh-my-zsh/custom/plugins/zshmarks ]]; then
    cd ~/.oh-my-zsh/custom/plugins
    git clone https://github.com/jocelynmallon/zshmarks.git
    echo "edit .zshrc plugins=( [plugins...] zshmarks [plugins...])"
if