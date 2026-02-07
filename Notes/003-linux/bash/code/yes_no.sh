function user_confirm()
{
    local msg=$1
    while true; do
        read -p "$msg[Yy/Nn]" yn
        case $yn in
            [Yy]* ) echo 0; break;;
            [Nn]* ) echo 1; break;;
            * ) echo "Please answer yes[Yy] or no[Nn].";;
        esac
    done
}

if [[ $(user_confirm "Do you want to do this?") ]]; then
    echo "you said yes"
else
    echo "you said no"
fi
