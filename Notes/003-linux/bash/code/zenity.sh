# calendar
echo  $(zenity --calendar)

# input
zenity --entry --title "Name request" --text "Please enter your name:"

# select file
zenity --file-selection --multiple --filename "${HOME}/"
zenity --file-selection --save --confirm-overwrite --filename "${HOME}/"

# notification
zenity --info --width=400 --height=200 --text "This is a notification!"


# yes or no
 zenity --question --text "Are you sure you want to quit?" --no-wrap --ok-label "Yes" --cancel-label "No"
 echo $?

# select color
zenity --color-selection --color red --show-palette

# list select
zenity --list --column Selection --column Distribution FALSE Debian TRUE  Fedora
