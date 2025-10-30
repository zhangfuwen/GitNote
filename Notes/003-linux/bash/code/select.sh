PS3="Enter a number: "

select character in Sheldon Leonard Penny Howard Raj Exit
do
    echo "Selected character: $character"
    echo "Selected number: $REPLY"
    if [[ $character = "Exit" ]]; then
      break
    fi
done