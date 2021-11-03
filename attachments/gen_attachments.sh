find . -not -path '\.' | xargs -I {} echo "["{}"]("{}")" ; echo "" > README.md
