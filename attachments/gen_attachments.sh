find . -not -path '\.' | xargs -I {} echo "["{}"]("{}")" > README.md
