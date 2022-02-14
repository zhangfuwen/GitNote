#!/bin/bash

function trimLeading()
{
	echo "${1}" | sed -e 's/^[ \t]*//'
}

function trimTrailing()
{
	echo "${1}" | sed -e 's/\ *$//g'
}

function createMdFromTxt()
{
    for file in $(find . -name "*.txt"); do
        title=$(cat $file | head -n 3 | tail -n 1)
        title=$(trimLeading $title)

        newfile="${file%.txt}.md"
        echo $newfile

        echo "# $title" > $newfile
        echo "" >> $newfile
        cat $file >> $newfile

    done

}

function createDirIndex()
{
    for d in $(find . -type d); do
        echo "# $d" > $d/index.md
    done
}


function createIndex()
{
    echo '# List' > ../list.md

    files=$(find . -name "*.txt")
    for f in $files; do
        title=$(cat $f | head -n 3 | tail -n 1)
        title=$(trimLeading $title)
        echo "" >> ../list.md
        echo "[$title](${f%.txt}.html)" >> ../list.md


    done
    echo ""
}

createIndex

# for file in $(find . -name "*.txt"); do mv -- "$file" "${file%.txt}.md"; done
