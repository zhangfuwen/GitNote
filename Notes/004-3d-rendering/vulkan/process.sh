#!/bin/bash

function getText() {
    if [[ ! -d texts ]]; then
        mkdir texts
    fi
    cd texts
    cat ../original_list.md| egrep -o "http.*html" | xargs wget --parent
    cd -
}


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
    cd texts
    for file in $(find . -name "*.txt"); do
        title=$(cat $file | head -n 3 | tail -n 1)
        title=$(trimLeading $title)

        newfile="${file%.txt}.md"
        echo $newfile

        echo "# $title" > $newfile
        echo "" >> $newfile
        cat $file >> $newfile

    done
    cd -
    if [[ ! -d extensions ]]; then
        mkdir extensions
    fi
    find texts -name "*.md" | xargs -I {} mv {} extensions    files=$(find ../texts/ -name "*.md")
}

function createDirIndex()
{
    for d in $(find . -type d); do
        echo "# $d" > $d/index.md
    done
}


function createIndex()
{
    echo "---" > list.md
    echo "" >> list.md
    echo "has_children: true" >> list.md
    echo "" >> list.md
    echo "---" >> list.md
    echo '# List' >> list.md
    echo "" >> list.md

    lines=()
    files=$(find extensions -name "*.md")
    for f in $files; do
        title=$(basename $f)
        title=${title%.md}
        lines+=("[$title](${f%.md}.html)")
    done


    readarray -t sorted < <(for a in "${lines[@]}"; do echo "$a"; done | sort)

#    for a in ${sorted[@]}; do echo $a; done

    for line in ${sorted[@]}; do
        echo $line >> list.md
        echo "" >> list.md
    done
}

function getEnc()
{
    encguess $1 | awk '{print $2}'
}

function conv2utf8()
{
    test -d extensions_new || mkdir extensions_new

    files=$(find extensions -name "*.md")
    echo "" > conv_failed_files
    echo "" > conv_done_files
    echo "" > conv_copy_files
    for file in $files; do
        newpath=extensions_new/$(basename $file)

        enc=$(getEnc $file)
        if [[ $enc =~ ^(US-ASCII|UTF-8)$ ]]; then
            echo "file $file $enc" >> conv_copy_files
            cp $file $newpath
        else
            if iconv -f $enc -t utf8 $file -o $newpath; then
                echo "$file $enc done" >> conv_done_files
            else
                echo $file >> conv_failed_files

            fi
        fi
    done
}

getText
#createMdFromTxt

#conv2utf8

#createIndex

# for file in $(find . -name "*.txt"); do mv -- "$file" "${file%.txt}.md"; done
