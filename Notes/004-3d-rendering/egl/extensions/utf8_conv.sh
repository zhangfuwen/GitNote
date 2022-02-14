#!/bin/bash
# converting all files in a dir to utf8

function smart()
{
    for f in $(find . -name "*.md") 
    do
        if test -f $f; then
            echo -e "\nConverting $f"
            CHARSET="$(file -bi "$f"|awk -F "=" '{print $2}')"
            if [ "$CHARSET" != utf-8 ]; then
                echo "file $f has encoding $CHARSET, converting to utf-8"
                iconv -f "$CHARSET" -t utf8 "$f" -o "$f"
            fi
        else
            echo -e "\nSkipping $f - it's a regular file";
        fi
    done
}


function brutal_force()
{
    for f in $(find . -name "*.md") 
    do
        iconv -f "us-ascii" -t utf8 "$f" -o "$f"
        if [[ $? != 0 ]]; then
            echo "failed to file $f"
        fi
    done

}

brutal_force
