#!/bin/bash 

function prepend()
{
    text=$1
    file=$2
    echo $text | cat - $2 > temp && mv temp $2
}
files=$(find . -name "*.adoc") 
for file in $files; do 
    line1=$(head -n 1 $file) 
    if [[ $line1 = "---" ]]; then 
        echo "skip $file" 
    else 
        path=$(dirname $file) 
        path=${path:1}
        filename=$(basename -s .adoc $file) 
        echo "path:$path" 
        echo "basename:$filename" 
        prepend ""  $file 
        prepend "---"  $file 
        prepend "layout: default"  $file 
        prepend "permalink:/Notes/004-3d-rendering/vulkan$path/$filename.html"  $file 
        prepend "---"  $file 
    fi 
done
