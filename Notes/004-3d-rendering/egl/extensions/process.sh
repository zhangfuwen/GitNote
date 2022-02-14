#!/bin/bash

function trimLeading()
{
	echo "${1}" | sed -e 's/^[ \t]*//'
}

function trimTrailing()
{
	echo "${1}" | sed -e 's/\ *$//g'
}

for file in $(find . -name "*.txt"); do
	title=$(cat $file | head -n 3 | tail -n 1)
	title=$(trimLeading $title)

	newfile="${file%.txt}.md"
	echo $newfile

	echo "# $title" > $newfile
	echo "" >> $newfile
	cat $file >> $newfile

done

# for file in $(find . -name "*.txt"); do mv -- "$file" "${file%.txt}.md"; done
