#!/bin/bash

# Script to push the changes made to github
git add $1
if [ $? -ne 0 ]; then
	echo "git add failed"
	exit 1
fi

message=""
count=0
for i in $@
do
	if [ $count -ge 1 ]; then
		message="$message $i" 
	fi
	((count++))
done
git commit -m "$message"
if [ $? -ne 0 ]; then
	echo "git commit failed"
	exit 1
fi

git push -u origin master
if [ $? -ne 0 ]; then
	echo "git push failed"
	exit 1
fi
