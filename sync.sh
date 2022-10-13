#!/usr/bin/env bash

git fetch
git rebase --autostash
git add .
git commit -m "update"