#!/bin/bash
set -e

git add -A
git commit -m "wip" --allow-empty
git push

ssh ryujin 'cd ~/beaml && git pull && python3 '"$1"
