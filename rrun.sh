#!/bin/bash
set -e

if [ -z "$1" ]; then
  echo "usage: ./rrun.sh <script.py>"
  exit 1
fi

git add -A
git commit -m "wip" --allow-empty
git push

ssh ryujin "cd ~/beaml && git pull && python3 $1"
