#!/bin/bash

# Check if .gitignore file exists
if [ ! -f .gitignore ]; then
  echo ".gitignore file not found!"
  exit 1
fi

# Read each line from .gitignore
while IFS= read -r pattern
do
  # Skip empty lines and comments
  [[ "$pattern" =~ ^\s*#.*$ ]] || [[ -z "$pattern" ]] && continue

  # Remove files or directories matching the pattern from Git index
  git rm -r --cached $pattern
done < .gitignore

# Commit the changes
git commit -m "Remove files matching patterns in .gitignore"

