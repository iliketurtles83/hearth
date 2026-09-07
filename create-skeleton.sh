#!/bin/bash

# Create skeleton of folders, files and functions
echo "# Repository Map" > repo_skeleton.md
tree -I 'node_modules|.git|__pycache__|.venv' >> repo_skeleton.md

echo -e "\n## File Archetypes & Signatures" >> repo_skeleton.md
find . -name "*.py" -not -path "*/.*" | xargs grep -E "^\s*(class |def )" >> repo_skeleton.md