#!/bin/sh


# This script is used to compile the materials for the course.

echo "Compiling materials... \n"

source .venv/bin/activate


echo "Installing required Python packages... \n"
# It installs the required Python packages and compiles the Quarto documents.
pip install -r requirements.txt

# Install R packages
echo "Installing required R packages... \n"
R --no-save < course_files/scripts/install_packages.R

# Compile the Quarto documents
echo "Compiling Quarto documents... \n"
# This command renders all the .qmd files in the materials directory.
# It will generate the HTML and PDF files for the course materials.
quarto render

# git status
git status

# git add 
./gitshell.sh

echo "Materials compiled successfully! \n"
