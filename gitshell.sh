#!/bin/sh

# Configure global Git user name and email for all repositories
git config --global user.name "Soumya Banerjee" 
git config --global user.email neel.soumya@gmail.com

# Stage all files in the _freeze directory (typically contains frozen/rendered outputs)
git add _freeze/*

# Stage setup documentation and course materials (Quarto markdown and YAML files)
git add setup.md materials/*.qmd materials/*.yml




