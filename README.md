# ml-unsupervised

## Introduction

This repository contains the materials for the course on applied unsupervised machine learning.

**Course Developers**: see our [guidelines page](https://cambiotraining.github.io/quarto-course-template/materials.html) if contributing materials.

These materials are released under a [CC BY 4.0](LICENSE.md) license.

## Steps for course developers

* Create a Python virtual environment

```bash
   python3 -m venv .venv
```

* Activate environment

```bash
source .venv/bin/activate
```

* Install all requirements

```bash
   pip install -r requirements.txt
```

* Install Quarto and VS Code

<!--* Preview Quarto markdown-->

* Change the `.qmd` files

* Render using quarto

```bash
quarto render
```

* Commit files in `_freeze` folder and any other `.qmd` files changed

```bash
chmod 755 gitshell.sh
./gitshell.sh
```

* A single command to compile the materials

```bash
./compile.sh
```
* Push changes to the repository

```bash
git commit -m "Updated materials"
git push
```

* To generate presentations

```bash
quarto render materials/quarto_presentation_day1.qmd --to revealjs 
```

## Directory structure

- **README.md**: Project overview and setup instructions  
- **requirements.txt**: Python dependencies  
- **references.bib**: Bibliography for the course  
- **course_files/scripts/**: Python scripts for data generation and examples  
- **course_files/data/**: Example datasets  
- **materials/**: Course content and chapters in Quarto format  
- **.venv/**: (Optional) Virtual environment for Python dependencies

## Contact

Soumya Banerjee

sb2333@cam.ac.uk
