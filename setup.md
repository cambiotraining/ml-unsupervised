---
title: "Data & Setup"
number-sections: false
---

<!-- 
Note for Training Developers:
We provide instructions for commonly-used software as commented sections below.
Uncomment the sections relevant for your materials, and add additional instructions where needed (e.g. specific packages used).
Note that we use tabsets to provide instructions for all three major operating systems.
-->

::: {.callout-tip level=2}
## Workshop Attendees

If you are attending one of our workshops, we will provide a training environment with all of the required software and data.  
If you want to setup your own computer to run the analysis demonstrated on this course, you can follow the instructions below.


## Data

The data used in these materials is provided as a zip file. 
Download and unzip the folder to your Desktop to follow along with the materials.

<!-- Note for Training Developers: add the link to 'href' -->
<a href="https://github.com/cambiotraining/ml-unsupervised/raw/refs/heads/main/course_files/data/data.zip">
  <button class="btn"><i class="fa fa-download"></i> Download</button>
</a>

## Software

### 📦 Installation Instructions (for Google Colab)

<!--
>
### 1. Clone the repository (optional, for full access)
```python
!git clone https://github.com/neelsoumya/python_machine_learning.git
%cd python_machine_learning
```

### 2. Install required Python packages


```python
!pip install numpy pandas keras tensorflow scikit-learn seaborn matplotlib scanpy
```

or

```python
!pip install -r requirements.txt
```


### 3. (Optional) Download specific files directly from GitHub
```python
# Example: Download a specific notebook
!wget https://raw.githubusercontent.com/neelsoumya/python_machine_learning/main/PCA_movie_ratings.ipynb
```

---

-->

**How to use in Colab:**

- Open a new notebook in [Google Colab](https://colab.research.google.com/)

- Run the commands in code cells.

- You can now create notebooks and run any of the scripts in Google Colab.

- If you want to run the notebooks in Colab, you can also use the "Open in Colab" badge below:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)


**Repository link:**  
[https://cambiotraining.github.io/ml-unsupervised/](https://cambiotraining.github.io/ml-unsupervised/)

<!--
### Quarto

To develop and render the course materials website, you will need to install _Quarto_:

- Download and install [Quarto](https://quarto.org/docs/get-started/) (available for all major OS). 
- If you are developing materials using executable `.qmd` documents, it is recommended that you also install the extensions for your favourite IDE (e.g. RStudio, VS Code).
- If you are developing materials using **JupyterLab** or **Jupyter Notebooks**, please install [Jupytext](https://jupytext.readthedocs.io/en/latest/install.html).
  - Use the [paired notebook](https://jupytext.readthedocs.io/en/latest/paired-notebooks.html) feature to have synchronised `.ipynb`/`.qmd` files. Only `.qmd` files should be pushed to the repository (`.ipynb` files have been added to `.gitignore`).

-->

<!--
### R and RStudio

::: {.tabset group="os"}

#### Windows

Download and install all these using default options:

- [R](https://cran.r-project.org/bin/windows/base/release.html)
- [RTools](https://cran.r-project.org/bin/windows/Rtools/)
- [RStudio](https://www.rstudio.com/products/rstudio/download/#download)

#### Mac OS

Download and install all these using default options:

- [R](https://cran.r-project.org/bin/macosx/)
- [RStudio](https://www.rstudio.com/products/rstudio/download/#download)

#### Linux

- Go to the [R installation](https://cran.r-project.org/bin/linux/) folder and look at the instructions for your distribution.
- Download the [RStudio](https://www.rstudio.com/products/rstudio/download/#download) installer for your distribution and install it using your package manager.

:::
-->


<!--
### Conda

Open a terminal and run:

```bash
wget -q -O - https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
rm Miniconda3-latest-Linux-x86_64.sh
conda init
conda config --add channels defaults; conda config --add channels bioconda; conda config --add channels conda-forge; conda config --set channel_priority strict
conda install -y mamba
```

Note: Windows users can use WSL2 (see @wsl).
-->

<!--
### Singularity

::: {.panel-tabset group="os"}

#### Windows

You can use _Singularity_ from the _Windows Subsystem for Linux_ (see @wsl).  
Once you setup WSL, you can follow the instructions for Linux.

#### Mac OS

Singularity is [not available for Mac OS](https://docs.sylabs.io/guides/3.0/user-guide/installation.html#install-on-windows-or-mac).

#### Linux

These instructions are for _Ubuntu_ or _Debian_-based distributions[^1].

[^1]: See the [Singularity documentation page](https://docs.sylabs.io/guides/3.0/user-guide/installation.html#install-on-linux) for other distributions.

```bash
sudo apt update && sudo apt upgrade && sudo apt install runc
CODENAME=$(lsb_release -c | sed 's/Codename:\t//')
wget -O singularity.deb https://github.com/sylabs/singularity/releases/download/v3.10.2/singularity-ce_3.10.2-${CODENAME}_amd64.deb
sudo dpkg -i singularity.deb
rm singularity.deb
```

:::
-->


### 2. Run locally

* Install VSCode (see below).

* Install required Python packages

```python
pip install numpy pandas keras tensorflow scikit-learn seaborn matplotlib scanpy
```

or

```python
pip install -r requirements.txt
```



### Visual Studio Code

::: {.panel-tabset group="os"}

#### Windows

- Go to the [Visual Studio Code download page](https://code.visualstudio.com/Download) and download the installer for your operating system. 
  Double-click the downloaded file to install the software, accepting all the default options. 
- After completing the installation, go to your Windows Menu, search for "Visual Studio Code" and launch the application. 
- Go to "_File > Preferences > Settings_", then select "_Text Editor > Files_" on the drop-down menu on the left. Scroll down to the section named "_EOL_" and choose "_\\n_" (this will ensure that the files you edit on Windows are compatible with the Linux operating system).

#### Mac OS

- Go to the [Visual Studio Code download page](https://code.visualstudio.com/Download) and download the installer for Mac.
- Go to the Downloads folder and double-click the file you just downloaded to extract the application. Drag-and-drop the "Visual Studio Code" file to your "Applications" folder. 
- You can now open the installed application to check that it was installed successfully (the first time you launch the application you will get a warning that this is an application downloaded from the internet - you can go ahead and click "Open").

#### Linux (Ubuntu)

- Go to the [Visual Studio Code download page](https://code.visualstudio.com/Download) and download the installer for your Linux distribution. Install the package using your system's installer.

:::


<!-- 
## WSL2

There are detailed instructions on how to install WSL on the [Microsoft documentation page](https://learn.microsoft.com/en-us/windows/wsl/install). 
But briefly:

- Click the Windows key and search for  _Windows PowerShell_, right-click on the app and choose **Run as administrator**. 
- Answer "Yes" when it asks if you want the App to make changes on your computer. 
- A terminal will open; run the command: `wsl --install`. 
  - This should start installing "ubuntu". 
  - It may ask for you to restart your computer. 
- After restart, click the Windows key and search for _Ubuntu_, click on the App and it should open a new terminal. 
- Follow the instructions to create a username and password (you can use the same username and password that you have on Windows, or a different one - it's your choice). 
  - **Note:** when you type your password nothing seems to be happening as the cursor doesn't move. However, the terminal is recording your password as you type. You will be asked to type the new password again to confirm it, so you can always try again if you get it wrong the first time.
- You should now have access to a Ubuntu Linux terminal. 
  This behaves very much like a regular Ubuntu server, and you can install apps using the `sudo apt install` command as usual. 


### Configuring WSL2

After installation, it is useful to **create shortcuts to your files on Windows**. 
Your main `C:\` drive is located in `/mnt/c/` and other drives will be equally available based on their letter. 
It may be convenient to create shortcuts to commonly-used directories, which you can do using _symbolic links_. 
Here are some commands to automatically create shortcuts to your Windows "Documents",  "Desktop" and "Downloads" folders:

```bash
ln -s $(wslpath $(powershell.exe '[environment]::getfolderpath("mydocuments")')) ~/Documents
ln -s $(wslpath $(powershell.exe '[environment]::getfolderpath("desktop")')) ~/Desktop
ln -s $(wslpath $(powershell.exe '[environment]::getfolderpath("downloads")')) ~/Downloads
```

You may also want to configure the Windows terminal to automatically open _WSL2_ (instead of the default Windows Command Prompt or Powershell):

- Search for and open the "<i class="fa-solid fa-terminal"></i> Terminal" application.
- Click on the downward arrow <i class="fa-solid fa-chevron-down"></i> in the toolbar.
- Click on "<i class="fa-solid fa-gear"></i> Settings".
- Under "Default Profile" select "<i class="fa-brands fa-linux"></i> Ubuntu". 
-->