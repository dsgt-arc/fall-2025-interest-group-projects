# Information Retrieval

---

## Setting up the Virtual Environment

To stay within your home directory quota, we'll create the virtual environment in the fast, temporary file system located at **`$TMPDIR`**.

Make sure to add this to your `~/.bashrc` file:
```bash
# set a temporary directory
export TMPDIR=${TMPDIR:-/tmp}
```

### Option A. Manual Virtual Environment Setup (Recommended for Jupyter Users)

This approach creates a reusable `.venv` in `$TMPDIR` that you can activate manually and connect to your Jupyter notebook.

#### 1. **Create and Activate the Environment in `$TMPDIR`:**

```bash
# Create the virtual environment directly in $TMPDIR
uv venv $TMPDIR/.venv

# Activate it
source $TMPDIR/.venv/bin/activate
```

#### 2. **Navigate to the `user/` directory and link dependencies:**

Navigate to your user directory and create a **symbolic link** named `.venv` that points back to the environment in `$TMPDIR`.

Assuming your current repo user directory is:
`~/dsgt-arc/fall-2025-interest-group-projects/user/<your-user-name>`

```bash
# Navigate to your project directory
cd ~/dsgt-arc/fall-2025-interest-group-projects/user/<your-user-name>

# Create a symbolic link named .venv in your user folder, 
# pointing to the actual environment in $TMPDIR
ln -s $TMPDIR/.venv $(pwd)/.venv

# Install project dependencies defined in pyproject.toml
cd ~/dsgt-arc/fall-2025-interest-group-projects/project/03-information-retrieval
uv pip install -e .
```

Once done, open the Jupyter notebook and select the newly linked **`.venv`** kernel.

---

## Java installation

Before running this notebook, you need to make sure Java is installed (needed by PyTerrier).

---

### Java Installation for PyTerrier

**PyTerrier requires Java 11 or higher**

#### Step 1: Check if Java is Already Installed

Run this in your terminal:

```bash
java --version
```

If you see **"openjdk version 11"** or higher, then Java is installed.

---

#### Step 2: Check if OpenJDK is installed

Run this in terminal to download OpenJDK:

```bash
# go to the home directory
cd ~

# create a ~/bin folder and cd into bin
mkdir -p /bin
cd bin

# install OpenJDK
wget https://download.java.net/java/GA/jdk17.0.2/dfd4a8d0985749f896bed50d7138ee7f/8/GPL/openjdk-17.0.2_linux-x64_bin.tar.gz

# unzip the file
tar -xzf openjdk-17.0.2_linux-x64_bin.tar.gz

# Set JAVA_HOME to the directory you just unzipped
export JAVA_HOME=~/bin/jdk-17.0.2
```

Add the new Java's `~/bin` directory to your `~/.bashrc` file. This ensures it's found before the system's old Java:
```bash
# open your .bashrc file, add this line at the end
export PATH=$JAVA_HOME/bin:$PATH

# after adding the line to the file, save it and source it
source ~/.bashrc

# check for java version
javac --version
```

---

### That's it!
