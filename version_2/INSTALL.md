# Installation and Setup Guide: Hidden Attractors FO Library

This guide provides step-by-step instructions to install and configure the `hidden-attractors-fo` package on **Windows**, **Linux**, and **macOS**.

---

## Prerequisites
* **Python**: Version `3.11` or newer.
* **Package Manager**: `pip` (included with standard Python installations).

---

## Step 1: Install the Python Package

Clone or copy the `version_2` directory to your local machine, navigate to its root directory in your terminal, and run:

### For standard usage (Recommended):
```bash
pip install -e .
```
*The `-e` flag installs the package in **editable mode**. This maps the codebase to your environment so that any updates you make are immediately active without needing a reinstall.*

### For running tests and developer tasks:
```bash
pip install -e ".[dev]"
```

---

## Step 2: C-Compiler Toolchain Setup (By Platform)

The library integrates high-performance numerical integrators compiled in C. These compile dynamically on demand. Set up your compiler toolchain below:

### 1. Windows Setup
Windows requires a compatible C compiler (like `gcc` via MinGW/MSYS2) registered in your system `PATH`.

1. **Install MSYS2**:
   Download and run the installer from [msys2.org](https://www.msys2.org/).
2. **Install GCC**:
   Open the **MSYS2 UCRT64** terminal and run:
   ```bash
   pacman -S mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-make
   ```
3. **Configure Environment Variables**:
   Add the binary path to your system's environment `PATH` variable. The default path is:
   `C:\msys64\ucrt64\bin`
4. **Verify installation**:
   Open a new PowerShell or Command Prompt window and check that GCC is recognized:
   ```bash
   gcc --version
   ```

### 2. Linux Setup
Most Linux distributions have GCC pre-installed or easily available in package repositories.

* **Ubuntu / Debian**:
  ```bash
  sudo apt update
  sudo apt install build-essential
  ```
* **Fedora / RHEL**:
  ```bash
  sudo dnf groupinstall "Development Tools"
  ```
* **Arch Linux**:
  ```bash
  sudo pacman -S base-devel
  ```

### 3. macOS Setup
macOS uses the Clang compiler included in Xcode Command Line Tools.

1. **Install Xcode Command Line Tools**:
   Open a terminal and run:
   ```bash
   xcode-select --install
   ```
2. **Install OpenMP support** (Recommended for multi-threaded basins/bifurcations):
   Install `libomp` via Homebrew:
   ```bash
   brew install libomp
   ```
3. *(Optional)* **Disable OpenMP**:
   If you wish to compile without OpenMP multi-threading support, set the environment variable:
   ```bash
   export ALLOW_NO_OPENMP=1
   ```

---

## Step 3: Verification

Once the package is installed, verify the command-line interface:

```bash
hidden-attractors --help
```

For development or validation work, run the test suite from `version_2/`:

```bash
cd version_2
python -m pytest -q
```

Archived validation records are stored under `validation/freeze_audit/`.


---

## Step 4: Run a Quick Test Experiment

To verify that the installation is fully functional, try running the official example of the fractional non-smooth Chua system with a biased describing function in quick mode:

```bash
python examples/chua_nonsmooth_biased_hidden_attractor/run_example.py --quick
```

The outputs (CSV trajectories and reports) will be written to `outputs/example_chua_nonsmooth_biased_hidden_attractor/` and the generated graphics will be exported to `version_2/library_figures/`.
