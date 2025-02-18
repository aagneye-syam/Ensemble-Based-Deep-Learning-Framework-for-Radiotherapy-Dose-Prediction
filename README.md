# Ensemble-Based-Deep-Learning-Framework-for-Radiotherapy-Dose-Prediction

use python 10

Dataset - https://drive.google.com/drive/folders/16hno2tiC_6lCygMIniS9F51_r6IpSjqW?usp=sharing

# Project Setup Guide

This guide explains how to set up and run the project using a Python virtual environment on macOS.

## System Requirements

- macOS
- Python 3.10.9 (required for TensorFlow compatibility)
- Terminal access

## Prerequisites

Ensure Python 3.10.9 is installed at:
```bash
/Library/Frameworks/Python.framework/Versions/3.10/bin/python3.10
```

## Setup Instructions

### 1. Create Virtual Environment

```bash
# Navigate to your project directory
cd /Users/aagneye-syam/Desktop/ensemble

# Create a new virtual environment
/Library/Frameworks/Python.framework/Versions/3.10/bin/python3.10 -m venv venv
```

### 2. Activate Virtual Environment

```bash
# Activate the virtual environment
source venv/bin/activate
```

### 3. Install Dependencies

First, create a `requirements.txt` file with the following content:

```txt
tensorflow-macos>=2.12.0
tensorflow-metal>=1.0.0
numpy>=1.22.4
pandas>=1.4.2
matplotlib>=3.5.2
tqdm>=4.64.0
h5py>=3.7.0
scikit-learn>=1.0.2
```

Then install the dependencies:

```bash
# Upgrade pip
pip install --upgrade pip

# Install TensorFlow for Mac
pip install tensorflow-macos tensorflow-metal

# Install other requirements
pip install -r requirements.txt
```

### 4. Running the Script

```bash
# Make sure your virtual environment is activated
source venv/bin/activate

# Run your script
python main.py
```

### 5. Deactivate Virtual Environment

When you're done working on the project:

```bash
deactivate
```

## Troubleshooting

### Common Issues

1. If you see "No module found" errors:
   ```bash
   # Verify your virtual environment is activated
   which python
   # Should show path to your virtual environment
   
   # Reinstall the missing package
   pip install <package-name>
   ```

2. If TensorFlow isn't working:
   ```bash
   # Verify TensorFlow installation
   python -c "import tensorflow as tf; print(tf.__version__)"
   ```

### Verification Steps

To verify your setup is correct:

```bash
# Activate virtual environment
source venv/bin/activate

# Check Python version
python --version
# Should show Python 3.10.9

# Verify TensorFlow installation
python -c "import tensorflow as tf; print(tf.__version__)"
```

## Daily Usage

1. Open Terminal
2. Navigate to project directory:
   ```bash
   cd /Users/aagneye-syam/Desktop/ensemble
   ```
3. Activate virtual environment:
   ```bash
   source venv/bin/activate
   ```
4. Run your script:
   ```bash
   python main.py
   ```
5. When finished:
   ```bash
   deactivate
   ```

## Maintaining Virtual Environment

To update packages:
```bash
pip install --upgrade -r requirements.txt
```

To list installed packages:
```bash
pip list
```

## Notes

- Always ensure you're using the virtual environment when running the script
- Do not use Python 3.13 as it's not compatible with TensorFlow
- Keep the `requirements.txt` file updated if ▋