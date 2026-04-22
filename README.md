# DS Project

This project standardizes a small supply-chain dataset, computes PCA, prints the PCA loading weights, and generates a scatter plot of the cases on the first two principal components.

## Requirements

- Python 3
- `pip`

The Python dependencies are listed in `requirements.txt`.

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

From the project folder, activate the virtual environment if it is not already active:

```bash
source .venv/bin/activate
```

Then run:

```bash
python main.py
```

## What the script does

Running `main.py` will:

1. Standardize the original numeric variables
2. Compute PCA manually from the covariance matrix and eigenvectors
3. Print the PCA scores for each case
4. Print the loading weights for each original variable on PC1 and PC2
5. Save a plot to `pca_plot.png`
6. Open the plot window if your local matplotlib backend supports it

## Output files

- `pca_plot.png`: scatter plot of the cases in PCA space

## Notes

- If `python main.py` fails because packages are missing, make sure the virtual environment is activated.
- On macOS, running from the terminal after activating `.venv` should open the plot window normally.
