# Healthcare Cost Comparison

Open Healthcare_Cost_Comparison.html to read completed results without running code.
Open Healthcare_Cost_Comparison.ipynb in Jupyter or VS Code and Run All to reproduce.
Keep insurance.csv in the same directory as the notebook. Python 3.11+ recommended.

Install dependencies:
    python -m pip install -r requirements.txt
Execute notebook from this directory:
    python -m nbconvert --execute --to notebook --inplace Healthcare_Cost_Comparison.ipynb
Alternative reproducible script:
    python experiment.py

The CSV files retain all test predictions, seed metrics and CV tuning results.
metadata.json records source URL, SHA-256, data rules, splits and parameters.
Training uses scikit-learn MLP, NOT PyTorch; no dropout/clipping was tested.
Costs are in original charges units; data provenance/currency is not independently verified.
This educational benchmark is not a medical or deployed insurance-pricing study.

Validation on 2026-10-08:
- All 12 notebook code cells ran top-to-bottom in a fresh in-process IPython shell.
- Standard Jupyter TCP and IPC kernels could not start because workspace sockets are restricted.
  This notebook has no kernel-dependent magics/widgets; all computations and outputs were executed.
- Rerun metrics matched the separate experiment.py run.
- Direct calculations of MAE/RMSE from every prediction vector matched saved metrics.
- ReLU chain-rule input gradients agreed with finite differences.
- Saved figures and an HTML-derived reading preview were visually inspected.
- One test split and seed SD do not establish a generalization confidence interval or a robust winner.
