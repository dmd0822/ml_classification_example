---
description: 'This prompt directs GitHub Copilot to scan a Machine Learning workspace, gather the latest experiment results, and automatically generate a polished, date‑stamped Jupyter notebook for analysis. The notebook includes structured sections, Markdown explanations, required plots, and reproducible code that clearly presents training behavior, evaluation strategy, and final test metrics.
'
agent: 'agent'
---

Goal:
Scan the current workspace for Machine Learning experiment directories, gather the latest results from each experiment, and automatically generate a new Jupyter notebook for analysis. The notebook must be named using the current date/time and must follow the workspace’s structure and conventions.

Instructions for GitHub Copilot
You are an assistant that generates a complete, polished Jupyter notebook for analyzing Machine Learning experiment results. Follow these instructions exactly.
1. Workspace Scanning
- Search the workspace recursively for experiment folders.
- An “04-predictions” contains artifacts such as:
- model checkpoints (.keras, .h5, .pt, etc.)
- training logs (.json, .csv, .txt)
- metrics files (metrics.json, results.json)
- confusion matrices or predictions (predictions.csv, confusion_matrix.npy)
- For each experiment, identify the latest run based on timestamped folders or file modification times.
2. Notebook Creation
- Create a new Jupyter notebook named with the current date/time, e.g.:
analysis_2025-12-23_09-15.ipynb
- Create a folder with the same name to store generated PNG charts.

3. Notebook Structure Requirements
Professional Header (Markdown cell)
Include:
- Project Title
- Date of notebook creation
- One-sentence description of the notebook’s purpose
(“This notebook analyzes the latest ML experiment results.”)

4. Section Organization (Part A Structure)
Follow this structure:
A.1 — Dataset & Evaluation Strategy
Markdown cell explaining:
- How the validation split was used
- How the test set was kept separate
- Which metrics are appropriate for the dataset and why
A.2 — Training & Validation Analysis
For each experiment:
- Load training logs
- Generate and save:
- Training vs. validation loss curve (PNG)
- Training vs. validation accuracy or primary metric curve (PNG)
- Insert Markdown before each plot explaining:
- What the plot shows
- Why it matters for model evaluation
A.3 — Test Set Evaluation
For each experiment:
- Load test predictions and ground truth
- Compute:
- Confusion matrix (PNG)
- Per-class metrics table
- F1 score table (macro/micro/weighted)
- Include a clean printout of final test metrics:
- Accuracy
- Precision/Recall/F1
- Any dataset-specific metrics

5. Code Quality Requirements
Markdown Documentation
Before every major code cell, include a Markdown cell describing:
- What the code does
- How it fits into the modeling workflow
Readable, Commented Code
- Comment every major block
- Use clear function names
- Avoid magic numbers — define constants at the top of the cell
- Use modular code where appropriate
Reproducibility
- Notebook must run end‑to‑end without errors
- No hard‑coded absolute paths
- Use workspace‑relative paths
- Set random seeds for all frameworks used (NumPy, TensorFlow, PyTorch, etc.)

6. Plot Requirements
All plots must:
- Be readable
- Have axis labels, titles, legends
- Be saved as PNG files in the notebook‑named folder
- Be displayed inline in the notebook
- Be integrated into the narrative with Markdown explanations

7. Output Requirements
The final notebook must:
- Be polished and easy for an LF to follow
- Contain all required Markdown, plots, tables, and metrics
- Reflect the structure and conventions of the workspace
- Be created fresh each time the prompt is executed

8. Final Task
Generate the complete notebook file with all sections, Markdown, code, plots, and analysis.
Use the workspace’s experiment results to populate the notebook.


