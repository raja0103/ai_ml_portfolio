# SupportIQ — Support Ticket Classification Portfolio

A Python portfolio project combining a Streamlit ticket collection and human-review interface, SQLite storage, and a separate machine-learning training pipeline.

**Current status:** Ticket submission, manual category review, database summaries, synthetic ticket import, and offline model training are implemented. The Streamlit app does **not yet load the model or predict categories**.

## Business problem

Support teams need consistently categorized tickets before they can evaluate automated routing. This project collects ticket text, records human labels separately from generated labels, and trains a baseline classifier for five issue categories.

## Categories

| Category | Meaning |
| --- | --- |
| access_identity | Login, passwords, authentication, access |
| billing_subscription | Payments, invoices, subscriptions |
| performance | Tasks work but take unusually long |
| integration_sync | External connections and data synchronization |
| product_defect | Features fail or produce incorrect results |

## Implemented features

- Submit a subject and description with input validation.
- Review tickets by source and ticket ID; correct existing human labels.
- Display recent tickets, human reviews, and dataset counts.
- Store tickets, reviews, and synthetic provenance in separate SQLite tables.
- Import synthetic tickets with a database backup and duplicate-source skipping.
- Train TF-IDF plus Logistic Regression and compare with a majority-class baseline.
- Save validation metrics, a confusion matrix, per-category results, and up to 20 errors.

## Repository files

This README belongs at the repository root. Put `requirements.txt` inside `support_aiml/`, beside `app.py`.

| Path | Purpose |
| --- | --- |
| support_aiml/app.py | Streamlit ticket submission and human-review interface |
| support_aiml/db.py | SQLite schema, queries, validation, and review storage |
| support_aiml/import_synthetic_tickets.py | Import the supplied synthetic SQLite dataset |
| support_aiml/train_model.py | Train and evaluate the text classifier |
| support_aiml/synthetic_tickets.sqlite | Required source database for synthetic import |
| support_aiml/supportiq.db | Working database; created by initialization if absent |
| support_aiml/models/ticket_classifier.joblib | Output of training |
| support_aiml/reports/synthetic_baseline_metrics.json | Output of validation |
| support_aiml/requirements.txt | Direct Python dependencies |

The repository also contains `data/` and `check_dataset.py`. The supplied training script reads SQLite, not `data/training_tickets.csv`. The dataset-check script was not included in this code review.

## Setup on Windows

Install Python with pip. Download this repository using **Code → Download ZIP**, extract it, and open the extracted folder in VS Code. In the terminal, enter the project directory:

```powershell
cd support_aiml
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

These commands use the virtual environment directly, so PowerShell activation is unnecessary. Dependencies are intentionally unpinned because the original environment versions were not supplied. For reproducible model loading, record the versions from a verified working environment and use the same versions for training and inference.

## Run the interface

From `support_aiml/`:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed in the terminal. The app initializes missing tables without deleting existing data. Use **Submit Ticket**, **Review Tickets**, **Saved Data**, and **Training Data** to explore the implemented workflow. Stop the server with **Ctrl+C**.

The interface works without the synthetic source database; importing and training require synthetic data.

## Import and train

Run these commands from `support_aiml/`. Ensure the supplied `synthetic_tickets.sqlite` is beside the Python scripts.

```powershell
.\.venv\Scripts\python.exe -c "from db import initialize_database; initialize_database()"
.\.venv\Scripts\python.exe import_synthetic_tickets.py
.\.venv\Scripts\python.exe train_model.py
```

The importer expects exactly **50,000 source rows** in a `synthetic_tickets` table with `source_id`, `subject`, `description`, `suggested_category`, `scenario_group`, and `source_type`. This is a code requirement, not an independently verified count of the uploaded database.

Import creates a timestamped backup of `supportiq.db`, skips previously imported source IDs, and does not create human reviews. Training uses generated suggested labels even when human reviews exist; it does not train on corrected human labels.

Training writes or replaces:

- `models/ticket_classifier.joblib`
- `reports/synthetic_baseline_metrics.json`

It does not modify the database or automatically connect predictions to the app.

## Model and evaluation

Input is the concatenation of subject and description. Scenario IDs and labels are excluded from text features.

- Split: 80% training / 20% validation at the scenario-group level, stratified by category, with random seed 42.
- TF-IDF: unigrams and bigrams, lowercase normalization, sublinear term frequency, up to 30,000 features.
- Classifier: Logistic Regression, maximum 1,000 optimization iterations.
- Reference: most-frequent DummyClassifier.
- Metrics: model accuracy, macro F1, category precision/recall/F1, confusion matrix, and baseline macro F1.

All variants within a scenario stay together. TF-IDF is fitted only on training text. The saved model is the evaluated model, without refitting on validation data.

**Results:** No numerical scores are claimed here because the actual dataset, saved model, and evaluation report were not supplied for verification. Run training and inspect the generated report.

## Limitations and next steps

This is a practice baseline using template-generated synthetic data. The code describes 50 underlying scenarios, so many rows are correlated variations rather than independent real-world examples. There is no separate final test set, hyperparameter search, production deployment, authentication, or monitoring implementation in the supplied files.

Next steps are to connect model inference to the interface, evaluate independently reviewed examples, introduce an untouched final test set, and document a verified dependency environment. Synthetic validation metrics alone do not establish production readiness.

## Data handling

Publish only synthetic or otherwise shareable data. Inspect database contents before publishing, since the working database can contain manually entered tickets. Keep credentials, virtual environments, caches, and database backups out of uploads. Load serialized model files only from trusted sources.

## Verification scope

Documentation was checked against the four supplied Python files. Syntax and a temporary-database workflow were checked separately. The full Streamlit application and model training were not executed against the repository dataset.
