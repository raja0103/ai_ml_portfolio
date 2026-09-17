# ============================================================
# SUPPORTIQ: SYNTHETIC TICKET CLASSIFICATION BASELINE
# ============================================================
# Purpose:
# 1. Read synthetic tickets from SQLite.
# 2. Validate their text, labels, and scenario groups.
# 3. Split scenarios into training and validation sets.
# 4. Train a simple reference baseline.
# 5. Train a TF-IDF + Logistic Regression classifier.
# 6. Evaluate predictions on unseen scenarios.
# 7. Save the trained model and evaluation report.
#
# Important:
# - Labels come from the synthetic generator, not human reviews.
# - This script does not modify the database.
# - This script does not connect the model to the Streamlit app.
# - Results describe synthetic practice data, not production quality.
# ============================================================


# ------------------------------------------------------------
# STEP 1: IMPORT REQUIRED LIBRARIES
# ------------------------------------------------------------

# json saves evaluation results in a readable JSON file.
import json

# Path builds file paths relative to this script.
from pathlib import Path

# joblib saves the trained pipeline for later prediction.
import joblib

# DummyClassifier provides a simple reference model.
from sklearn.dummy import DummyClassifier

# TfidfVectorizer converts ticket text into numerical features.
from sklearn.feature_extraction.text import TfidfVectorizer

# LogisticRegression learns to classify those numerical features.
from sklearn.linear_model import LogisticRegression

# These functions measure classification performance.
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# train_test_split separates training and validation groups.
from sklearn.model_selection import train_test_split

# Pipeline keeps text preprocessing and classification together.
from sklearn.pipeline import Pipeline

# Import our category list and SQLite data-reading function.
from db import CATEGORIES, get_synthetic_training_data


# ------------------------------------------------------------
# STEP 2: DEFINE OUTPUT LOCATIONS
# ------------------------------------------------------------

# Use the folder containing this script as the project base.
BASE_DIR = Path(__file__).resolve().parent

# The trained model will be stored here.
MODEL_DIR = BASE_DIR / "models"

# Evaluation metrics and example errors will be stored here.
REPORT_DIR = BASE_DIR / "reports"


def main():

    # --------------------------------------------------------
    # STEP 3: READ SYNTHETIC TICKETS FROM SQLITE
    # --------------------------------------------------------
    # The database function joins:
    # - tickets
    # - synthetic_ticket_metadata
    #
    # Each returned row contains:
    # ticket_id, subject, description, suggested_category,
    # scenario_group, and source_type.
    #
    # We are not reading the earlier CSV file.
    # We are not treating generated labels as human reviews.

    print("Reading synthetic tickets from SQLite...")

    rows = get_synthetic_training_data()

    if not rows:
        raise ValueError("No synthetic tickets found.")


    # --------------------------------------------------------
    # STEP 4: VALIDATE DATA BEFORE TRAINING
    # --------------------------------------------------------
    # Check:
    # - Every label belongs to one of our five categories.
    # - Every row has a scenario group.
    # - Subject and description are not blank.
    # - Each scenario group maps to exactly one category.
    #
    # These checks validate structure and consistency.
    # They do not prove that every generated label is correct.

    group_labels = {}

    for row in rows:
        category = row["suggested_category"]
        group = row["scenario_group"]

        if category not in CATEGORIES:
            raise ValueError(
                f"Unexpected category: {category}"
            )

        if not group:
            raise ValueError(
                "A ticket has no scenario_group."
            )

        if (
            not row["subject"].strip()
            or not row["description"].strip()
        ):
            raise ValueError("Empty ticket text found.")

        # Multiple variations can share one scenario group.
        # All variations in that group must have the same label.
        if (
            group in group_labels
            and group_labels[group] != category
        ):
            raise ValueError(
                f"Conflicting labels for group: {group}"
            )

        group_labels[group] = category


    # --------------------------------------------------------
    # STEP 5: PREPARE UNIQUE SCENARIO GROUPS
    # --------------------------------------------------------
    # The generated dataset has:
    # - 5 categories
    # - 10 base scenarios per category
    # - 50 scenario groups in total
    #
    # Many tickets are contextual variations of the same issue.
    # Therefore, we split scenario groups instead of ticket rows.

    groups = sorted(group_labels)

    labels_for_groups = [
        group_labels[group]
        for group in groups
    ]

    if set(labels_for_groups) != set(CATEGORIES):
        raise ValueError(
            "The dataset must contain all five categories."
        )


    # --------------------------------------------------------
    # STEP 6: SPLIT SCENARIOS INTO TRAINING AND VALIDATION
    # --------------------------------------------------------
    # test_size=0.20:
    # Reserve 20% of scenario groups for validation.
    #
    # random_state=42:
    # Make the split reproducible for the same input groups.
    #
    # stratify=labels_for_groups:
    # Preserve category representation across both sets.
    #
    # For the supplied dataset:
    # - Training: 40 scenarios, approximately 40,000 tickets
    # - Validation: 10 scenarios, approximately 10,000 tickets
    #
    # A scenario and all of its variations stay on one side.
    # This reduces leakage caused by near-identical examples.
    # It does not remove all limitations of synthetic data.

    train_groups, validation_groups = train_test_split(
        groups,
        test_size=0.20,
        random_state=42,
        stratify=labels_for_groups,
    )

    # Sets make membership checks efficient.
    train_groups = set(train_groups)
    validation_groups = set(validation_groups)

    # Confirm that no scenario appears in both sets.
    assert train_groups.isdisjoint(validation_groups)


    # --------------------------------------------------------
    # STEP 7: ASSIGN TICKET ROWS TO THEIR SCENARIO SPLITS
    # --------------------------------------------------------

    train_rows = [
        row
        for row in rows
        if row["scenario_group"] in train_groups
    ]

    validation_rows = [
        row
        for row in rows
        if row["scenario_group"] in validation_groups
    ]


    # --------------------------------------------------------
    # STEP 8: CREATE MODEL INPUTS AND TARGET LABELS
    # --------------------------------------------------------
    # X = ticket text supplied to the model.
    # y = category the model is expected to predict.
    #
    # Combine subject and description into one text input.
    # Do not include category names, scenario IDs, or source IDs
    # as input features. Those would expose label-related metadata.

    def ticket_text(row):
        return (
            row["subject"].strip()
            + " "
            + row["description"].strip()
        )

    x_train = [
        ticket_text(row)
        for row in train_rows
    ]

    y_train = [
        row["suggested_category"]
        for row in train_rows
    ]

    x_validation = [
        ticket_text(row)
        for row in validation_rows
    ]

    y_validation = [
        row["suggested_category"]
        for row in validation_rows
    ]

    print(f"Total tickets:        {len(rows)}")
    print(f"Training tickets:     {len(train_rows)}")
    print(f"Validation tickets:   {len(validation_rows)}")
    print(f"Training scenarios:   {len(train_groups)}")
    print(f"Validation scenarios: {len(validation_groups)}")
    print("Shared scenarios:     0")


    # --------------------------------------------------------
    # STEP 9: TRAIN A SIMPLE REFERENCE BASELINE
    # --------------------------------------------------------
    # A most-frequent baseline always predicts one class.
    # It does not learn ticket language.
    #
    # Our categories are balanced, so class counts are tied.
    # The baseline still selects one class and predicts it
    # for every validation example.
    #
    # This provides a reference for evaluating the text model.
    # Constant input values are sufficient because this
    # baseline ignores the input features.

    print("\nTraining majority-class baseline...")

    baseline = DummyClassifier(
        strategy="most_frequent"
    )

    baseline.fit(
        [[0]] * len(y_train),
        y_train,
    )

    baseline_predictions = baseline.predict(
        [[0]] * len(y_validation)
    )


    # --------------------------------------------------------
    # STEP 10: DEFINE THE TEXT CLASSIFICATION PIPELINE
    # --------------------------------------------------------
    # Stage A: TF-IDF
    # - Converts text into sparse numerical feature vectors.
    # - Gives weights to terms based on their frequency.
    # - Learns vocabulary and inverse document frequencies.
    #
    # lowercase=True:
    # Treat uppercase and lowercase forms consistently.
    #
    # ngram_range=(1, 2):
    # Use individual words and two-word sequences.
    # Examples: "payment", "subscription", "reset link".
    #
    # sublinear_tf=True:
    # Apply logarithmic scaling to term frequency.
    #
    # max_features=30000:
    # Limit vocabulary size to at most 30,000 features.
    #
    # Stage B: Logistic Regression
    # - Learns weights relating text features to categories.
    # - Produces a classification among the known categories.
    #
    # max_iter=1000:
    # Sets the maximum number of optimization iterations.
    # This is not the number of tickets or training epochs.

    print("Training TF-IDF + Logistic Regression...")

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                sublinear_tf=True,
                max_features=30000,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42,
            ),
        ),
    ])


    # --------------------------------------------------------
    # STEP 11: FIT THE PIPELINE USING TRAINING DATA ONLY
    # --------------------------------------------------------
    # During fit():
    # 1. TF-IDF learns its vocabulary from x_train.
    # 2. TF-IDF transforms x_train into numerical features.
    # 3. Logistic Regression learns from features and y_train.
    #
    # Validation text is not used to fit the vocabulary,
    # feature weights, or classifier.

    model.fit(x_train, y_train)


    # --------------------------------------------------------
    # STEP 12: PREDICT VALIDATION TICKET CATEGORIES
    # --------------------------------------------------------
    # During predict():
    # 1. The fitted TF-IDF transforms validation text.
    # 2. The fitted classifier predicts categories.
    #
    # The model is not retrained during prediction.

    predictions = model.predict(x_validation)


    # --------------------------------------------------------
    # STEP 13: CALCULATE EVALUATION METRICS
    # --------------------------------------------------------
    # Accuracy:
    # Fraction of validation predictions that are correct.
    #
    # Precision for a category:
    # Of tickets predicted as that category, how many are correct?
    #
    # Recall for a category:
    # Of actual tickets in that category, how many were found?
    #
    # F1:
    # Harmonic mean of precision and recall.
    #
    # Macro F1:
    # Average category F1, giving each category equal weight.
    #
    # zero_division=0:
    # Return zero where a metric has an undefined denominator.

    baseline_macro_f1 = f1_score(
        y_validation,
        baseline_predictions,
        labels=CATEGORIES,
        average="macro",
        zero_division=0,
    )

    accuracy = accuracy_score(
        y_validation,
        predictions,
    )

    macro_f1 = f1_score(
        y_validation,
        predictions,
        labels=CATEGORIES,
        average="macro",
        zero_division=0,
    )

    print("\n========== VALIDATION RESULTS ==========")
    print(f"Baseline Macro F1: {baseline_macro_f1:.4f}")
    print(f"Model Accuracy:    {accuracy:.4f}")
    print(f"Model Macro F1:    {macro_f1:.4f}")

    print("\nPer-category results:")

    print(classification_report(
        y_validation,
        predictions,
        labels=CATEGORIES,
        zero_division=0,
    ))


    # --------------------------------------------------------
    # STEP 14: COLLECT EXAMPLE MISTAKES FOR ERROR ANALYSIS
    # --------------------------------------------------------
    # Save up to 20 incorrect predictions.
    # These examples help investigate what the model confuses.
    #
    # They are the first errors encountered, not a representative
    # sample of every failure type.

    error_examples = []

    for row, predicted in zip(
        validation_rows,
        predictions,
    ):
        if predicted != row["suggested_category"]:
            error_examples.append({
                "ticket_id": row["ticket_id"],
                "subject": row["subject"],
                "actual_synthetic_label": (
                    row["suggested_category"]
                ),
                "predicted_category": str(predicted),
                "scenario_group": row["scenario_group"],
            })

            if len(error_examples) == 20:
                break


    # --------------------------------------------------------
    # STEP 15: CREATE OUTPUT FOLDERS
    # --------------------------------------------------------
    # exist_ok=True allows these folders to already exist.

    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)


    # --------------------------------------------------------
    # STEP 16: SAVE THE TRAINED PIPELINE
    # --------------------------------------------------------
    # Save both TF-IDF and Logistic Regression together.
    # This preserves preprocessing consistency for future inputs.
    #
    # Save exactly the model evaluated above.
    # Do not refit it on validation data before saving.
    #
    # Re-running the script replaces this model file.
    # Only load joblib model files from trusted sources.

    model_path = (
        MODEL_DIR / "ticket_classifier.joblib"
    )

    joblib.dump(model, model_path)


    # --------------------------------------------------------
    # STEP 17: BUILD A STRUCTURED EVALUATION REPORT
    # --------------------------------------------------------
    # Record:
    # - Data source and label provenance
    # - Split settings and scenario IDs
    # - Baseline and model metrics
    # - Confusion matrix
    # - Per-category metrics
    # - Example mistakes
    # - Known evaluation limitations
    #
    # Confusion matrix:
    # Rows = actual categories.
    # Columns = predicted categories.
    # Both follow the saved category_order.

    report = {
        "source": "SQLite synthetic tickets",
        "label_type": "generated suggested labels",
        "split_method": "stratified scenario-group holdout",
        "random_state": 42,
        "training_tickets": len(train_rows),
        "validation_tickets": len(validation_rows),
        "training_groups": sorted(train_groups),
        "validation_groups": sorted(validation_groups),
        "baseline_macro_f1": baseline_macro_f1,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "category_order": CATEGORIES,

        "confusion_matrix": confusion_matrix(
            y_validation,
            predictions,
            labels=CATEGORIES,
        ).tolist(),

        "classification_report": classification_report(
            y_validation,
            predictions,
            labels=CATEGORIES,
            output_dict=True,
            zero_division=0,
        ),

        "first_20_errors": error_examples,

        "limitations": [
            "Template-generated synthetic data, not real customer data.",
            "Only 50 underlying scenarios in the complete dataset.",
            "Validation rows are correlated variations of 10 scenarios.",
            "No separate final test set in this practice baseline.",
            "Metrics do not establish production readiness.",
        ],
    }


    # --------------------------------------------------------
    # STEP 18: SAVE THE EVALUATION REPORT
    # --------------------------------------------------------
    # JSON makes results readable and reusable by other scripts.
    # Re-running the script replaces this report file.

    report_path = (
        REPORT_DIR / "synthetic_baseline_metrics.json"
    )

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"\nModel saved:  {model_path}")
    print(f"Report saved: {report_path}")

    print(
        "\nPractice training completed. "
        "Database was not modified."
    )


# ------------------------------------------------------------
# STEP 19: RUN ONLY WHEN THIS FILE IS EXECUTED DIRECTLY
# ------------------------------------------------------------
# Running "python train_model.py" calls main().
# Importing this file from another script does not start training.

if __name__ == "__main__":
    main()