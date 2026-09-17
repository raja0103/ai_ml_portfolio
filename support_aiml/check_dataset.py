import csv
from collections import Counter
from pathlib import Path

DATA_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "training_tickets.csv"
)

EXPECTED_COLUMNS = {
    "ticket_id",
    "subject",
    "description",
    "category",
}

ALLOWED_CATEGORIES = {
    "access_identity",
    "billing_subscription",
    "performance",
    "integration_sync",
    "product_defect",
}


def main():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    with DATA_PATH.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        reader = csv.DictReader(file)

        if set(reader.fieldnames or []) != EXPECTED_COLUMNS:
            raise ValueError(
                "CSV columns must be: "
                "ticket_id,subject,description,category"
            )

        rows = list(reader)

    if not rows:
        raise ValueError("Dataset is empty.")

    seen_ids = set()
    seen_texts = set()
    category_counts = Counter()

    for line_number, row in enumerate(rows, start=2):
        if None in row or any(
            value is None for value in row.values()
        ):
            raise ValueError(
                f"Line {line_number}: Incorrect number of CSV fields."
            )

        row = {
            key: value.strip()
            for key, value in row.items()
        }

        if any(not value for value in row.values()):
            raise ValueError(
                f"Line {line_number}: Empty value found."
            )

        ticket_id = row["ticket_id"]

        if ticket_id in seen_ids:
            raise ValueError(
                f"Line {line_number}: Duplicate ID {ticket_id}"
            )

        seen_ids.add(ticket_id)

        category = row["category"]

        if category not in ALLOWED_CATEGORIES:
            raise ValueError(
                f"Line {line_number}: Invalid category {category}"
            )

        combined_text = " ".join(
            (row["subject"] + " " + row["description"])
            .casefold()
            .split()
        )

        if combined_text in seen_texts:
            raise ValueError(
                f"Line {line_number}: Duplicate ticket text."
            )

        seen_texts.add(combined_text)
        category_counts[category] += 1

    missing_categories = (
        ALLOWED_CATEGORIES - set(category_counts)
    )

    if missing_categories:
        raise ValueError(
            f"Missing categories: {sorted(missing_categories)}"
        )

    print(f"Total examples: {len(rows)}")
    print("\nCategory counts:")

    for category, count in sorted(category_counts.items()):
        print(f"  {category}: {count}")

    print("\nChecks passed. No files were changed.")
    print("This checks structure, not whether labels are correct.")


if __name__ == "__main__":
    main()