import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "supportiq.db"

CATEGORIES = [
    "access_identity",
    "billing_subscription",
    "performance",
    "integration_sync",
    "product_defect",
]


def get_connection():
    connection = sqlite3.connect(DB_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database():
    """Create missing tables without deleting existing data."""
    connection = get_connection()

    try:
        with connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS tickets (
                    ticket_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject TEXT NOT NULL
                        CHECK (
                            length(trim(subject)) BETWEEN 1 AND 200
                        ),
                    description TEXT NOT NULL
                        CHECK (
                            length(trim(description)) BETWEEN 1 AND 5000
                        ),
                    status TEXT NOT NULL DEFAULT 'open'
                        CHECK (
                            status IN (
                                'open',
                                'in_progress',
                                'resolved',
                                'closed'
                            )
                        ),
                    created_at TEXT NOT NULL
                        DEFAULT CURRENT_TIMESTAMP
                )
            """)

            connection.execute("""
                CREATE TABLE IF NOT EXISTS ticket_reviews (
                    ticket_id INTEGER PRIMARY KEY,
                    reviewed_category TEXT NOT NULL
                        CHECK (
                            reviewed_category IN (
                                'access_identity',
                                'billing_subscription',
                                'performance',
                                'integration_sync',
                                'product_defect'
                            )
                        ),
                    reviewed_at TEXT NOT NULL
                        DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (ticket_id)
                        REFERENCES tickets(ticket_id)
                )
            """)

            # Same structure used by the synthetic importer.
            connection.execute("""
                CREATE TABLE IF NOT EXISTS synthetic_ticket_metadata (
                    source_id TEXT PRIMARY KEY,
                    ticket_id INTEGER NOT NULL UNIQUE
                        REFERENCES tickets(ticket_id),
                    suggested_category TEXT NOT NULL,
                    scenario_group TEXT NOT NULL,
                    source_type TEXT NOT NULL
                )
            """)
    finally:
        connection.close()


def create_ticket(subject, description):
    subject = subject.strip()
    description = description.strip()

    if not 1 <= len(subject) <= 200:
        raise ValueError("Subject must contain 1–200 characters.")

    if not 1 <= len(description) <= 5000:
        raise ValueError(
            "Description must contain 1–5000 characters."
        )

    connection = get_connection()

    try:
        with connection:
            cursor = connection.execute(
                """
                INSERT INTO tickets (subject, description)
                VALUES (?, ?)
                """,
                (subject, description),
            )
            ticket_id = cursor.lastrowid

        return ticket_id
    finally:
        connection.close()


def get_recent_tickets():
    """Latest 100 tickets, including provenance and both labels."""
    connection = get_connection()

    try:
        rows = connection.execute("""
            SELECT
                t.ticket_id,
                t.subject,
                t.description,
                t.status,
                t.created_at,
                CASE
                    WHEN m.ticket_id IS NULL THEN 'manual_entry'
                    ELSE m.source_type
                END AS source_type,
                m.suggested_category,
                r.reviewed_category
            FROM tickets AS t
            LEFT JOIN synthetic_ticket_metadata AS m
                ON m.ticket_id = t.ticket_id
            LEFT JOIN ticket_reviews AS r
                ON r.ticket_id = t.ticket_id
            ORDER BY t.ticket_id DESC
            LIMIT 100
        """).fetchall()

        return [dict(row) for row in rows]
    finally:
        connection.close()


def get_tickets_for_review(
    include_reviewed=False,
    source_filter="manual",
    ticket_id=None,
):
    """Filter before limiting results, so manual tickets stay accessible."""
    if source_filter not in {"manual", "synthetic", "all"}:
        raise ValueError("Invalid source filter.")

    connection = get_connection()

    try:
        rows = connection.execute(
            """
            SELECT
                t.ticket_id,
                t.subject,
                t.description,
                r.reviewed_category,
                CASE
                    WHEN m.ticket_id IS NULL THEN 'manual_entry'
                    ELSE m.source_type
                END AS source_type
            FROM tickets AS t
            LEFT JOIN ticket_reviews AS r
                ON r.ticket_id = t.ticket_id
            LEFT JOIN synthetic_ticket_metadata AS m
                ON m.ticket_id = t.ticket_id
            WHERE
                (? = 1 OR r.ticket_id IS NULL)
                AND (
                    ? = 'all'
                    OR (? = 'manual' AND m.ticket_id IS NULL)
                    OR (? = 'synthetic' AND m.ticket_id IS NOT NULL)
                )
                AND (? IS NULL OR t.ticket_id = ?)
            ORDER BY t.ticket_id DESC
            LIMIT 100
            """,
            (
                int(include_reviewed),
                source_filter,
                source_filter,
                source_filter,
                ticket_id,
                ticket_id,
            ),
        ).fetchall()

        return [dict(row) for row in rows]
    finally:
        connection.close()


def save_review(ticket_id, category):
    if category not in CATEGORIES:
        raise ValueError("Please select a valid category.")

    connection = get_connection()

    try:
        with connection:
            ticket = connection.execute(
                "SELECT ticket_id FROM tickets WHERE ticket_id = ?",
                (ticket_id,),
            ).fetchone()

            if ticket is None:
                raise ValueError("Ticket not found.")

            connection.execute(
                """
                INSERT INTO ticket_reviews (
                    ticket_id,
                    reviewed_category
                )
                VALUES (?, ?)
                ON CONFLICT(ticket_id) DO UPDATE SET
                    reviewed_category = excluded.reviewed_category,
                    reviewed_at = CURRENT_TIMESTAMP
                """,
                (ticket_id, category),
            )
    finally:
        connection.close()


def get_reviewed_tickets():
    connection = get_connection()

    try:
        rows = connection.execute("""
            SELECT
                t.ticket_id,
                t.subject,
                t.description,
                r.reviewed_category,
                r.reviewed_at,
                CASE
                    WHEN m.ticket_id IS NULL THEN 'manual_entry'
                    ELSE m.source_type
                END AS source_type
            FROM tickets AS t
            JOIN ticket_reviews AS r
                ON r.ticket_id = t.ticket_id
            LEFT JOIN synthetic_ticket_metadata AS m
                ON m.ticket_id = t.ticket_id
            ORDER BY r.reviewed_at DESC, t.ticket_id DESC
            LIMIT 100
        """).fetchall()

        return [dict(row) for row in rows]
    finally:
        connection.close()


def get_dataset_summary():
    connection = get_connection()

    try:
        total = connection.execute(
            "SELECT COUNT(*) FROM tickets"
        ).fetchone()[0]

        reviewed = connection.execute(
            "SELECT COUNT(*) FROM ticket_reviews"
        ).fetchone()[0]

        synthetic = connection.execute(
            "SELECT COUNT(*) FROM synthetic_ticket_metadata"
        ).fetchone()[0]

        return {
            "total": total,
            "reviewed": reviewed,
            "pending": total - reviewed,
            "synthetic": synthetic,
            "manual": total - synthetic,
        }
    finally:
        connection.close()


def get_category_counts():
    """Keep synthetic suggestions separate from human review counts."""
    connection = get_connection()

    try:
        synthetic_rows = connection.execute("""
            SELECT suggested_category, COUNT(*) AS count
            FROM synthetic_ticket_metadata
            GROUP BY suggested_category
        """).fetchall()

        review_rows = connection.execute("""
            SELECT reviewed_category, COUNT(*) AS count
            FROM ticket_reviews
            GROUP BY reviewed_category
        """).fetchall()

        synthetic_counts = {
            row["suggested_category"]: row["count"]
            for row in synthetic_rows
        }

        review_counts = {
            row["reviewed_category"]: row["count"]
            for row in review_rows
        }

        return [
            {
                "category": category,
                "synthetic_examples": synthetic_counts.get(category, 0),
                "human_reviewed_examples": review_counts.get(category, 0),
            }
            for category in CATEGORIES
        ]
    finally:
        connection.close()


def get_training_data():
    """All human-reviewed rows with source metadata preserved."""
    connection = get_connection()

    try:
        rows = connection.execute("""
            SELECT
                t.ticket_id,
                t.subject,
                t.description,
                r.reviewed_category,
                CASE
                    WHEN m.ticket_id IS NULL THEN 'manual_entry'
                    ELSE m.source_type
                END AS source_type,
                m.scenario_group
            FROM tickets AS t
            JOIN ticket_reviews AS r
                ON r.ticket_id = t.ticket_id
            LEFT JOIN synthetic_ticket_metadata AS m
                ON m.ticket_id = t.ticket_id
            ORDER BY t.ticket_id
        """).fetchall()

        return [dict(row) for row in rows]
    finally:
        connection.close()


def get_synthetic_training_data():
    """Synthetic practice rows, not human-reviewed ground truth."""
    connection = get_connection()

    try:
        rows = connection.execute("""
            SELECT
                t.ticket_id,
                t.subject,
                t.description,
                m.suggested_category,
                m.scenario_group,
                m.source_type
            FROM tickets AS t
            JOIN synthetic_ticket_metadata AS m
                ON m.ticket_id = t.ticket_id
            ORDER BY t.ticket_id
        """).fetchall()

        return [dict(row) for row in rows]
    finally:
        connection.close()