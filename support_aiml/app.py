import logging
import sqlite3

import streamlit as st

from db import (
    CATEGORIES,
    initialize_database,
    create_ticket,
    get_recent_tickets,
    get_tickets_for_review,
    save_review,
    get_reviewed_tickets,
    get_dataset_summary,
    get_category_counts,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="SupportIQ",
    page_icon="🎫",
    layout="wide",
)

st.title("🎫 SupportIQ")
st.caption("Customer Support Intelligence Platform")

try:
    initialize_database()
except sqlite3.Error:
    logger.exception("Database initialization failed")
    st.error(
        "Database could not be initialized. "
        "Check the VS Code terminal for details."
    )
    st.stop()

if "success_message" in st.session_state:
    st.success(st.session_state.pop("success_message"))

if st.button("Refresh data"):
    st.rerun()

try:
    summary = get_dataset_summary()
except sqlite3.Error:
    logger.exception("Could not load summary")
    st.error("Database summary could not be loaded.")
    st.stop()

col1, col2, col3 = st.columns(3)

col1.metric("Total tickets", summary["total"])
col2.metric("Human-reviewed tickets", summary["reviewed"])
col3.metric("Not human-reviewed", summary["pending"])

st.caption(
    f"Manually entered: {summary['manual']:,} | "
    f"Synthetic imports: {summary['synthetic']:,}"
)

submit_tab, review_tab, saved_tab, training_tab = st.tabs([
    "Submit Ticket",
    "Review Tickets",
    "Saved Data",
    "Training Data",
])

# --------------------------------------------------
# SUBMIT TICKET
# --------------------------------------------------
with submit_tab:
    st.subheader("Create a ticket")

    if "ticket_form_number" not in st.session_state:
        st.session_state["ticket_form_number"] = 0

    form_number = st.session_state["ticket_form_number"]

    with st.form(f"ticket_form_{form_number}"):
        subject = st.text_input(
            "Ticket subject",
            placeholder="Example: Subscription not activated",
            max_chars=200,
            key=f"subject_{form_number}",
        )

        description = st.text_area(
            "Issue description",
            placeholder=(
                "My payment succeeded, "
                "but my subscription is still inactive."
            ),
            height=150,
            max_chars=5000,
            key=f"description_{form_number}",
        )

        submitted = st.form_submit_button("Save Ticket")

    if submitted:
        try:
            ticket_id = create_ticket(subject, description)
        except ValueError as error:
            st.warning(str(error))
        except sqlite3.Error:
            logger.exception("Ticket save failed")
            st.error("Ticket could not be saved.")
        else:
            st.session_state["success_message"] = (
                f"Ticket #{ticket_id} saved successfully."
            )
            st.session_state["ticket_form_number"] += 1
            st.rerun()

# --------------------------------------------------
# REVIEW TICKETS
# --------------------------------------------------
with review_tab:
    st.subheader("Assign or correct a category")

    with st.expander("Category guide", expanded=True):
        st.markdown("""
- **access_identity:** Login, password, authentication, access.
- **billing_subscription:** Payments, invoices, subscriptions.
- **performance:** A task works but takes unusually long.
- **integration_sync:** External connections or data synchronization.
- **product_defect:** A feature fails or produces incorrect results.
        """)

    st.info(
        "Read the issue before assigning a label. "
        "Leave unclear or unrelated tickets unreviewed."
    )

    source_options = {
        "Manually entered tickets": "manual",
        "Synthetic imported tickets": "synthetic",
        "All tickets": "all",
    }

    source_name = st.selectbox(
        "Ticket source",
        options=list(source_options),
    )

    include_reviewed = st.checkbox(
        "Include reviewed tickets to correct their labels"
    )

    ticket_id_text = st.text_input(
        "Find a specific ticket ID (optional)",
        placeholder="Example: 1",
    ).strip()

    selected_ticket_filter = None
    valid_filter = True

    if ticket_id_text:
        try:
            selected_ticket_filter = int(ticket_id_text)

            if selected_ticket_filter <= 0:
                raise ValueError
        except ValueError:
            valid_filter = False
            st.warning("Enter a positive whole-number ticket ID.")

    if valid_filter:
        try:
            review_tickets = get_tickets_for_review(
                include_reviewed=include_reviewed,
                source_filter=source_options[source_name],
                ticket_id=selected_ticket_filter,
            )
        except sqlite3.Error:
            logger.exception("Could not load review tickets")
            st.error("Tickets could not be loaded.")
            review_tickets = None

        if review_tickets is not None:
            if not review_tickets:
                st.info(
                    "No tickets match these filters. "
                    "Try All tickets or include reviewed tickets."
                )
            else:
                st.caption(
                    "Showing up to 100 latest matching tickets. "
                    "Use an exact ticket ID to find older records."
                )

                tickets_by_id = {
                    ticket["ticket_id"]: ticket
                    for ticket in review_tickets
                }

                selected_id = st.selectbox(
                    "Select a ticket",
                    options=list(tickets_by_id),
                    format_func=lambda ticket_id: (
                        f"#{ticket_id} — "
                        f"{tickets_by_id[ticket_id]['subject']}"
                    ),
                )

                ticket = tickets_by_id[selected_id]

                st.write("**Subject**")
                st.write(ticket["subject"])

                st.write("**Description**")
                st.write(ticket["description"])

                st.caption(f"Source: {ticket['source_type']}")

                current_category = ticket["reviewed_category"]

                st.write(
                    "**Current human-reviewed category:**",
                    current_category or "Not reviewed",
                )

                category_options = [
                    "-- Select category --"
                ] + CATEGORIES

                default_index = (
                    category_options.index(current_category)
                    if current_category in CATEGORIES
                    else 0
                )

                form_key = (
                    f"review_{selected_id}_"
                    f"{current_category or 'unreviewed'}"
                )

                with st.form(form_key):
                    category = st.selectbox(
                        "Correct category",
                        options=category_options,
                        index=default_index,
                    )

                    review_submitted = st.form_submit_button(
                        "Save Review"
                    )

                if review_submitted:
                    try:
                        save_review(selected_id, category)
                    except ValueError as error:
                        st.warning(str(error))
                    except sqlite3.Error:
                        logger.exception("Review save failed")
                        st.error("Review could not be saved.")
                    else:
                        st.session_state["success_message"] = (
                            f"Ticket #{selected_id} reviewed as "
                            f"{category}."
                        )
                        st.rerun()

# --------------------------------------------------
# SAVED DATA
# --------------------------------------------------
with saved_tab:
    st.subheader("Saved tickets and reviews")

    try:
        tickets = get_recent_tickets()
        reviewed_tickets = get_reviewed_tickets()
    except sqlite3.Error:
        logger.exception("Could not load saved data")
        st.error("Saved data could not be loaded.")
    else:
        st.write("**Latest 100 tickets**")

        if tickets:
            st.dataframe(tickets, hide_index=True)
        else:
            st.info("No tickets saved.")

        st.write("**Latest 100 human-reviewed tickets**")

        if reviewed_tickets:
            st.dataframe(reviewed_tickets, hide_index=True)
        else:
            st.info("No reviews saved.")

    st.caption(
        "Lists are limited to 100 rows for display. "
        "Top-level counts include all tickets. Timestamps are UTC."
    )

# --------------------------------------------------
# TRAINING DATA
# --------------------------------------------------
with training_tab:
    st.subheader("Category counts")

    st.write(
        "Synthetic suggested labels and human-reviewed labels "
        "are counted separately."
    )

    try:
        category_counts = get_category_counts()
    except sqlite3.Error:
        logger.exception("Could not load category counts")
        st.error("Category counts could not be loaded.")
    else:
        st.dataframe(category_counts, hide_index=True)

        left, right = st.columns(2)

        left.metric(
            "Synthetic examples",
            summary["synthetic"],
        )

        right.metric(
            "Human-reviewed examples",
            summary["reviewed"],
        )

    st.info(
        "Synthetic examples can support a clearly labeled practice "
        "baseline. They have generated suggested labels, not "
        "human-verified ground truth."
    )

    st.warning(
        "The imported dataset contains 10 base scenarios per category "
        "with contextual variations. Split by scenario_group for "
        "practice evaluation; do not randomly split near-identical "
        "variants across training and validation."
    )

    st.caption(
        "Reviewing a synthetic ticket does not make it real-world data. "
        "It remains synthetic and can appear in both count columns. "
        "Do not add the two columns to calculate unique tickets."
    )

    st.write(
        "This tab shows data availability only. "
        "Model training and AI prediction are not connected yet."
    )

st.divider()
st.caption("SupportIQ • SQLite storage • Human review • Synthetic practice data")