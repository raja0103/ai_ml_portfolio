"""Run beside app.py. Imports synthetic tickets without fabricating human reviews."""
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

BASE = Path(__file__).resolve().parent
SOURCE = BASE / 'synthetic_tickets.sqlite'
TARGET = BASE / 'supportiq.db'

def main():
    if not SOURCE.is_file() or not TARGET.is_file():
        raise SystemExit('Place this script and synthetic_tickets.sqlite beside your existing supportiq.db. Run the app first if its database does not exist.')
    src = sqlite3.connect(SOURCE.as_uri() + '?mode=ro', uri=True)
    try:
        rows = src.execute('SELECT source_id, subject, description, suggested_category, scenario_group, source_type FROM synthetic_tickets ORDER BY source_id').fetchall()
    finally:
        src.close()
    if len(rows) != 50000:
        raise SystemExit('Unexpected source row count. Import cancelled.')
    con = sqlite3.connect(TARGET, timeout=30)
    try:
        con.execute('PRAGMA foreign_keys = ON')
        columns = {r[1] for r in con.execute('PRAGMA table_info(tickets)')}
        if not {'ticket_id','subject','description','status','created_at'}.issubset(columns):
            raise SystemExit('Expected SupportIQ tickets table not found. Import cancelled.')
        backup = BASE / ('supportiq_backup_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.db')
        dest = sqlite3.connect(backup)
        try:
            con.backup(dest)
        finally:
            dest.close()
        added = 0
        with con:
            con.execute('''CREATE TABLE IF NOT EXISTS synthetic_ticket_metadata (
                source_id TEXT PRIMARY KEY,
                ticket_id INTEGER NOT NULL UNIQUE REFERENCES tickets(ticket_id),
                suggested_category TEXT NOT NULL,
                scenario_group TEXT NOT NULL,
                source_type TEXT NOT NULL
            )''')
            existing = {r[0] for r in con.execute('SELECT source_id FROM synthetic_ticket_metadata')}
            for source_id, subject, description, category, group, source_type in rows:
                if source_id in existing:
                    continue
                cur = con.execute('INSERT INTO tickets(subject,description) VALUES (?,?)',(subject,description))
                con.execute('INSERT INTO synthetic_ticket_metadata VALUES (?,?,?,?,?)',(source_id,cur.lastrowid,category,group,source_type))
                added += 1
        print('Backup:', backup.name)
        print('New tickets imported:', added)
        print('Total tickets:', con.execute('SELECT COUNT(*) FROM tickets').fetchone()[0])
        print('Human reviews were NOT created or changed.')
        print('Re-running this script skips previously imported source IDs.')
    finally:
        con.close()

if __name__ == '__main__':
    main()
