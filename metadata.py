import sqlite3
import os
from tabulate import tabulate

DB_PATH = "metadata.db"

def display_table():
    if not os.path.exists(DB_PATH):
        print("Database not found!")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Check if table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='photos_metadata';")
    if not cursor.fetchone():
        print("Table 'photos_metadata' does not exist.")
        conn.close()
        return

    # Fetch all rows
    cursor.execute("SELECT * FROM photos_metadata;")
    rows = cursor.fetchall()
    headers = [description[0] for description in cursor.description]

    if not rows:
        print("No data found in photos_metadata.")
    else:
        # Print table with geolocation
        print(tabulate(rows, headers=headers, tablefmt="grid"))

    conn.close()

if __name__ == "__main__":
    display_table()
