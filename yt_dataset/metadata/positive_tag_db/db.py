import sqlite3
import csv
import os

DB_PATH = "yt_dataset/metadata/positive_tag_db/"

def create_tables():
    # Connect to the SQLite database (creates a new database if it doesn't exist)
    conn = sqlite3.connect(DB_PATH + 'positive_tags.db')

    # Create a cursor object to interact with the database
    cursor = conn.cursor()

    # Create table 1
    cursor.execute('''CREATE TABLE IF NOT EXISTS m01b82r (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        file_title TEXT NOT NULL,
                        yt_id TEXT NOT NULL
                    )''')

    # Create table 2
    cursor.execute('''CREATE TABLE IF NOT EXISTS m01j4z9 (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        file_title TEXT NOT NULL,
                        yt_id TEXT NOT NULL
                    )''')

    # Create table 3
    cursor.execute('''CREATE TABLE IF NOT EXISTS m0_ksk (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        file_title TEXT NOT NULL,
                        yt_id TEXT NOT NULL

                    )''')

    # Commit the changes and close the connection
    conn.commit()
    conn.close()

# Call the function to create the tables
# create_tables()

def write_to_database(table_name, yt_id):
    # Connect to the SQLite database
    conn = sqlite3.connect(DB_PATH + 'positive_tags.db')

    # Create a cursor object to interact with the database
    cursor = conn.cursor()

    # Insert the data into the table
    cursor.execute(f"INSERT INTO {table_name} (yt_id) VALUES (?)", (yt_id,))

    # Commit the changes and close the connection
    conn.commit()
    conn.close()

def export_sqlite_to_csv(db_path, table_name, csv_file):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {table_name}")
    rows = cursor.fetchall()
    column_names = [description[0] for description in cursor.description]
    if not os.path.isfile(csv_file):
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(column_names)
            writer.writerows(rows)
    else:
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(rows)
    conn.close()


export_sqlite_to_csv(db_path="yt_dataset/metadata/positive_tag_db/positive_tags.db", table_name="m01j4z9", csv_file="yt_dataset/metadata/positive_tag_db/m01j4z9.csv")
