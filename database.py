import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "prediction_history.db"


def create_database():

    conn = sqlite3.connect(DB_PATH)

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            brand TEXT,
            ram INTEGER,
            storage INTEGER,
            battery INTEGER,
            camera INTEGER,
            screen REAL,
            predicted_price REAL,
            prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


def save_prediction(
    brand, ram, storage, battery,
    camera, screen, price
):

    conn = sqlite3.connect(DB_PATH)

    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO predictions
        (brand, ram, storage, battery,
         camera, screen, predicted_price)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        brand, ram, storage, battery,
        camera, screen, price
    ))

    conn.commit()
    conn.close()


def get_history():

    conn = sqlite3.connect(DB_PATH)

    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM predictions
        ORDER BY prediction_date DESC
    """)

    rows = cursor.fetchall()

    conn.close()

    return rows


create_database()