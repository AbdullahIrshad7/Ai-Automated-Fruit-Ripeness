"""
Database module for Fruit Ripeness Detection System.
Provides SQLite data access layer for user authentication,
image uploads, and prediction audit tracking.
"""

import sqlite3
import os
from datetime import datetime

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fruit_app.db")


def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Establish connection to SQLite database."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DEFAULT_DB_PATH):
    """Initialize database tables and seed test accounts."""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    # Users Table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL UNIQUE,
        password TEXT NOT NULL,
        role TEXT DEFAULT 'user',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # Images Table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        image_path TEXT NOT NULL,
        tags TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')

    # Predictions Audit Table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        fruit_name TEXT NOT NULL,
        ripeness TEXT NOT NULL,
        confidence REAL NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')

    # Seed Default Users if not present (as defined in test cases TC001, TC003)
    cursor.execute('SELECT COUNT(*) FROM users')
    if cursor.fetchone()[0] == 0:
        cursor.execute(
            'INSERT INTO users (username, password, role) VALUES (?, ?, ?)',
            ('user1', 'validPass', 'user')
        )
        cursor.execute(
            'INSERT INTO users (username, password, role) VALUES (?, ?, ?)',
            ('admin', 'admin123', 'admin')
        )

    conn.commit()
    conn.close()


def authenticate_user(username: str, password: str, db_path: str = DEFAULT_DB_PATH):
    """Authenticate username and password against users table."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT id, username, role FROM users WHERE username = ? AND password = ?', (username, password))
    user = cursor.fetchone()
    conn.close()
    if user:
        return dict(user)
    return None


def register_user(username: str, password: str, role: str = 'user', db_path: str = DEFAULT_DB_PATH) -> bool:
    """Register a new user account."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute('INSERT INTO users (username, password, role) VALUES (?, ?, ?)', (username, password, role))
        conn.commit()
        success = True
    except sqlite3.IntegrityError:
        success = False
    finally:
        conn.close()
    return success


def log_prediction(fruit_name: str, ripeness: str, confidence: float, user_id: int = 1, db_path: str = DEFAULT_DB_PATH):
    """Record prediction entry into SQLite database."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO predictions (user_id, fruit_name, ripeness, confidence) VALUES (?, ?, ?, ?)',
        (user_id, fruit_name, ripeness, confidence)
    )
    conn.commit()
    conn.close()


def get_recent_predictions(limit: int = 10, db_path: str = DEFAULT_DB_PATH) -> list:
    """Fetch recent prediction logs."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute('''
    SELECT p.id, u.username, p.fruit_name, p.ripeness, p.confidence, p.timestamp
    FROM predictions p
    LEFT JOIN users u ON p.user_id = u.id
    ORDER BY p.id DESC LIMIT ?
    ''', (limit,))
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_db_stats(db_path: str = DEFAULT_DB_PATH) -> dict:
    """Retrieve high level database record counts."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM users')
    total_users = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(*) FROM predictions')
    total_predictions = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(*) FROM images')
    total_images = cursor.fetchone()[0]
    conn.close()
    return {
        "users": total_users,
        "predictions": total_predictions,
        "images": total_images
    }
