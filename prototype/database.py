"""
Desktop Prototype SQLite Database Manager.
"""

import sqlite3
import os

DB_FILE = os.path.join(os.path.dirname(__file__), "fruit_prototype.db")


def init_prototype_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Users Table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL UNIQUE,
        password TEXT NOT NULL
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

    # Seed initial test user if none exists
    cursor.execute('SELECT COUNT(*) FROM users')
    if cursor.fetchone()[0] == 0:
        cursor.execute('INSERT INTO users (username, password) VALUES (?, ?)', ('user1', 'validPass'))
        cursor.execute('INSERT INTO users (username, password) VALUES (?, ?)', ('admin', 'admin123'))

    conn.commit()
    conn.close()


def authenticate(username, password):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('SELECT id, username FROM users WHERE username = ? AND password = ?', (username, password))
    user = cursor.fetchone()
    conn.close()
    return user


def save_image_record(user_id, image_path, tags):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('INSERT INTO images (user_id, image_path, tags) VALUES (?, ?, ?)', (user_id, image_path, tags))
    conn.commit()
    conn.close()
