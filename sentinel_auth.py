import sqlite3
import bcrypt
import os

DB_NAME = "sentinel_users.db"

def init_user_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash BLOB
        )
    ''')
    conn.commit()
    conn.close()

def create_user(username, password):
    """Returns True if successful, False if username exists."""
    init_user_db()
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # Check if user exists
    c.execute("SELECT username FROM users WHERE username=?", (username,))
    if c.fetchone():
        conn.close()
        return False
    
    # Hash password
    # salt = bcrypt.gensalt()
    # hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    
    # We need to run this command to install bcrypt first, but for the code content:
    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    
    c.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, hashed))
    conn.commit()
    conn.close()
    return True

def verify_user(username, password):
    """Returns True if credentials are valid."""
    init_user_db()
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    c.execute("SELECT password_hash FROM users WHERE username=?", (username,))
    result = c.fetchone()
    conn.close()
    
    if result:
        stored_hash = result[0]
        return bcrypt.checkpw(password.encode('utf-8'), stored_hash)
    return False
