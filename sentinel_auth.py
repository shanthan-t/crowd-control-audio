import pymongo
import bcrypt
import os

# --- Configuration ---
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "sentinel_auth"
COLLECTION_NAME = "users"

def get_collection():
    client = pymongo.MongoClient(MONGO_URI)
    db = client[DB_NAME]
    return db[COLLECTION_NAME]

def create_user(username, password):
    """Returns True if successful, False if username exists."""
    users = get_collection()
    
    # Check if user exists
    if users.find_one({"username": username}):
        return False
    
    # Hash password
    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    
    # Store only username and password
    user_doc = {
        "username": username,
        "password": hashed  # Storing hash as binary
    }
    
    users.insert_one(user_doc)
    return True

def verify_user(username, password):
    """Returns True if credentials are valid."""
    users = get_collection()
    
    user = users.find_one({"username": username})
    
    if user:
        stored_hash = user['password']
        return bcrypt.checkpw(password.encode('utf-8'), stored_hash)
    
    return False
