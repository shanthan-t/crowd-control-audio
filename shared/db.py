import pymongo
import datetime
import threading

# --- Configuration ---
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME_LOGS = "sentinel_logs"
COLLECTION_LOGS = "system_events"

class Database:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(Database, cls).__new__(cls)
                    cls._instance.client = pymongo.MongoClient(MONGO_URI)
                    cls._instance.db = cls._instance.client[DB_NAME_LOGS]
                    cls._instance.logs = cls._instance.db[COLLECTION_LOGS]
                    # Create index on timestamp for fast retrieval
                    cls._instance.logs.create_index([("timestamp", -1)])
        return cls._instance

    def log_event(self, people_count, audio_status, risk_level):
        """Logs a system event to MongoDB."""
        event = {
            "timestamp": datetime.datetime.now(),
            "people_count": people_count,
            "audio_status": audio_status,
            "risk_level": risk_level
        }
        try:
            self.logs.insert_one(event)
        except Exception as e:
            print(f"[DB] Insert Error: {e}")

    def get_latest_event(self):
        """Retrieves the most recent event."""
        try:
            return self.logs.find_one(sort=[("timestamp", -1)])
        except Exception as e:
            print(f"[DB] Read Error: {e}")
            return None

    def get_recent_history(self, limit=50):
        """Retrieves history for charts."""
        try:
            cursor = self.logs.find(sort=[("timestamp", -1)]).limit(limit)
            return list(cursor)[::-1] # Return in chronological order
        except Exception as e:
            print(f"[DB] History Error: {e}")
            return []

    # --- Staff Management ---
    def add_staff(self, name, age, phone, zone):
        """Adds a new staff member."""
        staff_member = {
            "name": name,
            "age": age,
            "phone": phone,
            "zone": zone,
            "added_at": datetime.datetime.now()
        }
        try:
            self.db.staff.insert_one(staff_member)
            return True
        except Exception as e:
            print(f"[DB] Staff Insert Error: {e}")
            return False

    def get_all_staff(self):
        """Retrieves all staff members."""
        try:
            return list(self.db.staff.find())
        except Exception as e:
            print(f"[DB] Staff Read Error: {e}")
            return []

    def update_staff_zone(self, name, new_zone):
        """Updates the zone for a specific staff member."""
        try:
            result = self.db.staff.update_one(
                {"name": name},
                {"$set": {"zone": new_zone}}
            )
            # Return True if user was found (even if zone didn't change)
            return result.matched_count > 0
        except Exception as e:
            print(f"[DB] Staff Update Error: {e}")
            return False

    # --- Staff Management ---
    def delete_staff(self, staff_id):
        """Deletes a staff member by ID (Robust)."""
        try:
            from bson.objectid import ObjectId
            result = self.db.staff.delete_one({"_id": ObjectId(staff_id)})
            return result.deleted_count > 0
        except Exception as e:
            print(f"[DB] Staff Delete Error: {e}")
            return False

# Singleton Accessor
def get_db():
    return Database()
