import sqlite3
from config import Config

def init_database():
    config = Config()
    conn = sqlite3.connect(config.DATABASE_URL)
    
    # Create feedback table
    conn.execute('''CREATE TABLE IF NOT EXISTS feedback
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  original TEXT NOT NULL,
                  correction TEXT,
                  user_correction TEXT,
                  is_correct BOOLEAN,
                  domain TEXT,
                  model_type TEXT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    
    # Create user corrections table
    conn.execute('''CREATE TABLE IF NOT EXISTS user_corrections
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  original TEXT NOT NULL,
                  correction TEXT NOT NULL,
                  domain TEXT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    
    conn.commit()
    conn.close()
    print("Database initialized successfully")

if __name__ == '__main__':
    init_database()