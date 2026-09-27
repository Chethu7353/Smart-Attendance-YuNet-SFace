import mysql.connector
from datetime import datetime

from security import get_database_config

DB_CONFIG = get_database_config()


def get_connection():
    return mysql.connector.connect(**DB_CONFIG)


# ==================================================
# LOGIN - CAMERA 1 / ENTRY
# ==================================================

def login_person(name):

    now = datetime.now()

    date = now.strftime("%Y-%m-%d")
    login_time = now.strftime("%H:%M:%S")

    conn = get_connection()
    cursor = conn.cursor()

    # Check whether person already has an active session
    cursor.execute("""
        SELECT id
        FROM attendance
        WHERE name = %s
        AND date = %s
        AND status = 'ACTIVE'
    """, (name, date))

    existing = cursor.fetchone()

    if existing:

        cursor.close()
        conn.close()

        return False, "ALREADY INSIDE"


    # Create login session
    cursor.execute("""
        INSERT INTO attendance
        (name, date, login_time, logout_time, status)
        VALUES (%s, %s, %s, NULL, 'ACTIVE')
    """, (
        name,
        date,
        login_time
    ))

    conn.commit()

    cursor.close()
    conn.close()

    return True, "LOGIN"


# ==================================================
# LOGOUT - CAMERA 2 / EXIT
# ==================================================

def logout_person(name):

    now = datetime.now()

    date = now.strftime("%Y-%m-%d")
    logout_time = now.strftime("%H:%M:%S")

    conn = get_connection()
    cursor = conn.cursor()

    # Find active session
    cursor.execute("""
        SELECT id
        FROM attendance
        WHERE name = %s
        AND date = %s
        AND status = 'ACTIVE'
        ORDER BY id DESC
        LIMIT 1
    """, (name, date))

    existing = cursor.fetchone()

    if not existing:

        cursor.close()
        conn.close()

        return False, "NO ACTIVE LOGIN"


    attendance_id = existing[0]


    # Update logout
    cursor.execute("""
        UPDATE attendance
        SET logout_time = %s,
            status = 'COMPLETED'
        WHERE id = %s
    """, (
        logout_time,
        attendance_id
    ))

    conn.commit()

    cursor.close()
    conn.close()

    return True, "LOGOUT"


# ==================================================
# DATABASE TEST
# ==================================================

if __name__ == "__main__":

    try:

        conn = get_connection()

        cursor = conn.cursor()

        cursor.execute("""
            SELECT DATABASE()
        """)

        database = cursor.fetchone()

        print("====================================")
        print("✅ MySQL Connected")
        print("Database:", database[0])
        print("====================================")

        cursor.close()
        conn.close()

    except Exception as e:

        print("❌ MySQL connection failed")
        print(e)