import os
from dotenv import load_dotenv

load_dotenv()


def get_database_config():

    return {
        "host": os.getenv("DB_HOST"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "database": os.getenv("DB_NAME")
    }


def get_camera_config():

    return {
        "entry": os.getenv("ENTRY_CAMERA"),
        "exit": os.getenv("EXIT_CAMERA")
    }