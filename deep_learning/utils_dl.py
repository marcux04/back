from pymongo import MongoClient
from dotenv import load_dotenv
import os
import pandas as pd
import json

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "bank_marketing_db")

def save_prediction_to_mongo(record: dict, collection_name="dl_predictions"):
    if not MONGO_URI:
        return
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    db[collection_name].insert_one(record)

def save_training_log(record: dict, collection_name="dl_training_logs"):
    if not MONGO_URI:
        return
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    db[collection_name].insert_one(record)