# db_utils.py
import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()

def get_db_connection():
    return psycopg2.connect(
        dbname="recipe_db",
        user="recipe_user",
        password="",
        host="localhost"
    )

def fetch_all_recipes():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT title, normalized_ingredients 
        FROM recipes 
        WHERE normalized_ingredients IS NOT NULL
    """)
    recipes = cur.fetchall()
    conn.close()
    return pd.DataFrame(recipes, columns=["title", "normalized_ingredients"])