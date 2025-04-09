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
    query = """
           SELECT title, normalized_ingredients, cooking_process 
           FROM recipes 
           WHERE normalized_ingredients IS NOT NULL 
       """
    df = pd.read_sql(query, conn)
    conn.close()

    # Конвертація в рядок і розділення на список
    df["normalized_ingredients"] = df["normalized_ingredients"].astype(str).str.split(',')

    # Фільтрація порожніх списків
    df = df[df["normalized_ingredients"].apply(len) > 0]

    return df