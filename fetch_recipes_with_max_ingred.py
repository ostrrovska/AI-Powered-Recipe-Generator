import os
import requests
from dotenv import load_dotenv
from time import sleep
from db_utils import get_db_connection

# Load config
load_dotenv()
API_KEY = os.getenv("SPOONACULAR_API_KEY")
BASE_URL = "https://api.spoonacular.com/recipes"

def fetch_simple_recipe_ids(count=5, max_ingredients=5):
    """Get random recipe IDs with limited number of ingredients"""
    response = requests.get(
        f"{BASE_URL}/random",
        params={
            "apiKey": API_KEY,
            "number": count,
            "numberOfIngredients": max_ingredients  # Максимальна кількість інгредієнтів
        }
    )
    return [recipe["id"] for recipe in response.json()["recipes"]]


def get_recipe_details(recipe_id):
    """Improved instruction extraction"""
    recipe = requests.get(
        f"{BASE_URL}/{recipe_id}/information",
        params={"apiKey": API_KEY}
    ).json()

    # Extract cooking process from multiple possible fields
    instructions = ""
    if recipe.get("analyzedInstructions"):
        instructions = "\n".join(
            step["step"] for step in recipe["analyzedInstructions"][0]["steps"]
        )
    elif recipe.get("instructions"):
        instructions = recipe["instructions"]
    elif recipe.get("summary"):
        instructions = recipe["summary"]  # Fallback to summary if no instructions

    recipe["full_instructions"] = instructions
    return recipe


def store_raw_recipe(conn, recipe):
    """Store data with cleanName if available, otherwise name"""
    ingredients = []
    for ing in recipe.get("extendedIngredients", []):
        if ing is None:
            continue
        # Safely get cleanName if exists, otherwise name
        clean_name = ing.get("cleanName", ing.get("name"))
        if clean_name:  # Only add if not empty/None
            ingredients.append(clean_name)

    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO recipes 
            (url, title, section, original_ingredients, cooking_process)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (url) DO NOTHING
        """, (
            recipe.get("sourceUrl", ""),
            recipe["title"],
            "spoonacular_simple",
            str(ingredients) if ingredients else "[]",
            recipe["full_instructions"] or "No instructions provided"
        ))
    conn.commit()
    print(f"Stored simple recipe: {recipe['title']} ({len(ingredients)} ingredients)")


def main():
    conn = get_db_connection()

    # Завантажуємо рецепти з максимум 5 інгредієнтами
    recipe_ids = fetch_simple_recipe_ids(10, max_ingredients=5)

    for recipe_id in recipe_ids:
        try:
            recipe = get_recipe_details(recipe_id)
            store_raw_recipe(conn, recipe)
            sleep(1.5)  # Stay within rate limits
        except Exception as e:
            print(f"Error processing recipe {recipe_id}: {e}")
            continue

    conn.engine.dispose()


if __name__ == "__main__":
    main()