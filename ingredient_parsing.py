import pandas as pd
import spacy
import ast
import psycopg2

nlp = spacy.load("en_core_web_md")
def get_db_connection():
    return psycopg2.connect(
        dbname="recipe_db",       # Your database name
        user="recipe_user",       # Your database user
        password="1234", # Your database password
        host="localhost"          # Your database host
    )


def deduplicate_ingredients():
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        # Update all recipes using PostgreSQL's array functions
        cur.execute("""
            UPDATE recipes
            SET normalized_ingredients = ARRAY(
                SELECT elem
                FROM (
                    SELECT DISTINCT ON (lower(elem)) elem, idx
                    FROM unnest(normalized_ingredients) WITH ORDINALITY AS arr(elem, idx)
                    WHERE elem IS NOT NULL AND elem <> ''
                    ORDER BY lower(elem), idx
                ) AS unique_ingredients
                ORDER BY idx
            )
            WHERE normalized_ingredients IS NOT NULL;
        """)
        conn.commit()
        print("Successfully removed duplicates from all recipes")

    except Exception as e:
        print(f"Error deduplicating ingredients: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()


def normalize_ingredient(ingredient_text):
    # Якщо ingredient_text — це рядок, який виглядає як список, розпарсимо його
    if isinstance(ingredient_text, str) and ingredient_text.startswith("[") and ingredient_text.endswith("]"):
        try:
            # Використовуємо ast.literal_eval для безпечного парсингу
            ingredients = ast.literal_eval(ingredient_text)
        except (ValueError, SyntaxError):
            # Якщо парсинг не вдається, обробляємо як звичайний рядок
            ingredients = [ingredient_text]
    else:
        # Якщо це не список, обробляємо як єдиний інгредієнт
        ingredients = [ingredient_text]

    # Список для зберігання нормалізованих інгредієнтів
    normalized_ingredients = []

    for ingredient in ingredients:
        if (ingredient.strip().lower() == "half and half" or
                ingredient.strip().lower() == "half-and-half"):
            normalized_ingredients.append("half-and-half")
            continue
        if ingredient.strip().lower() == "salt and pepper":
            normalized_ingredients.append("salt")
            normalized_ingredients.append("pepper")
            continue
        doc = nlp(ingredient)

        # Set of measurement units to exclude
        measurement_units = {
            "cup", "teaspoon", "tablespoon", "tablespoons", "gram", "ounce", "pound", "can",
            "clove", "pinch", "dash", "quart", "liter", "milliliter", "gallon",
            "stick", "rib", "head", "package", "inch", "piece", "fluid", "container",
            "jar", "loaf", "bottle", "pack", "pint", "cube", "stalk", "slice", "bulb",
            "strip", "packet", "envelope", "box", "bag", "carton", "sprig", "leaf",
            "half", "purpose", "pound", "ounce", "gram", "milliliter", "liter", "gallon",
            "quart", "pint", "dash", "pinch", "clove", "can", "package", "container",
            "jar", "loaf", "bottle", "pack", "cube", "stalk", "bulb", "strip", "packet",
            "envelope", "box", "bag", "carton", "sprig", "leaf", "fluid", "inch", "piece", "cup",
            "bite", "size", "bunch", "cups", "all", "sized", "chunks", "chunk", "casing", "dice"
        }

        # Set of fractions to exclude
        fractions = {'½', '¼', '¾', '⅓', '⅔', '⅛', '⅜', '⅝', '⅞', '⅙', '⅚', }

        additional_exclude = {
            "optional", "more", "as", "needed", "to", "taste", "divided", "enough", "cover",
            "cut", "into", "pieces", "such", "for", "with", "optional)", "needed)", "etc.", "or",
            '®', "cubed", "medium", "large", "small", "undrained", "fashioned", "instant", "diced",
            'unsalted', 'semisweet', 'table','frozen', 'fat','free','powdered', 'kosher', 'light',
            'whole','ground', 'sun', 'roasted', 'runny', 'short', 'sharp', 'wheat', 'steel'
        }

        food_terms = {
            # Herbs
            "oregano", "basil", "thyme", "rosemary", "sage",
            "dill", "mint", "parsley", "cilantro", "chives",
            # Spices
            "cumin", "paprika", "cinnamon", "nutmeg", "cloves",
            "cardamom", "turmeric", "ginger", "coriander"
        }

        # List to store relevant terms
        relevant_terms = []
        current_compound = []

        for token in doc:
            # Skip measurements, numbers, etc. (keep your existing exclusion logic)
            if (token.like_num or token.text in fractions or
                    token.lemma_.lower() in measurement_units or
                    token.is_punct or token.lemma_.lower() in additional_exclude):
                continue

            if token.text.lower() in food_terms:
                relevant_terms.append(token.text.lower())
                continue
            #if token.pos_ == "X" and token.text.lower() not in food_terms:
            #    print(f"Unknown X-tagged food term: {token.text}")

            # Handle compound phrases
            if token.dep_ == "compound":
                current_compound.append(token.text)
                continue

            # When we find the head noun of a compound phrase
            if current_compound:
                if token.pos_ in {"NOUN", "PROPN"}:
                    current_compound.append(token.lemma_.lower())
                    relevant_terms.append(" ".join(current_compound))
                    current_compound = []
                    continue
                else:
                    # If next token isn't a noun, add compound words separately
                    relevant_terms.extend(current_compound)
                    current_compound = []

            # Handle regular tokens
            if token.pos_ in {"NOUN", "PROPN"}:
                relevant_terms.append(token.lemma_.lower())
            elif token.pos_ == "VERB" and token.dep_ == "ROOT":
                relevant_terms.append(token.text.lower())  # Keep verbs like "baking" as-is
            elif token.pos_ == "ADJ":
                relevant_terms.append(token.lemma_.lower())

            # Add any remaining compound terms
        if current_compound:
            relevant_terms.extend(current_compound)

            # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in relevant_terms:
            if term not in seen:
                seen.add(term)
                unique_terms.append(term)

        # Join terms into a single ingredient
        normalized = " ".join(relevant_terms)
        normalized = ' '.join(normalized.split())  # Clean up spaces
        normalized_ingredients.append(normalized)

    # Return ingredients as a comma-separated string
    return ",".join(normalized_ingredients)

def normalize_and_store_ingredients():
    conn = get_db_connection()
    cur = conn.cursor()

    # Fetch all recipes with non-normalized ingredients
    cur.execute("SELECT id, original_ingredients FROM recipes")
    recipes = cur.fetchall()

    for recipe_id, ingredients in recipes:
        try:
            # Normalize the ingredients
            normalized = normalize_ingredient(ingredients)

            # Update the database
            cur.execute("""
                UPDATE recipes
                SET normalized_ingredients = %s
                WHERE id = %s
            """, (normalized.split(','), recipe_id))
            conn.commit()
        except Exception as e:
            print(f"Error normalizing recipe {recipe_id}: {e}")
            conn.rollback()

    cur.close()
    conn.close()
    print("Ingredient normalization complete.")

if __name__ == "__main__":
    normalize_and_store_ingredients()
    deduplicate_ingredients()

