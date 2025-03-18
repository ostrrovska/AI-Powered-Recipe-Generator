import pandas as pd
import spacy
import ast

nlp = spacy.load("en_core_web_md")

df = pd.read_csv("recipes.csv")


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
            '®', "cubed", "medium", "large", "small", "undrained", "fashioned", "instant"
        }

        # List to store relevant terms
        relevant_terms = []

        for token in doc:
            # Skip numbers, fractions, and measurement units
            if (token.like_num or
                    token.text in fractions or
                    token.lemma_.lower() in measurement_units or
                    token.is_punct or
                    token.lemma_.lower() in additional_exclude):
                continue

            # Handle compound nouns
            if token.dep_ == "compound":
                relevant_terms.append(f"{token.text}")
            # Focus on nouns, proper nouns, and adjectives that modify nouns
            elif token.pos_ in {"NOUN", "PROPN", "ADJ"}:
                if token.pos_ == "ADJ" and token.head.pos_ in {"NOUN", "PROPN", "VERB"}:
                    relevant_terms.append(token.lemma_.lower())
                elif token.pos_ in {"NOUN", "PROPN"}:
                    relevant_terms.append(token.lemma_.lower())

        # Join terms into a single ingredient
        normalized = " ".join(relevant_terms)
        normalized = ' '.join(normalized.split())  # Clean up spaces
        normalized_ingredients.append(normalized)

    # Return ingredients as a comma-separated string
    return ",".join(normalized_ingredients)


df["normalized_ingredients"] = df["ingredients"].apply(normalize_ingredient)

#for original, normalized in zip(df["ingredients"], df["normalized_ingredients"]):
#    print(f"Оригінал: {original}")
 #   print(f"Нормалізовано: {normalized}\n")

df.to_csv("normalized_recipes.csv", index=False)
