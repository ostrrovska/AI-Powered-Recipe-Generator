import pandas as pd
import spacy

nlp = spacy.load("en_core_web_sm")

df = pd.read_csv("recipes.csv")


def normalize_ingredient(ingredient_text):
    doc = nlp(ingredient_text)
    measurement_units = {
        "cup", "cups", "teaspoon", "teaspoons", "tablespoon", "tablespoons",
        "gram", "grams", "ounce", "ounces", "pound", "pounds", "can", "cans",
        "clove", "cloves", "pinch", "dash", "quart", "quarts", "liter", "liters",
        "milliliter", "milliliters", "ml", "gallon", "gallons", "stick", "sticks"
    }

    remove_with_flour = {"purpose", "wheat", "almond"}
    relevant_terms = []
    has_flour = False

    # Explicit list of critical nouns to always include
    critical_nouns = {"garlic", "cheese", "butter", "salt", "pepper"}

    # Check for flour presence
    for token in doc:
        if token.lemma_.lower() == "flour":
            has_flour = True

    for chunk in doc.noun_chunks:
        chunk_terms = []
        for token in chunk:
            lemma = token.lemma_.lower().strip()

            # Skip numbers, units, stops, punctuation
            if token.like_num or lemma in measurement_units or token.is_stop or token.is_punct:
                continue

            # Skip flour-related terms if flour is present
            if has_flour and lemma in remove_with_flour:
                continue

            # Always include critical nouns (even if they appear as VERB/ADJ due to parsing errors)
            if lemma in critical_nouns:
                chunk_terms.append(lemma)
                continue

            # Include adjectives and nouns
            if token.pos_ in ("ADJ", "NOUN", "PROPN"):
                chunk_terms.append(lemma)

        if chunk_terms:
            relevant_terms.extend(chunk_terms)

    # Fallback: If no terms found but garlic exists in text
    if not relevant_terms and "garlic" in ingredient_text.lower():
        return "garlic"

    return " ".join(relevant_terms)

# Тестові приклади
test_ingredients = [
    "2 cups all-purpose flour",
    "1 tablespoon chopped fresh parsley",
    "3 cloves garlic, minced",
    "1/2 teaspoon salt",
    "4 large tomatoes",
    "200 ml whole milk",
    "5 sticks unsalted butter, softened",
    "1 pinch of cinnamon powder",
    "1/4 cup grated Parmesan cheese",
    "2 tablespoons freshly squeezed lemon juice",
    "1 cup whole wheat flour",
    "1/2 cup almond flour",
    "3 ounces shredded mozzarella cheese"
]

for ingredient in test_ingredients:
    normalized = normalize_ingredient(ingredient)
    print(f"Original: {ingredient}")
    print(f"Normalized: {normalized}\n")
