import spacy
from sympy.physics.units import current

# Завантажуємо модель spaCy
nlp = spacy.load("en_core_web_md")

def normalize_ingredient(ingredient_text):
    doc = nlp(ingredient_text)

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
        "bite", "size", "bunch", "cups","all", "sized", "chunks", "chunk"
    }

    # Set of fractions to exclude
    fractions = {'½', '¼', '¾', '⅓', '⅔', '⅛', '⅜', '⅝', '⅞', '⅙', '⅚', '®'}

    # List to store relevant terms
    relevant_terms = []

    compound_phrases = []
    current_compound = []

    for token in doc:
        # Skip measurements, numbers, etc. (keep your existing exclusion logic)
        if (token.like_num or token.text in fractions or
                token.lemma_.lower() in measurement_units or
                token.is_punct ):
            continue

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

    return " ".join(relevant_terms)


# Приклад використання
ingredients = [
    'baking powder',
    'baking soda',
    'spring onions',


]

for ingredient in ingredients:
    normalized = normalize_ingredient(ingredient)
    print(f"Оригінал: {ingredient}")
    print(f"Нормалізовано: {normalized}\n")

def analyze_text_structure(text):
    for sentence in text:
        doc = nlp(sentence)
        print(f"Речення: {sentence}")
        print("-" * 50)
        for token in doc:
            print(f"Токен: {token.text}")
            print(f"Лема: {token.lemma_}")
            print(f"Частина мови: {token.pos_}")
            print(f"Залежність: {token.dep_}")
            print(f"Голова: {token.head.text} (POS: {token.head.pos_})")
            if token.dep_ == "compound":
                print(f"Це compound: {token.text} -> {token.head.text}")
            print("-" * 30)
        print("\n" + "=" * 50 + "\n")

# Викликаємо функцію для аналізу
analyze_text_structure(ingredients)