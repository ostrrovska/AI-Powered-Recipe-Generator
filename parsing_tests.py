import spacy

nlp = spacy.load("en_core_web_sm")

text = [
    "5 tablespoons butter, divided",
    "1 onion, chopped",
    "1 stalk celery, chopped",
    "3 cups chicken broth",
    "8 cups broccoli florets",
    "3 tablespoons all-purpose flour",
    "2 cups milk",
    "ground black pepper to taste"
]

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
analyze_text_structure(text)