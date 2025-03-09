import requests
import time
import pandas as pd
from bs4 import BeautifulSoup

# Mimic a browser request to avoid blocking
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

def extract_recipe_info(url):
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise error if request fails

        soup = BeautifulSoup(response.text, "html.parser")

        # Extract title
        title = soup.find("h1", {"class": "article-heading"}).text.strip()

        # Extract ingredients
        ingredients = []
        ingredient_section = soup.find("ul", {"class": "mm-recipes-structured-ingredients__list"})
        if ingredient_section:
            for li in ingredient_section.find_all("li"):
                ingredient = li.get_text().strip()
                ingredients.append(ingredient)

        return title, ingredients

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None, None
    except AttributeError as e:
        print(f"Element not found: {e}")
        return None, None

def get_recipe_links(section_url, num_pages=5):
    """
    Scrape recipe links from multiple pages of a search section.
    """
    recipe_links = []
    for page in range(num_pages):
        offset = page * 24  # AllRecipes shows 24 recipes per page
        search_url = f"{section_url}&offset={offset}"
        response = requests.get(search_url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract recipe links
        links = [a["href"] for a in soup.find_all("a", {"class": "comp mntl-card-list-card--extendable mntl-universal-card mntl-document-card mntl-card card card--no-image"})]
        recipe_links.extend(links)

        # Add delay to avoid being blocked
        time.sleep(2)

    return recipe_links

def main():
    # Define search sections
    search_sections = [
        {"name": "soup", "url": "https://www.allrecipes.com/search?q=soup"},
        {"name": "sandwich", "url": "https://www.allrecipes.com/search?sandwich=sandwich&offset=0&q=sandwich"},
        {"name": "pasta", "url": "https://www.allrecipes.com/search?q=pasta"},
        # Add more sections as needed
    ]

    all_recipes = []

    # Loop through each section
    for section in search_sections:
        print(f"Scraping {section['name']} recipes...")
        section_url = section["url"]
        recipe_links = get_recipe_links(section_url, num_pages=5)  # Scrape 5 pages (120 recipes)

        # Scrape details for each recipe
        for link in recipe_links:
            time.sleep(2)  # Add delay to avoid being blocked
            title, ingredients = extract_recipe_info(link)
            if title and ingredients:
                all_recipes.append({"section": section["name"], "title": title, "ingredients": ingredients})

    # Save to CSV
    df = pd.DataFrame(all_recipes)
    df.to_csv("recipes.csv", index=False)
    print(f"Scraped {len(all_recipes)} recipes.")

if __name__ == "__main__":
    main()