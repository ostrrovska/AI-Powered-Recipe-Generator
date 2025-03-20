import requests
import time
import pandas as pd
from bs4 import BeautifulSoup
import psycopg2

def get_db_connection():
    return psycopg2.connect(
        dbname="recipe_db",
        user="recipe_user",
        password="1234",
        host="localhost"
    )
 # Connect to the database
conn = get_db_connection()
cur = conn.cursor()

# Mimic a browser request to avoid blocking
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

def extract_recipe_info(url):
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Extract title
        title = soup.find("h1", {"class": "article-heading"}).text.strip()

        # Extract ingredients
        ingredients = []
        ingredient_section = soup.find("ul", {"class": "mm-recipes-structured-ingredients__list"})
        if ingredient_section:
            for li in ingredient_section.find_all("li"):
                ingredients.append(li.get_text().strip())

        # Extract cooking process
        cooking_steps = []
        instructions_section = soup.find("ol", {"class": "comp mntl-sc-block mntl-sc-block-startgroup mntl-sc-block-group--OL"})
        if instructions_section:
            for li in instructions_section.find_all("li",
                                                    class_="comp mntl-sc-block mntl-sc-block-startgroup mntl-sc-block-group--LI"):
                # Find the <p> tag with the instruction text
                instruction_p = li.find("p", class_="comp mntl-sc-block mntl-sc-block-html")
                if instruction_p:
                    # Extract and clean the text
                    step_text = instruction_p.get_text().strip()
                    step_text = " ".join(step_text.split())  # Remove extra spaces
                    if step_text:  # Only add non-empty steps
                        cooking_steps.append(step_text)

        cooking_process = "\n".join(cooking_steps) if cooking_steps else None

        return title, ingredients, cooking_process

    except Exception as e:
        print(f"Error extracting {url}: {e}")
        return None, None, None

def get_recipe_links(section_url, num_pages=1):
    """
    Scrape recipe links from multiple pages of a search section.
    """
    recipe_links = []

    # List of possible class names
    class_names = [
        "comp mntl-card-list-card--extendable mntl-universal-card mntl-document-card mntl-card card card--no-image",
        "comp mntl-card-list-items mntl-universal-card mntl-document-card mntl-card card card--no-image",
        #"comp mntl-card-list-items mntl-universal-card mntl-document-card mntl-card card card--no-image",  # Replace with the actual second class
        #"comp mntl-card-list-card--extendable mntl-universal-card mntl-document-card mntl-card card card--no-image"  # Add more classes if necessary
    ]

    for page in range(num_pages):
        offset = page * 24  # Assuming 24 recipes per page
        search_url = f"{section_url}&offset={offset}"
        response = requests.get(search_url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract links using multiple class names
        for class_name in class_names:
            links = [a["href"] for a in soup.find_all("a", class_=class_name)]
            recipe_links.extend(links)

        # Add delay to avoid being blocked
        time.sleep(2)

    return recipe_links

def main():
    # Define search sections
    search_sections = [
        {"name": "soup", "url": "https://www.allrecipes.com/search?q=soup"},
        #{"name": "dessert", "url": "https://www.allrecipes.com/search?q=dessert"},
        #{"name": "breakfast", "url": "https://www.allrecipes.com/search?q=breakfast"},
        #{"name": "pasta", "url": "https://www.allrecipes.com/search?q=pasta"},
        #{"name": "rice", "url": "https://www.allrecipes.com/search?q=Rice"},
        #{"name": "salad", "url": "https://www.allrecipes.com/search?q=salad"},
        #{"name": "meat", "url": "https://www.allrecipes.com/search?q=meat"},
        #{"name": "vegan", "url": "https://www.allrecipes.com/search?q=vegan"},
        #{"name": "fish", "url": "https://www.allrecipes.com/search?q=fish"},
        # Add more sections as needed
    ]

    # Loop through each section
    for section in search_sections:
        print(f"Scraping {section['name']} recipes...")
        section_url = section["url"]
        recipe_links = get_recipe_links(section_url, num_pages=2)  # Scrape 2 pages

        # Scrape details for each recipe
        for link in recipe_links:
            time.sleep(2)  # Add delay to avoid being blocked
            title, ingredients, cooking_process = extract_recipe_info(link)  # Updated to include cooking_process

            if title and ingredients:
                try:
                    # Insert into the database
                    cur.execute("""
                            INSERT INTO recipes (url, title, section, original_ingredients, cooking_process)
                            VALUES (%s, %s, %s, %s, %s)
                            ON CONFLICT (url) DO NOTHING;
                        """, (
                        link,  # URL
                        title,  # Title
                        section['name'],  # Section
                        str(ingredients),  # Ingredients (converted to string)
                        cooking_process  # Cooking process
                    ))
                    conn.commit()  # Save changes
                except Exception as e:
                    print(f"Error inserting {link}: {e}")
                    conn.rollback()  # Undo changes if there's an error

    # Close the database connection
    cur.close()
    conn.close()
    print("Scraping and database insertion complete.")

if __name__ == "__main__":
    main()