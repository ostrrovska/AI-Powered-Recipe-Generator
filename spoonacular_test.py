import os
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("SPOONACULAR_API_KEY")

response = requests.get(
    f"https://api.spoonacular.com/recipes/random?apiKey={API_KEY}&number=1"
)

if response.status_code == 200:
    print("✅ Connection successful! Here's a random recipe:")
    print(response.json()["recipes"][0]["title"])
else:
    print("❌ Failed to connect. Check your API key or network.")