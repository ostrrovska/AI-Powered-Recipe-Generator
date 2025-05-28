import tkinter as tk
from tkinter import messagebox
from recipe_recommender import get_recommendations


# Example function to integrate with your existing logic
def get_recommendations_ui():
    user_input = entry_ingredients.get()
    require_all = var_require_all.get()

    if not user_input.strip():
        messagebox.showerror("Error", "Please enter some ingredients.")
        return

    # Call your actual recommendation function
    try:
        recommendations = get_recommendations(user_input, require_all=require_all)
        results_text.delete(1.0, tk.END)  # Clear textbox

        if recommendations.empty:
            results_text.insert(tk.END, "No recipes found.\n")
        else:
            for _, row in recommendations.iterrows():
                results_text.insert(tk.END, f"Recipe: {row['title']}\n")
                results_text.insert(tk.END, f"Ingredients: {', '.join(row['normalized_ingredients'])}\n\n")
                results_text.insert(tk.END, f"Cooking process: {row['cooking_process']}\n\n")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")


# Main Tkinter window
root = tk.Tk()
root.title("Recipe Recommender")

# Input field for ingredients
tk.Label(root, text="Enter ingredients (comma-separated):").pack(pady=5)
entry_ingredients = tk.Entry(root, width=50)
entry_ingredients.pack(pady=5)

# Checkbox for "require all ingredients"
var_require_all = tk.BooleanVar()
tk.Checkbutton(root, text="Require all ingredients", variable=var_require_all).pack(pady=5)

# Button to fetch recommendations
tk.Button(root, text="Get Recommendations", command=get_recommendations_ui).pack(pady=10)

# Text area to display results
results_text = tk.Text(root, width=100, height=15)
results_text.pack(pady=5)

# Run the Tkinter event loop
root.mainloop()