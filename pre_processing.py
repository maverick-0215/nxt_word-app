import requests
import re
import json

# Function to fetch and clean data from the URL
def fetch_and_clean_data(url):
    response = requests.get(url)
    response.raise_for_status()  # Check for request errors
    lines = response.text.split('\n')  # Split text into lines

    cleaned_lines = []
    for line in lines:
        cleaned_line = line.strip().lower()  # Clean the line
        if cleaned_line:  # Only keep non-empty lines
            cleaned_line = re.sub(r'[^a-zA-Z0-9 \.]', '', cleaned_line)  # Remove special characters
            cleaned_lines.append(cleaned_line)

    return cleaned_lines

# Load and clean text data from the URL
url = 'https://www.gutenberg.org/files/1661/1661-0.txt'
cleaned_lines = fetch_and_clean_data(url)

# Convert to a single text and split into words
text = ' '.join(cleaned_lines)
words = text.split()

# Build vocabulary mappings
unique_words = sorted(set(words))
stoi = {s: i + 1 for i, s in enumerate(unique_words)}  # String to index
stoi['.'] = 0  
itos = {i: s for s, i in stoi.items()}  

# Save mappings to JSON files
with open(r"C:\Users\Acer\Desktop\nxt_word app/stoi.json", "w") as f:
    json.dump(stoi,f)

with open(r"C:\Users\Acer\Desktop\nxt_word app/itos.json", "w") as f:
    json.dump(itos,f)


print("Vocabulary mappings saved to 'stoi.json' and 'itos.json'")
