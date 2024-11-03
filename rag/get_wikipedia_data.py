import os
import requests

import pandas as pd
from dateutil.parser import parse as date_parser


# Get the Wikipedia page for "2022" since OpenAI's models stop in 2021
params = {
    "action": "query",
    "prop": "extracts",
    "exlimit": 1,
    "titles": "2022",
    "explaintext": 1,
    "formatversion": 2,
    "format": "json",
}
resp = requests.get("https://en.wikipedia.org/w/api.php", params=params)
response_dict = resp.json()
response_sentences = response_dict["query"]["pages"][0]["extract"].split("\n")
print(f"{len(response_sentences)} sentences found")

df = pd.DataFrame()
df["text"] = response_sentences

# Clean up text to remove empty lines and headings
# Example of headings: "== Nobel prizes =="
df = df[(df["text"].str.len() > 0) & (~df["text"].str.startswith("=="))]

# In some cases dates are used as headings instead of being part of the
# text sample; adjust so dated text samples start with dates
# Example:
# row x: "December 7"
# row x + 1: "event A bla bla bla"
# row x + 2: "event B bla bla bla"
prefix = ""
for i, row in df.iterrows():
    # If the row already has " - ", it already has the needed date prefix
    if " – " not in row["text"]:
        try:
            # If the row's text is a date, set it as the new prefix for the next rows without date
            date_parser(row["text"])
            prefix = row["text"]
        except:
            # If the row's text isn't a date, add the prefix
            row["text"] = prefix + " – " + row["text"]
df = df[df["text"].str.contains(" – ")].reset_index(drop=True)
print(f"{len(df)} sentences after cleaning")

os.makedirs("./rag/data", exist_ok=True)
df.to_csv("./rag/data/wiki_2022_data.csv")
