import requests

# Get the Wikipedia page for "2022" since OpenAI's models stop in 2021
params = {
    "action": "query",
    "prop": "extracts",
    "exlimit": 1,
    "titles": "2022",
    "explaintext": 1,
    "formatversion": 2,
    "format": "json"
}
resp = requests.get("https://en.wikipedia.org/w/api.php", params=params)
response_dict = resp.json()
response_sentences = response_dict["query"]["pages"][0]["extract"].split("\n")
print(f"Found {len(response_sentences)} sentences")
