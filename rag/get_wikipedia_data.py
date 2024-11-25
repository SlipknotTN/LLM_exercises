"""
Script to extract the data from a Wikipedia page using the API.

Example:
    python rag/get_wikipedia_data.py \
    --page_title "Castelnuovo di Garfagnana" \
    --wikipedia_lang it \
    --output_data_filepath ./rag/data/wiki_it_castelnuovo_garfagnana.csv

Output format: CSV file with "text" column, each row is an event.
"""

import argparse
import os

import pandas as pd
import requests
from dateutil.parser import parse as date_parser


def do_parsing():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Extract text from a Wikipedia page and save to CSV",
    )
    parser.add_argument(
        "--page_title",
        required=True,
        type=str,
        help="Wikipedia page title to parse"
    )
    parser.add_argument(
        "--wikipedia_lang",
        required=False,
        default="en",
        help="Wikipedia site language, language code to build the url, <wikipedia_lang>.wikipedia.org"
    )
    parser.add_argument(
        "--output_data_filepath",
        required=True,
        type=str,
        help="Output text dataset in CSV format, e.g. ./rag/data/wiki_page_data.csv",
    )
    return parser.parse_args()


def main():
    args = do_parsing()
    print(args)

    # "query" action documentation: https://en.wikipedia.org/w/api.php?action=help&modules=query
    params = {
        "action": "query",
        "prop": "extracts",
        "exlimit": 1,
        "titles": args.page_title,
        "explaintext": 1,
        "formatversion": 2,
        "format": "json",
    }
    resp = requests.get(f"https://{args.wikipedia_lang}.wikipedia.org/w/api.php", params=params)
    response_dict = resp.json()

    # Split sentences
    response_sentences = response_dict["query"]["pages"][0]["extract"].split("\n")
    print(f"{len(response_sentences)} sentences found")

    # TODO: Clean the text, remove paragraph names?!

    df = pd.DataFrame()
    df["text"] = response_sentences

    # TODO: How to clean paragraphs?! One paragraph: one embedding? History paragraph is long?
    # TODO: How to manage bullet points?

    # Clean up text to remove empty lines and headings
    # Example of headings: "== History =="
    #df = df[(df["text"].str.len() > 0) & (~df["text"].str.startswith("=="))]
    print(f"{len(df)} sentences after cleaning")

    os.makedirs(os.path.dirname(args.output_data_filepath), exist_ok=True)
    df.to_csv(args.output_data_filepath)


if __name__ == "__main__":
    main()
