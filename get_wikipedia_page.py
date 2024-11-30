"""
Script to extract the data from a Wikipedia page using the API.

Example:
    python rag/get_wikipedia_data.py \
    --page_title "Castelnuovo di Garfagnana" \
    --wikipedia_lang it \
    --skip_sections "Collegamenti_esterni" "Altri_progetti" \
    --output_data_filepath ./rag/data/wiki_it_castelnuovo_garfagnana.csv

Output format: CSV file with "text" column, each row is an event.
"""

import argparse
import os
from collections import defaultdict
from typing import Optional

import bs4
import pandas as pd
import requests
from bs4 import BeautifulSoup


def get_dict_key_from_headings(
    last_h2_level_paragraph: str,
    last_h3_level_paragraph: Optional[str] = None,
    last_h4_level_paragraph: Optional[str] = None,
) -> str:
    key = f"{last_h2_level_paragraph}"
    if last_h3_level_paragraph is not None:
        key += f" - {last_h3_level_paragraph}"
    if last_h4_level_paragraph is not None:
        key += f" - {last_h4_level_paragraph}"
    return key


def get_cleaned_text(element) -> str:
    """
    Strip text and remove '\n' inside the paragraph
    """
    return element.get_text().strip().replace(u"\xa0"," ").replace("\n", " ")


def do_parsing():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Extract text from a Wikipedia page and save to CSV",
    )
    parser.add_argument(
        "--paragraphs_to_join",
        nargs="+",
        default=[],
        help="Name of the paragraphs for which we want to join the content into a single sentence",
    )
    parser.add_argument(
        "--page_title", required=True, type=str, help="Wikipedia page title to parse"
    )
    parser.add_argument(
        "--wikipedia_lang",
        required=False,
        default="en",
        help="Wikipedia site language, language code to build the url, <wikipedia_lang>.wikipedia.org",
    )
    parser.add_argument(
        "--output_data_filepath",
        required=True,
        type=str,
        help="Output text dataset in CSV format, e.g. ./rag/data/wiki_page_data.csv",
    )
    parser.add_argument(
        "--skip_sections",
        required=False,
        default=[],
        nargs="+",
        type=str,
        help="Section names to skip",
    )
    return parser.parse_args()


def main():
    args = do_parsing()
    print(args)

    # "query" action documentation: https://en.wikipedia.org/w/api.php?action=help&modules=query
    # Don't pass "explaintext": 1 to get the text in HTML format. It is a bit more complex to parse, but we have
    # all the information to understand when a list is present
    params = {
        "action": "query",
        "prop": "extracts",
        "exlimit": 1,
        "titles": args.page_title,
        "exsectionformat": "wiki",
        "format": "json",
    }

    resp = requests.get(
        f"https://{args.wikipedia_lang}.wikipedia.org/w/api.php", params=params
    )
    response_dict = resp.json()

    page_dict = next(iter(response_dict["query"]["pages"].values()))
    title = page_dict["title"]
    html_text = page_dict["extract"]
    soup = BeautifulSoup(html_text, "html.parser")
    print(soup.prettify())

    # Use BeatifulSoap the go element by element
    # headings -> new sectopm level
    # <p>...</p> sentences
    # <ul><li>..</li><li>...</li>...</ul> <li> elements to merge
    sentences_dict = defaultdict(list)
    last_h2_level_paragraph = None
    last_h3_level_paragraph = None
    last_h4_level_paragraph = None
    for element in soup:
        if type(element) == bs4.Tag:
            if element.name == "p" and last_h2_level_paragraph is None:
                # Intro before the first headings
                sentences_dict[title].append(get_cleaned_text(element))
            elif element.name == "h2":
                # First level paragraph
                last_h2_level_paragraph = element.attrs["data-mw-anchor"]
                last_h3_level_paragraph = None
                last_h4_level_paragraph = None
                building_list = False
            elif element.name == "h3":
                # Second level paragraph
                last_h3_level_paragraph = element.attrs["data-mw-anchor"]
                last_h4_level_paragraph = None
                building_list = False
            elif element.name == "h4":
                # Third level paragraph
                last_h4_level_paragraph = element.attrs["data-mw-anchor"]
                building_list = False
            elif element.name == "p":
                # Sentence of a paragraph
                # Concatenate the headings to provide context
                key = get_dict_key_from_headings(
                    last_h2_level_paragraph,
                    last_h3_level_paragraph,
                    last_h4_level_paragraph,
                )
                # Search for <ul> inside <p>
                for p_children in element.children:
                    if type(p_children) == bs4.Tag and p_children.name == "ul":
                        raise ValueError("List <ul> inside a <p> not supported")
                sentences_dict[key].append(get_cleaned_text(element))

            elif element.name == "ul" or element.name == "dl":
                # Get the list elements and merge them when necessary
                # DO NOT MERGE when there is no sentence before ending with ":"
                # MERGE when the previous sentence ends with ":" or when another list element is preceding

                # Logic to merge the list elements
                list_content_str = ""
                for list_element in element.children:
                    if (
                        list_element.name == "li"
                        or list_element.name == "dd"
                        or list_element.name == "dt"
                    ):
                        list_text = get_cleaned_text(list_element)
                        list_content_str += list_text + "\n"
                list_content_str = (
                    list_content_str.replace("\n", "; ")
                    .replace(",;", ";")
                    .replace(";;", ";")
                    .replace(".;", ";")[: -len(", ")]
                )
                key = get_dict_key_from_headings(
                    last_h2_level_paragraph,
                    last_h3_level_paragraph,
                    last_h4_level_paragraph,
                )
                last_sentence_for_key = (
                    sentences_dict[key][-1] if len(sentences_dict[key]) > 0 else ""
                )
                if last_sentence_for_key.endswith(":"):
                    # Concatenate the list elements with the previous sentence which explains the list content,
                    sentences_dict[key][-1] += " " + list_content_str
                elif last_element_type == "ul" or last_element_type == "dl":
                    # The list could already been started with a different ul or dl element,
                    # in this case we don't support nesting and we simply concatenate
                    print(
                        f"WARNING: probably there is a nested list; it will be squashed into a single level, list element content: '{list_content_str}'"
                    )
                    sentences_dict[key][-1] += "; " + list_content_str
                else:
                    # The list is probably part of an entire section and not introduce with ":",
                    # so it does worth keeping split
                    sentences_dict[key].extend(list_content_str.split("; "))
            else:
                raise ValueError(f"Tag {element.name} not supported")
            last_element_type = element.name

    df_content = {"text": []}
    skip_keys_start = tuple(
        [skip_section + " - " for skip_section in args.skip_sections]
    )
    for key, key_sentences in sentences_dict.items():
        if key not in args.skip_sections and key.startswith(skip_keys_start) is False:
            for key_sentence in key_sentences:
                if key_sentence != "":
                    df_content["text"].append(f"{key} - {key_sentence}")

    df = pd.DataFrame.from_dict(df_content)
    print(f"{len(df)} sentences obtained from the page '{args.page_title}'")

    os.makedirs(os.path.dirname(args.output_data_filepath), exist_ok=True)
    df.to_csv(args.output_data_filepath)
    print(f"CSV file saved to '{args.output_data_filepath}'")


if __name__ == "__main__":
    main()
