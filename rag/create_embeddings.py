"""
Create embeddings from a CSV file

The CSV file must have a "text" column and each row represents a sentence or a unit of a long text.

Requirements:
- Extract the data in the input CSV format, e.g. using get_wikipedia_data.py

The default arguments loads the Wikipedia 2022 events CSV file.

Example (run this from the repository root to correctly load the OpenAI key):
    python rag/create_embeddings.py \
    --input_data_filepath ./rag/data/wiki_2022_data.csv \
    --output_embeddings_filepath ./rag/data/wiki_2022_embeddings.csv

The output CSV file will have "text" and "embeddings" columns.
"""

import argparse
import openai
import pandas as pd

from utils.openai_utils import set_openai_vocareum_key

def do_parsing():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Create text embeddings in CSV format from a CSV file with sentences",
    )
    parser.add_argument(
        "--input_data_filepath",
        required=False,
        type=str,
        default="./rag/data/wiki_2022_data.csv",
        help="Input CSV for which we want to extract the embeddings (one vector per row)"
    )
    parser.add_argument(
        "--request_size",
        type=int,
        default=100,
        help="Text embeddings extraction single request size"
    )
    parser.add_argument(
        "--embedding_model_name",
        required=False,
        type=str,
        default="text-embedding-ada-002",  # embeddings size 1536
        help="Embeddings model used to create the dataset embeddings. "
        "Check OpenAI documentation about available embedding models.",
    )
    parser.add_argument(
        "--output_embeddings_filepath",
        required=False,
        type=str,
        default="./rag/data/wiki_2022_embeddings.csv",
        help="Output dataset embeddings CSV",
    )
    return parser.parse_args()

def main():
    args = do_parsing()
    print(args)

    set_openai_vocareum_key()

    df = pd.read_csv(args.input_data_filepath, index_col=0)

    # Get the embeddings
    # In order to avoid a `RateLimitError` the data is sent in batches to the `Embedding.create` function
    embeddings = []
    for i in range(0, len(df), args.request_size):
        # Send text data to OpenAI model to get embeddings, the embeddings are at sentence level, not word
        response = openai.Embedding.create(
            input=df.iloc[i : i + args.request_size]["text"].tolist(), engine=args.embedding_model_name
        )

        # Add embeddings to list
        embeddings.extend([data["embedding"] for data in response["data"]])

    # Add embeddings list to dataframe
    df["embeddings"] = embeddings

    print(f"Embeddings space size using {args.embedding_model_name}: {len(embeddings[0])}")

    df.to_csv(args.output_embeddings_filepath)
    print(f"Embeddings saved to {args.output_embeddings_filepath}")

if __name__ == "__main__":
    main()
