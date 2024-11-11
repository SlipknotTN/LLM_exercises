"""
Answer to a question on Wikipedia 2022 events using RAG

Requirements:
- Get Wikipedia 2022 data get_wikipedia_data.py
- Create embeddings with create_embeddings.py

Example with default arguments (run this from the repository root to correctly load the OpenAI key):
python rag/answer_question.py \
--question "Who is the owner of Twitter?"

Expected output:

Initial answer: The owner of Twitter is currently Jack Dorsey.
RAG answer: Elon Musk
"""

import argparse

import numpy as np
import openai
import pandas as pd
from openai.embeddings_utils import distances_from_embeddings, get_embedding

from utils.openai_utils import count_tokens, set_openai_vocareum_key


def get_rows_sorted_by_relevance(
    question: str, df: pd.DataFrame, embedding_model_name: str
) -> pd.DataFrame:
    """
    Function that takes in a question string and a dataframe containing
    rows of text and associated embeddings, and returns that dataframe
    sorted from least to most relevant for that question
    """
    # Get embeddings for the question text
    question_embeddings = get_embedding(question, engine=embedding_model_name)

    # Make a copy of the dataframe and add a "distances" column containing
    # the cosine distances between each row's embeddings and the
    # embeddings of the question
    df_copy = df.copy()
    df_copy["distances"] = distances_from_embeddings(
        question_embeddings, df_copy["embeddings"].values, distance_metric="cosine"
    )

    # Sort the copied dataframe by the distances and return it
    # (shorter distance = more relevant so we sort in ascending order)
    df_copy.sort_values("distances", ascending=True, inplace=True)
    return df_copy


def do_parsing():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Answer question on Wikipedia 2022 data",
    )
    parser.add_argument(
        "--input_embeddings",
        required=False,
        type=str,
        default="./rag/data/wiki_2022_embeddings.csv",
        help="Input dataset embeddings CSV",
    )
    parser.add_argument("--question", required=True, type=str, help="Question to ask")
    parser.add_argument(
        "--embedding_model_name",
        required=False,
        type=str,
        default="text-embedding-ada-002",  # embeddings size 1536
        help="Embeddings model used to create the dataset embeddings. "
        "It is necessary to re-use the same. "
        "Check OpenAI documentation about available embedding models.",
    )
    parser.add_argument(
        "--closest_sentences_output_filepath",
        required=False,
        type=str,
        help="Pass it to save the intermediate dataframe with the closest sentences to the question",
    )
    parser.add_argument(
        "--max_tokens_count",
        required=False,
        type=int,
        default=1000,
        help="Maximum number of tokens to use in the prompt",
    )
    args = parser.parse_args()
    return args


def main():
    args = do_parsing()
    print(args)

    # Init OpenAI
    set_openai_vocareum_key()

    # Retrieve the embeddings for the WikiPedia 2022 events
    df_embed = pd.read_csv(args.input_embeddings, index_col=0)

    # Convert embeddings to numpy array
    df_embed["embeddings"] = df_embed["embeddings"].apply(eval).apply(np.array)

    # Create the embeddings for the question using under the hood openai.Embedding.create
    df_sorted_distances = get_rows_sorted_by_relevance(
        question=args.question,
        df=df_embed,
        embedding_model_name=args.embedding_model_name,
    )

    if args.closest_sentences_output_filepath:
        df_sorted_distances.to_csv(args.closest_sentences_output_filepath)

    # Create the prompt with a template to get an answer to the question
    prompt_template = """
    Answer the question based on the context below, and if the question
    can't be answered based on the context, say "I don't know"

    Context: 

    {}

    ---

    Question: {}
    Answer:
    """

    # We want to exploit the available number of tokens for the model, but with a limit, because we are charged based
    # on the number of tokens
    current_token_count = count_tokens(prompt_template) + count_tokens(args.question)
    print(f"Prompt template + question number of tokens: {current_token_count}")

    # Add context until max tokens (which can be exceeded with the last step)
    context = []
    for text in df_sorted_distances["text"].values:

        # Increase the counter based on the number of tokens in this row
        text_token_count = count_tokens(text)
        current_token_count += text_token_count

        # Add the row of text to the list if we haven't exceeded the max.
        # The last step can exceed max_tokens_count
        if current_token_count <= args.max_tokens_count:
            context.append(text)
        else:
            break

    # Create the prompt with the context in a specific format to highlight each line (event)
    prompt = prompt_template.format("\n\n###\n\n".join(context), args.question)
    print(f"Prompt: {prompt}")
    print(f"Prompt tokens: {count_tokens(prompt)}")

    # From the documentation: the token count of your prompt plus max_tokens cannot exceed the model's context length.

    # Answer without using the context
    initial_answer = openai.Completion.create(
        model="gpt-3.5-turbo-instruct", prompt=args.question, max_tokens=150
    )["choices"][0]["text"].strip()
    print(f"Initial answer: {initial_answer}")

    # Answer using the context
    answer_with_context = openai.Completion.create(
        model="gpt-3.5-turbo-instruct", prompt=prompt, max_tokens=150
    )["choices"][0]["text"].strip()
    print(f"RAG answer: {answer_with_context}")


if __name__ == "__main__":
    main()
