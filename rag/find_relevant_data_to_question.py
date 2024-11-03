import numpy as np
import pandas as pd
from openai.embeddings_utils import distances_from_embeddings, get_embedding
from utils.openai_utils import set_openai_vocareum_key

# Check OpenAI documentation about available embedding models
EMBEDDING_MODEL_NAME = "text-embedding-ada-002"  # embeddings size 1536


def get_rows_sorted_by_relevance(question, df):
    """
    Function that takes in a question string and a dataframe containing
    rows of text and associated embeddings, and returns that dataframe
    sorted from least to most relevant for that question
    """
    # Get embeddings for the question text
    question_embeddings = get_embedding(question, engine=EMBEDDING_MODEL_NAME)

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


set_openai_vocareum_key()

df = pd.read_csv("./rag/data/wiki_2022_embeddings.csv", index_col=0)

# Convert embeddings to numpy array
df["embeddings"] = df["embeddings"].apply(eval).apply(np.array)

# Create the embeddings for the question using under the hood openai.Embedding.create
question = "When did Russia invade Ukraine?"
sorted_distances_df = get_rows_sorted_by_relevance(question=question, df=df)
sorted_distances_df.to_csv("./rag/data/russia_invasion_distances_sorted.csv")
