import openai
import pandas as pd

from utils.openai_utils import set_openai_vocareum_key

# Check OpenAI documentation about available embedding models
EMBEDDING_MODEL_NAME = "text-embedding-ada-002"  # embeddings size 1536

set_openai_vocareum_key()

df = pd.read_csv("./rag/data/wiki_2022_data.csv", index_col=0)

# Get the embeddings
batch_size = 100
embeddings = []
for i in range(0, len(df), batch_size):
    # Send text data to OpenAI model to get embeddings, the embeddings are at sentence level, not word
    response = openai.Embedding.create(
        input=df.iloc[i : i + batch_size]["text"].tolist(), engine=EMBEDDING_MODEL_NAME
    )

    # Add embeddings to list
    embeddings.extend([data["embedding"] for data in response["data"]])

# Add embeddings list to dataframe
df["embeddings"] = embeddings

print(f"Embeddings space size using {EMBEDDING_MODEL_NAME}: {len(embeddings[0])}")

df.to_csv("./rag/data/wiki_2022_embeddings.csv")
