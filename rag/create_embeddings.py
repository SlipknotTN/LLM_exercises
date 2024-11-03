import pandas as pd
import openai

# Check OpenAI documentation about available embedding models
EMBEDDING_MODEL_NAME = "text-embedding-ada-002"  # embeddings size 1536

# Load the vocareum API key provided for the course
vocareum_key_filepath = "./keys/openai_voc_key.txt"

with open(vocareum_key_filepath, "r") as in_fp:
    vocareum_key_str = in_fp.readline()
    openai.api_base = "https://openai.vocareum.com/v1"
    openai.api_key = vocareum_key_str

df = pd.read_csv("./rag/data/wiki_2022_data.csv", index_col=0)

# Get the embeddings
batch_size = 100
embeddings = []
for i in range(0, len(df), batch_size):
    # Send text data to OpenAI model to get embeddings, the embeddings are at sentence level, not word
    response = openai.Embedding.create(
        input=df.iloc[i:i+batch_size]["text"].tolist(),
        engine=EMBEDDING_MODEL_NAME
    )

    # Add embeddings to list
    embeddings.extend([data["embedding"] for data in response["data"]])

# Add embeddings list to dataframe
df["embeddings"] = embeddings

print(f"Embeddings space size using {EMBEDDING_MODEL_NAME}: {len(embeddings[0])}")

df.to_csv("./rag/data/wiki_2022_embeddings.csv")
