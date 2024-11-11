import pandas as pd

from utils.openai_utils import count_tokens

# Load the data sorted by distance with the question (you can't change the question here,
# you would need to recompute the embeddings and the distances
df_distances = pd.read_csv("./rag/data/russia_invasion_distances_sorted.csv", index_col=0)

question = "When did Russia invade Ukraine?"

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
current_token_count = count_tokens(prompt_template) + count_tokens(question)
print(f"Prompt template + question number of tokens: {current_token_count}")

max_tokens_count = 1000

# Add context until max tokens
context = []
for text in df_distances["text"].values:

    # Increase the counter based on the number of tokens in this row
    text_token_count = count_tokens(text)
    current_token_count += text_token_count

    # Add the row of text to the list if we haven't exceeded the max
    if current_token_count <= max_tokens_count:
        context.append(text)
    else:
        break

prompt = prompt_template.format("\n\n###\n\n".join(context), question)
print(prompt)