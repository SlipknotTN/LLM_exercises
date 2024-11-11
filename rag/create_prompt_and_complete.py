import openai
import pandas as pd

from utils.openai_utils import count_tokens, set_openai_vocareum_key

# It has a training cutoff of September 2021 and token limit of 4,096 tokens.
completion_model = "gpt-3.5-turbo-instruct"

set_openai_vocareum_key()

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
print(f"Prompt tokens: {count_tokens(prompt)}")

initial_ukraine_answer = openai.Completion.create(
    model="gpt-3.5-turbo-instruct",
    prompt=question,
    max_tokens=150
)["choices"][0]["text"].strip()
print(f"Initial answer: {initial_ukraine_answer}")

answer_with_context = openai.Completion.create(
    model="gpt-3.5-turbo-instruct",
    prompt=prompt,
    max_tokens=150
)["choices"][0]["text"].strip()
print(f"RAG answer: {answer_with_context}")