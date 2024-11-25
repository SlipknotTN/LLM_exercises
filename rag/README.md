# RAG Exercise

Goal: use Wikipedia 2022 data to customize a chatbot on data not present in
the original dataset (the OpenAI models data end in 2021).

## Steps to create the chatbot
- Get the 2022 data from the Wikipedia API with `rag/get_wikipedia_2022_events_data.py`
- Create embeddings for the data with `rag/create_embeddings.py`
- Answer to a question on the new data `rag/answer_question.py`
  - Find relevant data to the question
  - Add the context to the prompt
  - Get the answer using OpenAI Completions API

## Useful links:
- Wikipedia API: https://en.wikipedia.org/w/api.php
- OpenAI embedding models: https://openai.com/index/new-and-improved-embedding-model/
- OpenAI completions API create: https://platform.openai.com/docs/api-reference/completions/create