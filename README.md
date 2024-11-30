# Udacity course "Large Language Models (LLMs) & Text Generation" exercises

https://www.udacity.com/enrollment/cd13318

## RAG exercises

Two exercises on RAG were presented during the course: the second one
is the final project, but most of the code is the same.

### RAG exercise 1: get answers about 2022 events

Goal: use Wikipedia 2022 data to customize a chatbot on data not present in
the original dataset (the OpenAI models data end in 2021).

### RAG exercise 2 (final project): get answers from a custom dataset

Goal: use Wikipedia API or any other dataset to get a pandas dataframe with at least
20 rows containing information not present in the original dataset.
Then build a chatbot able to answer on the new data.

### Steps to create the chatbot

The difference between the two exercises is the first page retrieval step,
then the scripts are exactly the same.

- Get the page data as dataframe CSV
  - Exercise 1: get the 2022 data from Wikipedia API with `get_wikipedia_2022_events_data.py`
  - Exercise 2: get a custom page data from Wikipedia API with `get_wikipedia_page.py`
- Create embeddings for the data with `create_embeddings.py`
- Answer to a question on the new data `answer_question.py`
  - Find relevant data to the question
  - Add the context to the prompt
  - Get the answer using OpenAI Completions API

### Useful links:
- Wikipedia API: https://en.wikipedia.org/w/api.php
- OpenAI embedding models: https://openai.com/index/new-and-improved-embedding-model/
- OpenAI completions API create: https://platform.openai.com/docs/api-reference/completions/create
