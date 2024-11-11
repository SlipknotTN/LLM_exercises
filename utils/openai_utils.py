import openai
import tiktoken


def set_openai_vocareum_key():
    # Load the vocareum API key provided for the course
    vocareum_key_filepath = "./keys/openai_voc_key.txt"
    with open(vocareum_key_filepath, "r") as in_fp:
        vocareum_key_str = in_fp.readline()
        openai.api_base = "https://openai.vocareum.com/v1"
        openai.api_key = vocareum_key_str


def count_tokens(text: str, encoding: str = "cl100k_base"):
    """
    Count the number of tokens before calculating the embeddings

    Args:
        text: text for which you want to count the tokens
        encoding: encoding name

    Returns:
        the number of tokens to represent the text
    """
    tokenizer = tiktoken.get_encoding(encoding)
    return len(tokenizer.encode(text))
