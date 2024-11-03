import openai


def set_openai_vocareum_key():
    # Load the vocareum API key provided for the course
    vocareum_key_filepath = "./keys/openai_voc_key.txt"
    with open(vocareum_key_filepath, "r") as in_fp:
        vocareum_key_str = in_fp.readline()
        openai.api_base = "https://openai.vocareum.com/v1"
        openai.api_key = vocareum_key_str
