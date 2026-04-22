import tiktoken

def main():
    print("Hello from llm-understanding!")

    enc = tiktoken.get_encoding("o200k_base")
    print(enc.encode("thisisawesome"))
    print(enc.decode(enc.encode("thisisawesome")))

    # To get the tokeniser corresponding to a specific model in the OpenAI API:
    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    print(enc.encode("thisisawesome"))
    print(enc.decode(enc.encode("thisisawesome")))


if __name__ == "__main__":
    main()
