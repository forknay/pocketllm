
def generate_bpe_list(text, size):
    """
    Create a byte pair encoding (BPE) vocab table from the given text.

    Args:
        text (str): The input text to create the BPE embedding table from.
        size (int): The size of the embedding table.
    Returns:
        list: A list of BPE tokens representing the embedding table.
    """
    words = text.split()
    vocab = {}
    for word in words:
        length = len(word)
        if length > 1:
            for i in range(length - 1):
                pair = word[i:i + 2]
                if vocab.get(pair):
                    vocab[pair] += 1
                else:
                    vocab[pair] = 1

    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
    chars = sorted(list(set(text)))
    size = size - len(chars)  # Reserve space for individual characters
    if size < 0:
        raise ValueError("Size must be greater than the number of unique characters in the text.")
    
    bpe_tokens = [token for token, _ in sorted_vocab[:size]]
    return chars + bpe_tokens

def build_bpe_vocab(bpe_list):
    encode = {}
    decode = {}
    for token in bpe_list:
        encode[token] = len(encode)
        decode[len(decode)] = token

    assert len(encode) == len(decode), "Encoding and decoding dictionaries must have the same length."
    return encode, decode, len(encode)

if __name__ == "__main__":
    with open("input.txt", "r", encoding="utf-8") as file:
        text = file.read()
    
    size = 1000 # Recommended by yours truly (github copilot)
    bpe_tokens = generate_bpe_list(text, size)
    encode, decode, vocab_size = build_bpe_vocab(bpe_tokens)

    print("BPE Vocabulary Size:", vocab_size)
    print("BPE Encoding Dictionary:", encode)
    print("BPE Decoding Dictionary:", decode)

