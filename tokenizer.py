import json
import os
import re


PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)


VOCAB_PATH = os.path.join(
    PROJECT_ROOT,
    "vocab.json"
)


SPECIAL_TOKENS = [
    "<PAD>",
    "<UNK>",
    "<BOS>",
    "<EOS>"
]


def tokenize(text):

    tokens = re.findall(
        r"\w+|[^\w\s]",
        text,
        re.UNICODE
    )

    return tokens


def create_vocabulary(text):

    tokens = tokenize(
        text
    )


    vocab = sorted(
        set(tokens)
    )


    vocab = (
        SPECIAL_TOKENS
        + vocab
    )


    token_to_id = {}


    for i, token in enumerate(
        vocab
    ):

        token_to_id[token] = i


    return token_to_id


def save_vocabulary(
    token_to_id
):

    with open(
        VOCAB_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            token_to_id,
            file,
            ensure_ascii=False,
            indent=4
        )


def load_vocabulary():

    if not os.path.exists(
        VOCAB_PATH
    ):

        return {}


    with open(
        VOCAB_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(
            file
        )


def encode(
    text,
    token_to_id,
    add_special_tokens=False
):

    tokens = tokenize(
        text
    )


    ids = []


    if add_special_tokens:

        ids.append(
            token_to_id["<BOS>"]
        )


    unk_id = token_to_id[
        "<UNK>"
    ]


    for token in tokens:

        token_id = token_to_id.get(
            token,
            unk_id
        )


        ids.append(
            token_id
        )


    if add_special_tokens:

        ids.append(
            token_to_id["<EOS>"]
        )


    return ids


def decode(
    ids,
    token_to_id
):

    id_to_token = {

        value: key

        for key, value
        in token_to_id.items()
    }


    tokens = []


    for token_id in ids:

        token = id_to_token.get(
            token_id,
            "<UNK>"
        )


        if token in SPECIAL_TOKENS:

            continue


        tokens.append(
            token
        )


    text = ""


    punctuation_without_space = {
        ".",
        ",",
        "!",
        "?",
        ":",
        ";",
        "%",
        ")",
        "]",
        "}"
    }


    punctuation_with_space_before = {
        "(",
        "[",
        "{"
    }


    for token in tokens:

        if not text:

            text = token


        elif token in punctuation_without_space:

            text += token


        elif token in punctuation_with_space_before:

            text += " " + token


        else:

            text += " " + token


    return text


if __name__ == "__main__":

    text = (
        "Python is a popular programming language."
    )


    vocabulary = create_vocabulary(
        text
    )


    save_vocabulary(
        vocabulary
    )


    print(
        "Vocabulary size:",
        len(vocabulary)
    )


    encoded = encode(
        text,
        vocabulary,
        add_special_tokens=True
    )


    print(
        "Encoded:",
        encoded
    )


    decoded = decode(
        encoded,
        vocabulary
    )


    print(
        "Decoded:",
        decoded
    )