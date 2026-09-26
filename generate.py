import json
import os

import torch

from model import GPTModel


MODEL_PATH = "model.pth"

VOCAB_PATH = "vocab.json"

EMBEDDING_SIZE = 16

BLOCK_SIZE = 8

NUM_HEADS = 4

NUM_LAYERS = 4


def load_vocabulary():

    with open(
        VOCAB_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def load_model():

    word_to_id = load_vocabulary()

    vocab_size = len(
        word_to_id
    )

    model = GPTModel(
        vocab_size,
        EMBEDDING_SIZE,
        BLOCK_SIZE,
        NUM_HEADS,
        NUM_LAYERS
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location="cpu"
        )
    )

    model.eval()

    return model, word_to_id


def generate(
    model,
    tokens,
    max_new_tokens,
    temperature=1.0
):

    with torch.no_grad():

        for _ in range(
            max_new_tokens
        ):

            input_tokens = tokens[
                :, -BLOCK_SIZE:
            ]

            logits = model(
                input_tokens
            )

            logits = logits[
                :, -1, :
            ]

            logits = logits / max(
                temperature,
                0.1
            )

            probabilities = torch.softmax(
                logits,
                dim=-1
            )

            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            tokens = torch.cat(
                (
                    tokens,
                    next_token
                ),
                dim=1
            )

    return tokens


if __name__ == "__main__":

    model, word_to_id = load_model()

    id_to_word = {
        value: key
        for key, value in word_to_id.items()
    }

    start_tokens = torch.tensor([
        [
            word_to_id["The"]
        ]
    ])

    generated_tokens = generate(
        model,
        start_tokens,
        10,
        1.0
    )

    generated_words = []

    for token in generated_tokens[0]:

        generated_words.append(
            id_to_word[
                token.item()
            ]
        )

    print(
        " ".join(generated_words)
    )