import torch

from tokenizer import encode


block_size = 128

batch_size = 8


def create_data(
    text,
    token_to_id
):

    if not text or not text.strip():

        raise ValueError(
            "Training text is empty."
        )


    tokens = encode(
        text,
        token_to_id,
        add_special_tokens=True
    )


    if len(tokens) < block_size + 1:

        raise ValueError(
            f"Training data is too small. "
            f"Need at least {block_size + 1} tokens."
        )


    return torch.tensor(
        tokens,
        dtype=torch.long
    )


def get_batch(
    data,
    block_size=block_size,
    batch_size=batch_size,
    device="cpu"
):

    if len(data) < block_size + 1:

        raise ValueError(
            "Training data is too small."
        )


    max_start = (
        len(data)
        - block_size
        - 1
    )


    starts = torch.randint(
        0,
        max_start + 1,
        (
            batch_size,
        )
    )


    inputs = []

    targets = []


    for start in starts:

        start = start.item()


        input_tokens = data[
            start:
            start + block_size
        ]


        target_tokens = data[
            start + 1:
            start + block_size + 1
        ]


        inputs.append(
            input_tokens
        )


        targets.append(
            target_tokens
        )


    inputs = torch.stack(
        inputs
    )


    targets = torch.stack(
        targets
    )


    inputs = inputs.to(
        device
    )


    targets = targets.to(
        device
    )


    return (
        inputs,
        targets
    )