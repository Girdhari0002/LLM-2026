from fastapi import FastAPI

from pydantic import BaseModel


import torch

import json

import os

import sys


PROJECT_ROOT = os.path.dirname(

    os.path.dirname(

        os.path.abspath(__file__)

    )

)


sys.path.append(
    PROJECT_ROOT
)


from model import GPTModel


from train import train_model


from tokenizer import (
    encode,
    decode
)


app = FastAPI()


EMBEDDING_SIZE = 256

BLOCK_SIZE = 128

NUM_HEADS = 8

NUM_LAYERS = 8


DEVICE = (

    "cuda"

    if torch.cuda.is_available()

    else "cpu"

)


MODEL_PATH = os.path.join(

    PROJECT_ROOT,

    "model.pth"

)


VOCAB_PATH = os.path.join(

    PROJECT_ROOT,

    "vocab.json"

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


def load_model():

    token_to_id = (
        load_vocabulary()
    )


    if not token_to_id:

        raise Exception(

            "Vocabulary not found. "
            "Train the model first."

        )


    if not os.path.exists(
        MODEL_PATH
    ):

        raise Exception(

            "Model not found. "
            "Train the model first."

        )


    vocab_size = len(
        token_to_id
    )


    model = GPTModel(

        vocab_size,

        EMBEDDING_SIZE,

        BLOCK_SIZE,

        NUM_HEADS,

        NUM_LAYERS

    )


    state_dict = torch.load(

        MODEL_PATH,

        map_location=DEVICE

    )


    model.load_state_dict(
        state_dict
    )


    model.to(
        DEVICE
    )


    model.eval()


    return (

        model,

        token_to_id

    )


class GenerateRequest(BaseModel):

    prompt: str

    temperature: float = 0.7

    maxTokens: int = 50

    topK: int = 40

    topP: float = 0.9

    repetitionPenalty: float = 1.15


class TrainRequest(BaseModel):

    text: str

    steps: int = 50000

    learningRate: float = 0.0003


@app.get("/health")
def health():

    return {

        "success": True,

        "message":
            "Python LLM API is running",

        "device":
            DEVICE,

        "gpu":
            (

                torch.cuda.get_device_name(0)

                if torch.cuda.is_available()

                else "None"

            )

    }


@app.post("/train")
def train(
    request: TrainRequest
):

    try:

        print()

        print(
            "=" * 50
        )

        print(
            "Training request received from UI"
        )

        print(
            "Training text length:",
            len(request.text)
        )

        print(
            "Steps:",
            request.steps
        )

        print(
            "Learning rate:",
            request.learningRate
        )

        print(
            "Device:",
            DEVICE
        )


        if torch.cuda.is_available():

            print(

                "GPU:",

                torch.cuda.get_device_name(0)

            )


        print(
            "=" * 50
        )


        result = train_model(

            text=request.text,

            steps=request.steps,

            learning_rate=request.learningRate

        )


        return {

            "success": True,

            "message":
                "Model trained successfully.",

            "steps":
                result.get(
                    "steps",
                    request.steps
                ),

            "loss":
                result.get(
                    "loss",
                    None
                ),

            "vocabSize":
                result.get(
                    "vocab_size",
                    None
                )

        }


    except Exception as error:

        print(
            "Training error:",
            error
        )


        return {

            "success": False,

            "message":
                str(error)

        }


def apply_repetition_penalty(
    logits,
    generated_tokens,
    penalty
):

    if penalty <= 1.0:

        return logits


    recent_tokens = generated_tokens[
        0,
        -64:
    ]


    unique_tokens = torch.unique(
        recent_tokens
    )


    for token_id in unique_tokens:

        token_id = token_id.item()


        if logits[0, token_id] < 0:

            logits[0, token_id] *= penalty

        else:

            logits[0, token_id] /= penalty


    return logits


def apply_top_k(
    logits,
    top_k
):

    if top_k <= 0:

        return logits


    top_k = min(

        top_k,

        logits.size(-1)

    )


    values, _ = torch.topk(

        logits,

        top_k

    )


    minimum_value = values[
        :,
        -1
    ].unsqueeze(
        -1
    )


    logits = torch.where(

        logits < minimum_value,

        torch.full_like(
            logits,
            float("-inf")
        ),

        logits

    )


    return logits


def apply_top_p(
    logits,
    top_p
):

    if top_p >= 1.0:

        return logits


    if top_p <= 0:

        return logits


    sorted_logits, sorted_indices = torch.sort(

        logits,

        descending=True

    )


    sorted_probabilities = torch.softmax(

        sorted_logits,

        dim=-1

    )


    cumulative_probabilities = torch.cumsum(

        sorted_probabilities,

        dim=-1

    )


    remove_tokens = (

        cumulative_probabilities
        > top_p

    )


    remove_tokens[
        :,
        1:
    ] = remove_tokens[
        :,
        :-1
    ].clone()


    remove_tokens[
        :,
        0
    ] = False


    sorted_logits = sorted_logits.masked_fill(

        remove_tokens,

        float("-inf")

    )


    logits = torch.full_like(
        logits,
        float("-inf")
    )


    logits.scatter_(

        1,

        sorted_indices,

        sorted_logits

    )


    return logits


@app.post("/generate")
def generate(
    request: GenerateRequest
):

    try:

        if not request.prompt.strip():

            return {

                "success": False,

                "message":
                    "Prompt cannot be empty."

            }


        model, token_to_id = (
            load_model()
        )


        ids = encode(

            request.prompt,

            token_to_id

        )


        if len(ids) == 0:

            ids = [

                token_to_id["<BOS>"]

            ]


        generated_tokens = torch.tensor(

            [ids],

            dtype=torch.long,

            device=DEVICE

        )


        temperature = max(

            min(
                request.temperature,
                2.0
            ),

            0.1

        )


        max_tokens = max(

            min(
                request.maxTokens,
                512
            ),

            1

        )


        top_k = max(

            request.topK,

            0

        )


        top_p = max(

            min(
                request.topP,
                1.0
            ),

            0.1

        )


        repetition_penalty = max(

            request.repetitionPenalty,

            1.0

        )


        eos_id = token_to_id.get(
            "<EOS>"
        )


        bos_id = token_to_id.get(
            "<BOS>"
        )


        pad_id = token_to_id.get(
            "<PAD>"
        )


        unk_id = token_to_id.get(
            "<UNK>"
        )


        generated_count = 0


        with torch.no_grad():

            for _ in range(
                max_tokens
            ):


                input_for_model = (

                    generated_tokens[

                        :,

                        -BLOCK_SIZE:

                    ]

                )


                logits = model(

                    input_for_model

                )


                logits = logits[

                    :,

                    -1,

                    :

                ]


                logits = (

                    logits
                    / temperature

                )


                logits = apply_repetition_penalty(

                    logits,

                    generated_tokens,

                    repetition_penalty

                )


                if bos_id is not None:

                    logits[
                        :,
                        bos_id
                    ] = float(
                        "-inf"
                    )


                if pad_id is not None:

                    logits[
                        :,
                        pad_id
                    ] = float(
                        "-inf"
                    )


                if unk_id is not None:

                    logits[
                        :,
                        unk_id
                    ] = float(
                        "-inf"
                    )


                if eos_id is not None:

                    if generated_count < 3:

                        logits[
                            :,
                            eos_id
                        ] = float(
                            "-inf"
                        )


                logits = apply_top_k(

                    logits,

                    top_k

                )


                logits = apply_top_p(

                    logits,

                    top_p

                )


                probabilities = torch.softmax(

                    logits,

                    dim=-1

                )


                if (

                    torch.isnan(
                        probabilities
                    ).any()

                    or

                    torch.isinf(
                        probabilities
                    ).any()

                    or

                    probabilities.sum() <= 0

                ):

                    probabilities = torch.softmax(

                        logits,

                        dim=-1

                    )


                next_token = torch.multinomial(

                    probabilities,

                    num_samples=1

                )


                generated_tokens = torch.cat(

                    (

                        generated_tokens,

                        next_token

                    ),

                    dim=1

                )


                generated_count += 1


                if (

                    eos_id is not None

                    and

                    generated_count >= 3

                    and

                    next_token.item()
                    == eos_id

                ):

                    break


        generated_ids = generated_tokens[

            0

        ].tolist()


        generated_text = decode(

            generated_ids,

            token_to_id

        )


        return {

            "success": True,

            "text":
                generated_text

        }


    except Exception as error:

        print(
            "Generation error:",
            error
        )


        return {

            "success": False,

            "message":
                str(error)

        }