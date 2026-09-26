import os

import torch
import torch.nn as nn

from model import GPTModel

from dataset import (
    create_data,
    get_batch
)

from tokenizer import (
    create_vocabulary,
    save_vocabulary
)


PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)


MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "model.pth"
)


BEST_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "best_model.pth"
)


CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT,
    "checkpoint.pth"
)


EMBEDDING_SIZE = 256

BLOCK_SIZE = 128

NUM_HEADS = 8

NUM_LAYERS = 8


BATCH_SIZE = 8

GRAD_ACCUMULATION_STEPS = 4


TRAIN_STEPS = 50000

LEARNING_RATE = 0.0003

WEIGHT_DECAY = 0.01


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print(
    "=" * 50
)

print(
    "GPT TRAINING"
)

print(
    "=" * 50
)

print(
    "Device:",
    DEVICE
)


if DEVICE == "cuda":

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "VRAM:",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / 1024**3,
            2
        ),
        "GB"
    )


print(
    "=" * 50
)


def evaluate_loss(
    model,
    data,
    eval_steps=20
):

    model.eval()

    losses = []

    with torch.no_grad():

        for _ in range(
            eval_steps
        ):

            x, y = get_batch(
                data,
                BLOCK_SIZE,
                BATCH_SIZE,
                DEVICE
            )

            with torch.amp.autocast(
                device_type=DEVICE,
                dtype=torch.float16,
                enabled=(
                    DEVICE == "cuda"
                )
            ):

                logits = model(
                    x
                )

                logits = logits.view(
                    -1,
                    logits.size(-1)
                )

                y = y.view(
                    -1
                )

                loss = nn.CrossEntropyLoss()(
                    logits,
                    y
                )

            losses.append(
                loss.item()
            )

    model.train()

    return (
        sum(losses)
        / len(losses)
    )


def train_model(
    text,
    steps=TRAIN_STEPS,
    learning_rate=LEARNING_RATE
):

    if not text or not text.strip():

        raise ValueError(
            "Training text is empty."
        )


    if steps < 1:

        raise ValueError(
            "Training steps must be at least 1."
        )


    if learning_rate <= 0:

        raise ValueError(
            "Learning rate must be greater than 0."
        )


    print(
        "\nCreating vocabulary..."
    )


    vocab = create_vocabulary(
        text
    )


    save_vocabulary(
        vocab
    )


    print(
        "Vocabulary size:",
        len(vocab)
    )


    print(
        "\nCreating token data..."
    )


    data = create_data(
        text,
        vocab
    )


    print(
        "Total tokens:",
        len(data)
    )


    minimum_tokens = (
        BLOCK_SIZE + 1
    )


    if len(data) < minimum_tokens:

        raise ValueError(
            f"Not enough training data. "
            f"Need at least {minimum_tokens} tokens."
        )


    # -------------------------------------------------
    # FIXED TRAIN / VALIDATION SPLIT
    # -------------------------------------------------

    minimum_validation_tokens = (
        BLOCK_SIZE + 1
    )

    minimum_training_tokens = (
        BLOCK_SIZE + 1
    )

    minimum_total_tokens = (
        minimum_training_tokens
        + minimum_validation_tokens
    )


    if len(data) < minimum_total_tokens:

        raise ValueError(
            f"Training data is too small. "
            f"Need at least {minimum_total_tokens} tokens."
        )


    # Keep at least 129 tokens for validation.
    validation_size = max(
        minimum_validation_tokens,
        int(len(data) * 0.10)
    )


    # Make sure training data also has enough tokens.
    maximum_validation_size = (
        len(data)
        - minimum_training_tokens
    )


    validation_size = min(
        validation_size,
        maximum_validation_size
    )


    split_index = (
        len(data)
        - validation_size
    )


    train_data = data[
        :split_index
    ]


    val_data = data[
        split_index:
    ]


    if len(train_data) < minimum_training_tokens:

        raise ValueError(
            "Training portion is too small."
        )


    if len(val_data) < minimum_validation_tokens:

        raise ValueError(
            "Validation portion is too small."
        )


    print(
        "Training tokens:",
        len(train_data)
    )


    print(
        "Validation tokens:",
        len(val_data)
    )


    print(
        "\nCreating GPT model..."
    )


    model = GPTModel(

        vocab_size=len(vocab),

        embedding_size=EMBEDDING_SIZE,

        block_size=BLOCK_SIZE,

        num_heads=NUM_HEADS,

        num_layers=NUM_LAYERS

    ).to(
        DEVICE
    )


    total_params = sum(

        p.numel()

        for p in model.parameters()

    )


    print(
        "Parameters:",
        f"{total_params:,}"
    )


    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=learning_rate,

        weight_decay=WEIGHT_DECAY

    )


    criterion = nn.CrossEntropyLoss()


    scaler = torch.amp.GradScaler(

        "cuda",

        enabled=(
            DEVICE == "cuda"
        )

    )


    best_val_loss = float(
        "inf"
    )


    print(
        "\nStarting training..."
    )


    print(
        "=" * 50
    )


    optimizer.zero_grad(
        set_to_none=True
    )


    for step in range(
        1,
        steps + 1
    ):


        total_loss = 0.0


        for _ in range(
            GRAD_ACCUMULATION_STEPS
        ):


            x, y = get_batch(

                train_data,

                BLOCK_SIZE,

                BATCH_SIZE,

                DEVICE

            )


            with torch.amp.autocast(

                device_type=DEVICE,

                dtype=torch.float16,

                enabled=(
                    DEVICE == "cuda"
                )

            ):


                logits = model(
                    x
                )


                logits = logits.view(

                    -1,

                    logits.size(-1)

                )


                y = y.view(
                    -1
                )


                loss = criterion(

                    logits,

                    y

                )


                loss = (

                    loss

                    / GRAD_ACCUMULATION_STEPS

                )


            scaler.scale(
                loss
            ).backward()


            total_loss += loss.item()


        scaler.unscale_(
            optimizer
        )


        torch.nn.utils.clip_grad_norm_(

            model.parameters(),

            max_norm=1.0

        )


        scaler.step(
            optimizer
        )


        scaler.update()


        optimizer.zero_grad(

            set_to_none=True

        )


        if (

            step == 1

            or step % 500 == 0

            or step == steps

        ):


            train_loss = (
                total_loss
            )


            val_loss = evaluate_loss(

                model,

                val_data,

                eval_steps=5

            )


            print(

                f"Step {step}/{steps} "

                f"| Train Loss: {train_loss:.4f} "

                f"| Val Loss: {val_loss:.4f}"

            )


            if DEVICE == "cuda":

                allocated = (

                    torch.cuda.memory_allocated()

                    / 1024**3

                )


                reserved = (

                    torch.cuda.memory_reserved()

                    / 1024**3

                )


                print(

                    f"GPU Memory: "

                    f"{allocated:.2f} GB allocated / "

                    f"{reserved:.2f} GB reserved"

                )


            if val_loss < best_val_loss:

                best_val_loss = val_loss


                torch.save(

                    {

                        "model_state_dict":
                            model.state_dict(),

                        "optimizer_state_dict":
                            optimizer.state_dict(),

                        "step":
                            step,

                        "val_loss":
                            val_loss,

                        "vocab_size":
                            len(vocab),

                        "embedding_size":
                            EMBEDDING_SIZE,

                        "block_size":
                            BLOCK_SIZE,

                        "num_heads":
                            NUM_HEADS,

                        "num_layers":
                            NUM_LAYERS

                    },

                    BEST_MODEL_PATH

                )


                print(
                    "Best model saved."
                )


            torch.save(

                {

                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "step":
                        step,

                    "val_loss":
                        val_loss

                },

                CHECKPOINT_PATH

            )


            print(
                "Checkpoint saved."
            )


            print(
                "-" * 50
            )


    torch.save(

        model.state_dict(),

        MODEL_PATH

    )


    print(
        "\nTraining completed!"
    )


    print(
        "Final model:",
        MODEL_PATH
    )


    print(
        "Best model:",
        BEST_MODEL_PATH
    )


    print(
        "Checkpoint:",
        CHECKPOINT_PATH
    )


    return {

        "steps":
            steps,

        "loss":
            best_val_loss,

        "vocab_size":
            len(vocab)

    }


if __name__ == "__main__":

    DATA_PATH = os.path.join(

        PROJECT_ROOT,

        "data",

        "input.txt"

    )


    if not os.path.exists(
        DATA_PATH
    ):

        raise FileNotFoundError(

            f"Training file not found: "
            f"{DATA_PATH}"

        )


    print(
        "\nLoading training data..."
    )


    with open(

        DATA_PATH,

        "r",

        encoding="utf-8"

    ) as file:

        text = file.read()


    print(
        "Characters:",
        len(text)
    )


    train_model(
        text
    )