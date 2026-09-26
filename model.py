import torch
import torch.nn as nn


class TransformerBlock(nn.Module):

    def __init__(
        self,
        embedding_size,
        num_heads,
        block_size
    ):
        super().__init__()


        if embedding_size % num_heads != 0:

            raise ValueError(
                "embedding_size must be divisible by num_heads."
            )


        self.embedding_size = embedding_size

        self.num_heads = num_heads

        self.head_size = (
            embedding_size // num_heads
        )

        self.block_size = block_size


        self.layer_norm1 = nn.LayerNorm(
            embedding_size
        )

        self.layer_norm2 = nn.LayerNorm(
            embedding_size
        )


        self.query_layer = nn.Linear(
            embedding_size,
            embedding_size
        )

        self.key_layer = nn.Linear(
            embedding_size,
            embedding_size
        )

        self.value_layer = nn.Linear(
            embedding_size,
            embedding_size
        )

        self.attention_output = nn.Linear(
            embedding_size,
            embedding_size
        )


        self.ffn = nn.Sequential(

            nn.Linear(
                embedding_size,
                embedding_size * 4
            ),

            nn.GELU(),

            nn.Linear(
                embedding_size * 4,
                embedding_size
            )
        )


        mask = torch.tril(
            torch.ones(
                block_size,
                block_size
            )
        )


        self.register_buffer(
            "causal_mask",
            mask.view(
                1,
                1,
                block_size,
                block_size
            )
        )


    def forward(self, x):

        batch_size = x.shape[0]

        sequence_length = x.shape[1]


        x_norm = self.layer_norm1(
            x
        )


        Q = self.query_layer(
            x_norm
        )

        K = self.key_layer(
            x_norm
        )

        V = self.value_layer(
            x_norm
        )


        Q = Q.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_size
        )

        K = K.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_size
        )

        V = V.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_size
        )


        Q = Q.transpose(
            1,
            2
        )

        K = K.transpose(
            1,
            2
        )

        V = V.transpose(
            1,
            2
        )


        scores = (
            Q @ K.transpose(-2, -1)
        )


        scores = scores / (
            self.head_size ** 0.5
        )


        mask = self.causal_mask[
            :,
            :,
            :sequence_length,
            :sequence_length
        ]


        scores = scores.masked_fill(
            mask == 0,
            float("-inf")
        )


        attention_weights = torch.softmax(
            scores,
            dim=-1
        )


        output = (
            attention_weights @ V
        )


        output = output.transpose(
            1,
            2
        )


        output = output.contiguous().view(
            batch_size,
            sequence_length,
            self.embedding_size
        )


        output = self.attention_output(
            output
        )


        attention_output = (
            x + output
        )


        ffn_input = self.layer_norm2(
            attention_output
        )


        ffn_output = self.ffn(
            ffn_input
        )


        output = (
            attention_output
            + ffn_output
        )


        return output


class GPTModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_size,
        block_size,
        num_heads,
        num_layers
    ):
        super().__init__()


        self.block_size = block_size

        self.embedding_size = embedding_size


        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_size
        )


        self.position_embedding = nn.Embedding(
            block_size,
            embedding_size
        )


        self.transformer_blocks = nn.Sequential(

            *[
                TransformerBlock(
                    embedding_size,
                    num_heads,
                    block_size
                )

                for _ in range(
                    num_layers
                )
            ]
        )


        self.final_layer_norm = nn.LayerNorm(
            embedding_size
        )


        self.output_layer = nn.Linear(
            embedding_size,
            vocab_size,
            bias=False
        )


        self.apply(
            self._initialize_weights
        )


    def _initialize_weights(
        self,
        module
    ):

        if isinstance(
            module,
            nn.Linear
        ):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )


            if module.bias is not None:

                nn.init.zeros_(
                    module.bias
                )


        elif isinstance(
            module,
            nn.Embedding
        ):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )


    def forward(
        self,
        tokens
    ):

        sequence_length = tokens.shape[1]


        if sequence_length > self.block_size:

            raise ValueError(
                "Sequence is longer than block size."
            )


        token_vectors = (
            self.token_embedding(
                tokens
            )
        )


        positions = torch.arange(
            sequence_length,
            device=tokens.device
        )


        position_vectors = (
            self.position_embedding(
                positions
            )
        )


        x = (
            token_vectors
            + position_vectors
        )


        x = self.transformer_blocks(
            x
        )


        x = self.final_layer_norm(
            x
        )


        logits = self.output_layer(
            x
        )


        return logits