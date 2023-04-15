# Imports
import json
import os
import torch
import torch.nn as nn
from torch.nn import functional
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# torch.manual_seed(1337)  # Set global seed
# Set visible GPU devices
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3"


class SelfAttentionHead(nn.Module):
    def __init__(self, n_embd, head_size, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k, q, v = (
            self.key(x),
            self.query(x),
            self.value(x),
        )  # all sizes: (B, T, head_size)
        wei = (
            q @ k.transpose(1, 2)
        ) / C**0.5  # (B, T, head_size) @ (B, head_size, T) ---> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = functional.softmax(wei, dim=-1)  # (B, T, T)
        out = wei @ v  #  (B, T, T) @ (B, T, head_size) --> (B, T, head_size)
        return out  # (B, T, head_size)


class MultiHeadedAttention(nn.Module):
    def __init__(self, n_embd, num_heads, head_size, block_size, dropout_rate):
        super().__init__()
        self.attention_heads = nn.ModuleList(
            [SelfAttentionHead(n_embd, head_size, block_size) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.attention_heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout_rate):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),  # proj
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        return self.network(x)


class DecoderBlock(nn.Module):
    def __init__(self, n_embd, num_heads, block_size, dropout_rate):
        super().__init__()
        head_size = n_embd // num_heads
        self.multi_head_attention = MultiHeadedAttention(
            n_embd, num_heads, head_size, block_size, dropout_rate
        )
        self.feed_forward = FeedForward(n_embd, dropout_rate)
        self.layer_norm1 = nn.LayerNorm(n_embd)
        self.layer_norm2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.multi_head_attention(self.layer_norm1(x))  # Residual connections
        x = x + self.feed_forward(self.layer_norm2(x))  # Residual connections
        return x


# Bigram language model
class BigramLM(nn.Module):
    def __init__(
        self, vocab_size, n_embd, block_size, n_heads, n_blocks, device, dropout_rate
    ):
        super().__init__()

        self.tok_embedding = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding = nn.Embedding(block_size, n_embd)
        self.decoder_blocks = nn.Sequential(
            *[
                DecoderBlock(n_embd, n_heads, block_size, dropout_rate)
                for _ in range(n_blocks)
            ]
        )
        self.layer_norm = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.block_size = block_size
        self.device = device

    def forward(self, x, targets=None):
        B, T = x.shape
        tok_emb = self.tok_embedding(x)  # size: (B x T x n_embd)
        pos_emb = self.pos_embedding(
            torch.arange(T, device=self.device)
        )  # size: (T x n_embd)
        # pos_emb = self.pos_embedding(torch.arange(T))  # size: (T x n_embd)
        temp = tok_emb + pos_emb  # size: (B x T x n_embd)

        temp = self.decoder_blocks(temp)
        temp = self.layer_norm(temp)
        logits = self.lm_head(temp)  # size: (B x T x vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)  # 2-D
            targets = targets.view(B * T)  # 1-D
            loss = functional.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, x, num_new_tokens):
        for _ in range(num_new_tokens):
            # Crop x
            x_cropped = x[:, -self.block_size :]

            # Get the logits from the current x
            logits = self(x_cropped)[0]  # shape: B x T x C

            # only last time step actually gives pred for next word
            # Check previous cell
            logits = logits[:, -1, :]  # shape: B x C

            # Apply softmax
            probs = functional.softmax(logits, dim=-1)  # shape: B x C

            # Get next word prediction based on probs
            x_next = torch.multinomial(probs, num_samples=1)  # shape: B x 1

            # Concat with current x
            # H -> e = He -> l = Hel -> l = Hell -> o
            x = torch.cat((x, x_next), dim=1)
        return x


class ScriptWriter:
    def __init__(self, params, text):
        self.params = params
        self.batch_size = int(params["batch_size"])
        self.block_size = int(params["block_size"])
        self.epochs = int(params["epochs"])
        self.eval_interval = int(params["eval_interval"])
        self.lr = float(params["learning_rate"])
        self.eval_iters = int(params["eval_iters"])
        self.n_embd = int(params["n_embd"])
        self.n_blocks = int(params["n_blocks"])
        self.n_heads = int(params["n_heads"])
        self.dropout_rate = float(params["dropout_rate"])
        self.text = text
        self.device = f"cuda" if torch.cuda.is_available() else "cpu"

    def create_model_input(self):
        # Get unique characters in the dataset
        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)

        # Create encoder and decoder to map characters to int and vice-a-versa
        char_to_int = {char: i for i, char in enumerate(self.chars)}
        int_to_char = {i: char for i, char in enumerate(self.chars)}

        self.encode = lambda char_seq: [char_to_int[c] for c in char_seq]
        self.decode = lambda tokens: "".join([int_to_char[token] for token in tokens])

        # Define the tokens for the entire text
        self.tokens = torch.tensor(self.encode(text), dtype=torch.long)

        # Split tokens into train and val (90-10 split)
        edge = int(0.9 * len(self.tokens))
        self.tokens_train = self.tokens[:edge]
        self.tokens_val = self.tokens[edge:]

    def get_batch(self, which="train"):
        data = self.tokens_train if which == "train" else self.tokens_val
        random_start_indices = torch.randint(
            len(data) - self.block_size, (self.batch_size,)
        )
        x = torch.stack([data[i : i + self.block_size] for i in random_start_indices])
        y = torch.stack(
            [data[i + 1 : i + self.block_size + 1] for i in random_start_indices]
        )
        x, y = x.to(self.device), y.to(self.device)
        return x, y

    @torch.no_grad()
    def estimate_loss(self):
        result = {}
        self.model.eval()

        for split in ("train", "val"):
            losses = torch.zeros(self.eval_iters)
            for i in range(self.eval_iters):
                x_batch, y_batch = self.get_batch(which=split)
                logits, loss = self.model(x_batch, y_batch)
                # print(type(loss), loss.shape)
                losses[i] = loss.item()
            result[split] = losses.mean()

        self.model.train()
        return result

    def fit(self):
        # Define the model
        model = BigramLM(
            self.vocab_size,
            self.n_embd,
            self.block_size,
            self.n_heads,
            self.n_blocks,
            self.device,
            self.dropout_rate,
        )

        self.model = model
        self.model.to(self.device)
        # self.model = model

        # Set the optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

        # Let's train the model now!
        for epoch in range(self.epochs):
            # estimate the loss every eval_interval
            if epoch % self.eval_interval == 0:
                losses = self.estimate_loss()
                print(
                    f"Epoch {epoch}: train loss: {losses['train']:.3f}, val loss: {losses['val']:.3f}"
                )

            # do one forward pass and one backward pass for one batch
            (
                x_batch,
                y_batch,
            ) = self.get_batch()  # default: sample from training data
            logits, loss = self.model(x_batch, y_batch)  # forward pass and get loss
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()  # Backward pass
            self.optimizer.step()  # update params step

    def predict(self, output_file_path):
        # Now let's test the predictions
        prompts = ["MICHAEL:\n", "DWIGHT:\n", "JIM:\n", "PAM:\n", "\n"]
        break_line = "#" * 50
        dashed_line = "+-" * 100
        with open(output_file_path, "w") as op_file:
            for param in self.params:
                op_file.write(f"{param} --> {self.params[param]}\n")

            op_file.write(f"\n{dashed_line}\n")

            for prompt in prompts:
                # Starting prompt: MICHAEL:\n
                start_char = torch.tensor(
                    self.encode(prompt), dtype=torch.long, device=self.device
                )
                # start_char = torch.tensor(self.encode(prompt), dtype=torch.long)
                start_char = torch.reshape(start_char, (1, -1))
                next_chars = self.decode(
                    self.model.generate(x=start_char, num_new_tokens=500)[0].tolist()
                )
                op_file.write(next_chars)
                op_file.write(f"\n{break_line}\n")


if __name__ == "__main__":
    # Read params
    with open("params.json") as params_file:
        params = json.load(params_file)

    # Read text
    with open("the-office-script.txt", "r", encoding="utf-8") as f:
        text = f.read()

    script_writer = ScriptWriter(params, text)
    script_writer.create_model_input()
    script_writer.fit()
    script_writer.predict(params["op_file_name"])
