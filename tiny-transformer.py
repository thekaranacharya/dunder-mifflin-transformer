# Imports
import json
import torch
import torch.nn as nn
from torch.nn import functional

# -----------------
# Hyperparameters
batch_size = 32
block_size = 512
epochs = 6000
eval_interval = 300
learning_rate = 3e-4
device = "cuda:3" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 384
n_blocks = 20
n_heads = 16
dropout_rate = 0.2
# -----------------

torch.manual_seed(1337)

# Get and read the data
# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open("the-office-script.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Get unique characters in the dataset
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Create encoder and decoder to map characters to int and vice-a-versa
char_to_int = {char: i for i, char in enumerate(chars)}
int_to_char = {i: char for i, char in enumerate(chars)}
encode = lambda char_seq: [char_to_int[c] for c in char_seq]
decode = lambda tokens: "".join([int_to_char[token] for token in tokens])

# Define the tokens for the entire text
tokens = torch.tensor(encode(text), dtype=torch.long)

# Split tokens into train and val (90-10 split)
edge = int(0.9 * len(tokens))
tokens_train = tokens[:edge]
tokens_val = tokens[edge:]


def get_batch(which="train"):
    data = tokens_train if which == "train" else tokens_val
    random_start_indices = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in random_start_indices])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in random_start_indices])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model):
    result = {}
    model.eval()

    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x_batch, y_batch = get_batch(which=split)
            logits, loss = model(x_batch, y_batch)
            losses[i] = loss.item()
        result[split] = losses.mean()

    model.train()
    return result


class SelfAttentionHead(nn.Module):
    def __init__(self, n_embd, head_size):
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
    def __init__(self, n_embd, num_heads, head_size):
        super().__init__()
        self.attention_heads = nn.ModuleList(
            [SelfAttentionHead(n_embd, head_size) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.attention_heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd):
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
    def __init__(self, n_embd, num_heads):
        super().__init__()
        head_size = n_embd // num_heads
        self.multi_head_attention = MultiHeadedAttention(n_embd, num_heads, head_size)
        self.feed_forward = FeedForward(n_embd)
        self.layer_norm1 = nn.LayerNorm(n_embd)
        self.layer_norm2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.multi_head_attention(self.layer_norm1(x))  # Residual connections
        x = x + self.feed_forward(self.layer_norm2(x))  # Residual connections
        return x


# Train bigram language model (BASELINE)
class BigramLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_embedding = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding = nn.Embedding(block_size, n_embd)
        # self.att_head = SelfAttentionHead()
        # self.multi_att_head = MultiHeadAttention(n_embd // 4, 4)  # 4 attention heads of 6 head_size each
        # self.feed_forward = FeedForward()

        # self.decoder_blocks = nn.Sequential(
        #     DecoderBlock(n_embd, 4),
        #     DecoderBlock(n_embd, 4),
        #     DecoderBlock(n_embd, 4),
        #     nn.LayerNorm(n_embd)
        # )
        self.decoder_blocks = nn.Sequential(
            *[DecoderBlock(n_embd, n_heads) for _ in range(n_blocks)]
        )
        self.layer_norm = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, x, targets=None):
        B, T = x.shape
        tok_emb = self.tok_embedding(x)  # size: (B x T x n_embd)
        pos_emb = self.pos_embedding(
            torch.arange(T, device=device)
        )  # size: (T x n_embd)
        temp = tok_emb + pos_emb  # size: (B x T x n_embd)
        # temp = self.att_head(temp)
        # temp = self.multi_att_head(temp)
        # temp = self.feed_forward(temp)
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
            x_cropped = x[:, -block_size:]
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


blm = BigramLM()
blm = blm.to(device)

optimizer = torch.optim.AdamW(blm.parameters(), lr=learning_rate)

# Let's train the model now!
for k in range(epochs):

    # estimate the loss every eval_interval
    if k % eval_interval == 0:
        losses = estimate_loss(model=blm)
        print(
            f"Epoch {k}: train loss: {losses['train']:.3f}, val loss: {losses['val']:.3f}"
        )

    x_batch, y_batch = get_batch()  # default: sample from training data
    logits, loss = blm(x_batch, y_batch)  # forward pass and get loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()  # Backward pass
    optimizer.step()  # update params step


# Now let's test the predictions
prompts = ["MICHAEL:\n", "DWIGHT:\n", "JIM:\n", "PAM:\n", "\n"]
break_line = "#" * 50
with open("output.txt", "w") as op_file:
    for prompt in prompts:
        # Starting prompt: MICHAEL:\n
        start_char = torch.tensor(encode(prompt), dtype=torch.long, device=device)
        start_char = torch.reshape(start_char, (1, -1))
        # print(start_char.shape)
        next_chars = decode(blm.generate(x=start_char, num_new_tokens=500)[0].tolist())
        op_file.write(next_chars)
        op_file.write(f"\n{break_line}\n")