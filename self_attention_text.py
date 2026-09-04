################## Task 1: Setup and Data Preparation ##################

import re
from collections import Counter
from torch.utils.data import TensorDataset, DataLoader
import torch
from datasets import load_dataset
import torch.nn as nn
import math
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

# 1. Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Selected device:", device)

# 2. Load AG News dataset
# AG News is a text classification dataset with 4 news categories: World, Sports, Business, Sci/Tech.
dataset = load_dataset("ag_news")

train_data = dataset["train"]
test_data = dataset["test"]

print("Train dataset size:", len(train_data))
print("Test dataset size:", len(test_data))

# 3. Basic tokenizer

# This tokenizer:
# - converts text to lowercase
# - keeps only words and numbers
# - removes punctuation
#
# Example:
# "Apple releases new iPhone!" -> ["apple", "releases", "new", "iphone"]
def tokenize(text):
    text = text.lower()
    tokens = re.findall(r"\b\w+\b", text)
    return tokens

# 4. Tokenize the training texts
# We first tokenize all training texts because the vocabulary should be built only from the training data, not from the test data
train_tokens = [tokenize(example["text"]) for example in train_data]

# 5. Build vocabulary
# We count how often each token appears in the training set
# Then we keep the most frequent words
#
# Special tokens:
# <PAD> = used to fill shorter sequences
# <UNK> = used for words that are not in the vocabulary
MAX_VOCAB_SIZE = 20000

counter = Counter()

for tokens in train_tokens:
    counter.update(tokens)

vocab = {
    "<PAD>": 0,
    "<UNK>": 1,
}

# Start indexing normal words from 2 because 0 and 1 are reserved
for word, _ in counter.most_common(MAX_VOCAB_SIZE - 2):
    vocab[word] = len(vocab)

print("Vocabulary size:", len(vocab))

# 6. Convert tokens to integer IDs

# Neural networks cannot process raw words directly, every token is mapped to an integer ID.
def tokens_to_ids(tokens, vocab):
    return [vocab.get(token, vocab["<UNK>"]) for token in tokens]

# 7. Padding and truncation

# All input sequences must have the same length in a batch.
# If a text is shorter than MAX_LEN, we add <PAD> tokens, if longer, we cut it to MAX_LEN
MAX_LEN = 50

def pad_or_truncate(token_ids, max_len):
    if len(token_ids) > max_len:
        return token_ids[:max_len]
    else:
        padding_length = max_len - len(token_ids)
        return token_ids + [vocab["<PAD>"]] * padding_length


# 8. Encode one text sample

example_text = train_data[0]["text"]
example_label = train_data[0]["label"]

example_tokens = tokenize(example_text)
example_ids = tokens_to_ids(example_tokens, vocab)
example_padded = pad_or_truncate(example_ids, MAX_LEN)

print("\nExample raw text:")
print(example_text)

print("\nExample label:")
print(example_label)

print("\nExample tokenized text:")
print(example_tokens)

print("\nExample token IDs:")
print(example_ids)

print("\nExample padded/truncated IDs:")
print(example_padded)

print("\nLength after padding/truncation:")
print(len(example_padded))


# 9. Dataset statistics
# The maximum original token sequence length tells us how long the longest text is before truncation
all_train_lengths = [len(tokens) for tokens in train_tokens]
max_sequence_length = max(all_train_lengths)

# AG News has 4 unique classes
unique_classes = set(example["label"] for example in train_data)

print("\nMaximum original token sequence length:", max_sequence_length)
print("Number of unique classes:", len(unique_classes))
print("Class labels:", sorted(unique_classes))

# 10. Encode full train and test datasets
# We prepare tensors that can later be used for training
def encode_dataset(split):
    input_ids = []
    labels = []

    for example in split:
        tokens = tokenize(example["text"])
        token_ids = tokens_to_ids(tokens, vocab)
        padded_ids = pad_or_truncate(token_ids, MAX_LEN)

        input_ids.append(padded_ids)
        labels.append(example["label"])

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)

    return input_ids, labels


train_input_ids, train_labels = encode_dataset(train_data)
test_input_ids, test_labels = encode_dataset(test_data)

print("\nEncoded train input shape:", train_input_ids.shape)
print("Encoded train labels shape:", train_labels.shape)
print("Encoded test input shape:", test_input_ids.shape)
print("Encoded test labels shape:", test_labels.shape)


################# Task 2: Implementing Self-Attention ##################
# Part 1: Create DataLoaders

# 1. Subsample the training data

TRAIN_SUBSET_SIZE = 10000

train_input_ids_small = train_input_ids[:TRAIN_SUBSET_SIZE]
train_labels_small = train_labels[:TRAIN_SUBSET_SIZE]

# 2. Create TensorDataset objects

# TensorDataset combines inputs and labels so that each sample contains: (input_ids, label)
train_dataset = TensorDataset(train_input_ids_small, train_labels_small)
test_dataset = TensorDataset(test_input_ids, test_labels)

# 3. Create DataLoaders
# DataLoader splits the dataset into batches
# shuffle=True is used for training so the model does not see examples in the same order every epoch
BATCH_SIZE = 64

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("\nNumber of training batches:", len(train_loader))
print("Number of test batches:", len(test_loader))

# Part 2: Embedding Layer + Positional Embeddings

# 1. Define embedding dimensions
# EMBED_DIM determines how many numbers are used to represent each token

# Example:
# token_id = 451 -> embedding vector = [0.12, -0.88, 1.45, ...]
EMBED_DIM = 128

# 2. Token embedding layer
# nn.Embedding creates a trainable lookup table.

# Shape: (vocab_size, embedding_dim) - (20000, 128)

# every token in the vocabulary gets its own 128-dimensional vector
token_embedding = nn.Embedding(num_embeddings=len(vocab), embedding_dim=EMBED_DIM)

# 3. Positional embedding layer

# Self-attention alone does NOT understand word order

# Example: "dog bites man" and "man bites dog" would contain the same tokens.
# To fix this, we add positional embeddings that tell the model where each token is located in the sequence.
position_embedding = nn.Embedding(num_embeddings=MAX_LEN, embedding_dim=EMBED_DIM)

# 4. Get one batch from the DataLoader
batch_input_ids, batch_labels = next(iter(train_loader))

print("\nBatch input shape:")
print(batch_input_ids.shape)

print("\nBatch labels shape:")
print(batch_labels.shape)

# 5. Create token embeddings
# Input: (batch_size, sequence_length)
# Output: (batch_size, sequence_length, embedding_dim)
token_embeddings = token_embedding(batch_input_ids)

print("\nToken embeddings shape:")
print(token_embeddings.shape)

# 6. Create positional embeddings
# We create positions: [0, 1, 2, ..., 49]
positions = torch.arange(MAX_LEN)

# Convert positions into embeddings
position_embeddings = position_embedding(positions)

print("\nPosition embeddings shape:")
print(position_embeddings.shape)

# 7. Combine token + positional embeddings
# The model now knows:
# - which token it sees
# - where the token is located
input_embeddings = token_embeddings + position_embeddings

print("\nFinal input embeddings shape:")
print(input_embeddings.shape)

# Part 3: Single-Head Self-Attention
class SingleHeadSelfAttention(nn.Module):
    """
    Minimal single-head self-attention layer.

    Input:
        x: tensor of shape (batch_size, seq_len, embed_dim)
        mask: tensor of shape (batch_size, seq_len)

    Output:
        attended_output: tensor of shape (batch_size, seq_len, embed_dim)
        attention_weights: tensor of shape (batch_size, seq_len, seq_len)
    """

    def __init__(self, embed_dim):
        super().__init__()

        self.embed_dim = embed_dim

        # These three linear layers learn how to transform input embeddings into queries, keys, and values
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        
        # 1. Compute Q, K, V
        # Shape of x: (batch_size, seq_len, embed_dim)
        # Shape of Q, K, V: (batch_size, seq_len, embed_dim)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # 2. Compute raw attention scores
        # We compare every query token with every key token
        # K.transpose(-2, -1) changes shape from: (batch_size, seq_len, embed_dim) to: (batch_size, embed_dim, seq_len)
        
        # Resulting scores shape: (batch_size, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1))

        # Scale scores to avoid extremely large values before softmax
        scores = scores / math.sqrt(self.embed_dim)

        # 3. Apply padding mask
        # Padding tokens should not receive attention
        # mask shape: (batch_size, seq_len)
        # We convert it to: (batch_size, 1, seq_len) so it can be applied to every query token
        if mask is not None:
            mask = mask.unsqueeze(1)

            # Positions where mask == 0 are padding tokens
            # We set their scores to a very negative value, so after softmax they become almost 0
            scores = scores.masked_fill(mask == 0, -1e9)

        # 4. Convert scores to attention weights
        # Softmax turns raw scores into probabilities
        # For each token, attention weights sum to 1
        attention_weights = F.softmax(scores, dim=-1)

        # 5. Compute attended output
        # We use the attention weights to take a weighted average of the value vectors
        attended_output = torch.matmul(attention_weights, V)

        return attended_output, attention_weights

# Test the self-attention layer on one batch

attention_layer = SingleHeadSelfAttention(embed_dim=EMBED_DIM)

# Create mask from input IDs:
# 1 = real token
# 0 = padding token
batch_mask = (batch_input_ids != vocab["<PAD>"]).long()

attended_output, attention_weights = attention_layer(input_embeddings, mask=batch_mask)

print("\nSelf-attention output shape:")
print(attended_output.shape)

print("\nAttention weights shape:")
print(attention_weights.shape)


# Part 4: Build a Simple Self-Attention Classifier

class SelfAttentionTextClassifier(nn.Module):
    """
    Simple text classification model using one self-attention layer.

    Pipeline:
    token IDs
    -> token embeddings
    -> positional embeddings
    -> self-attention
    -> mean pooling
    -> MLP classifier
    """

    def __init__(self, vocab_size, max_len, embed_dim, num_classes, pad_idx):
        super().__init__()

        self.pad_idx = pad_idx

        # Converts token IDs into trainable embedding vectors
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx
        )

        # Adds information about token positions
        self.position_embedding = nn.Embedding(
            num_embeddings=max_len,
            embedding_dim=embed_dim
        )

        # Our manually implemented self-attention layer
        self.attention = SingleHeadSelfAttention(embed_dim)

        # Small MLP classifier head
        # It maps the final text representation to class logits
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, input_ids):
        # input_ids shape: (batch_size, seq_len)
        batch_size, seq_len = input_ids.shape

        # 1. Create padding mask
    
        # 1 = real token, 0 = padding token
        mask = (input_ids != self.pad_idx).long()

        # 2. Token embeddings
        # Shape: (batch_size, seq_len) -> (batch_size, seq_len, embed_dim)
        token_emb = self.token_embedding(input_ids)

        # 3. Positional embeddings
        # positions: [0, 1, 2, ..., seq_len - 1]
        positions = torch.arange(seq_len, device=input_ids.device)

        # Shape: (seq_len,) -> (seq_len, embed_dim)
        pos_emb = self.position_embedding(positions)

        # Broadcasting automatically adds positional embeddings to every sequence in the batch
        x = token_emb + pos_emb

        # 4. Self-attention
        # attention_output shape: (batch_size, seq_len, embed_dim)
        attention_output, attention_weights = self.attention(x, mask=mask)

        # 5. Mean pooling over real tokens only

        # We want one vector for the whole text
        # Padding tokens should not influence the average
        mask_expanded = mask.unsqueeze(-1)

        summed = (attention_output * mask_expanded).sum(dim=1)
        counts = mask_expanded.sum(dim=1).clamp(min=1)

        pooled = summed / counts

        # 6. Classifier
        # logits shape: (batch_size, num_classes)
        logits = self.classifier(pooled)

        return logits, attention_weights


# Create the model

NUM_CLASSES = 4
PAD_IDX = vocab["<PAD>"]

model = SelfAttentionTextClassifier(
    vocab_size=len(vocab),
    max_len=MAX_LEN,
    embed_dim=EMBED_DIM,
    num_classes=NUM_CLASSES,
    pad_idx=PAD_IDX
).to(device)

print("\nModel architecture:")
print(model)

# Test one forward pass

batch_input_ids = batch_input_ids.to(device)
batch_labels = batch_labels.to(device)

logits, attention_weights = model(batch_input_ids)

print("\nLogits shape:")
print(logits.shape)

print("\nAttention weights shape from full model:")
print(attention_weights.shape)


# Part 5: Training Loop

# 1. Loss function
criterion = nn.CrossEntropyLoss()

# 2. Optimizer
# The optimizer updates model weights during training.
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 3. Training settings
NUM_EPOCHS = 5

# 4. Training loop

start_time = time.time()

for epoch in range(NUM_EPOCHS):
    # Put model into training mode

    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    # Iterate over training batches
    for batch_input_ids, batch_labels in train_loader:

        # Move batch to GPU/CPU
        batch_input_ids = batch_input_ids.to(device)
        batch_labels = batch_labels.to(device)

        # Forward pass
        # logits shape: (batch_size, num_classes)
        logits, _ = model(batch_input_ids)
        
        # Compute loss
        loss = criterion(logits, batch_labels)

        # Backpropagation
        # Clear old gradients
        optimizer.zero_grad()

        # Compute gradients
        loss.backward()

        # Update model weights
        optimizer.step()

        # Statistics
        total_loss += loss.item()

        # Predicted class = highest logit
        predictions = logits.argmax(dim=1)
        correct_predictions += (predictions == batch_labels).sum().item()
        total_samples += batch_labels.size(0)

    # Epoch metrics
    avg_loss = total_loss / len(train_loader)
    accuracy = correct_predictions / total_samples

    print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
    print(f"Train loss: {avg_loss:.4f}")
    print(f"Train accuracy: {accuracy:.4f}")


# 5. Training time
end_time = time.time()
training_time = end_time - start_time
print(f"\nTraining time: {training_time:.2f} seconds")

# Evaluation on Test Set

model.eval()

correct_predictions = 0
total_samples = 0

with torch.no_grad():
    for batch_input_ids, batch_labels in test_loader:
        batch_input_ids = batch_input_ids.to(device)
        batch_labels = batch_labels.to(device)
        logits, _ = model(batch_input_ids)
        predictions = logits.argmax(dim=1)
        correct_predictions += (predictions == batch_labels).sum().item()
        total_samples += batch_labels.size(0)

test_accuracy = correct_predictions / total_samples

print(f"\nTest accuracy: {test_accuracy:.4f}")


################# Task 3: Attention Visualization ##################

################# Task 3: Attention Visualization ##################

import matplotlib.pyplot as plt

# Reverse vocabulary: token ID -> token string
id_to_token = {idx: token for token, idx in vocab.items()}


def plot_attention_heatmap(input_ids, attention_matrix, title, save_path):
    """
    Plot and save a token-to-token attention heatmap.

    The function removes padding tokens so that the heatmap only shows
    the real text tokens.
    """

    # Move tensors to CPU because matplotlib works with CPU arrays
    input_ids = input_ids.cpu()
    attention_matrix = attention_matrix.detach().cpu()

    # Count how many tokens are real tokens, not <PAD>
    valid_length = (input_ids != PAD_IDX).sum().item()

    # Keep only real tokens and remove padding
    input_ids = input_ids[:valid_length]
    attention_matrix = attention_matrix[:valid_length, :valid_length]

    # Convert token IDs back to readable tokens
    tokens = [
        id_to_token.get(token_id.item(), "<UNK>")
        for token_id in input_ids
    ]

    print(f"\nTokens for {title}:")
    print(tokens)

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(attention_matrix, cmap="viridis")
    plt.colorbar()

    plt.xticks(ticks=range(valid_length), labels=tokens, rotation=90)
    plt.yticks( ticks=range(valid_length), labels=tokens)

    plt.title(title)
    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")

    plt.tight_layout()

    # Save image so it can be included in the Markdown report
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# Select two test examples from different classes

example_indices = []
seen_labels = set()

for i in range(len(test_labels)):
    label = test_labels[i].item()

    # Take the first example of a label we have not used yet
    if label not in seen_labels:
        example_indices.append(i)
        seen_labels.add(label)

    # Stop after selecting two different classes
    if len(example_indices) == 2:
        break

print("\nSelected example indices:", example_indices)
print("Selected true labels:", [test_labels[i].item() for i in example_indices])

# Run the trained model and plot attention heatmaps
model.eval()

for plot_id, idx in enumerate(example_indices, start=1):
    # Take one test example and add a batch dimension
    single_input = test_input_ids[idx].unsqueeze(0).to(device)
    true_label = test_labels[idx].item()

    with torch.no_grad():
        logits, attention_weights = model(single_input)

    predicted_label = logits.argmax(dim=1).item()

    plot_attention_heatmap(
        input_ids=single_input[0],
        attention_matrix=attention_weights[0],
        title=f"Attention Heatmap {plot_id} | True label: {true_label}, Predicted: {predicted_label}",
        save_path=f"attention_heatmap_{plot_id}.png"
    )

# 2D Projection of Sequence Embeddings

def extract_sequence_embeddings(model, data_loader, max_batches=20):

    """
    Extract pooled sequence embeddings from the trained model.

    We use the model up to the self-attention + pooling step,
    but before the final classifier.
    """
    model.eval()

    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (batch_input_ids, batch_labels) in enumerate(data_loader):
            if batch_idx >= max_batches:
                break

            batch_input_ids = batch_input_ids.to(device)

            # Create padding mask
            mask = (batch_input_ids != PAD_IDX).long()

            # Token embeddings
            token_emb = model.token_embedding(batch_input_ids)

            # Positional embeddings
            batch_size, seq_len = batch_input_ids.shape
            positions = torch.arange(seq_len, device=batch_input_ids.device)
            pos_emb = model.position_embedding(positions)

            # Token meaning + position information
            x = token_emb + pos_emb

            # Self-attention output
            attention_output, _ = model.attention(x, mask=mask)

            # Mean pooling over real tokens only
            mask_expanded = mask.unsqueeze(-1)

            summed = (attention_output * mask_expanded).sum(dim=1)
            counts = mask_expanded.sum(dim=1).clamp(min=1)

            pooled = summed / counts

            all_embeddings.append(pooled.cpu())
            all_labels.append(batch_labels)

    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    return all_embeddings, all_labels

# Extract embeddings from the test set

sequence_embeddings, embedding_labels = extract_sequence_embeddings(
    model,
    test_loader,
    max_batches=20
)

print("\nSequence embeddings shape:")
print(sequence_embeddings.shape)
print("\nEmbedding labels shape:")
print(embedding_labels.shape)

# Project embeddings to 2D using t-SNE

tsne = TSNE(n_components=2, random_state=42, perplexity=30, init="pca", learning_rate="auto")
embeddings_2d = tsne.fit_transform(sequence_embeddings)

print("\nt-SNE output shape:")
print(embeddings_2d.shape)

# Plot 2D embeddings
plt.figure(figsize=(9, 7))

scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=embedding_labels, alpha=0.7)

plt.colorbar(scatter, label="Class label")
plt.title("t-SNE Projection of Learned Sequence Embeddings")
plt.xlabel("t-SNE dimension 1")
plt.ylabel("t-SNE dimension 2")

plt.tight_layout()
plt.savefig("sequence_embeddings_tsne.png", dpi=300, bbox_inches="tight")
plt.show()