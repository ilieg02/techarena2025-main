import os
import json
import random
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

OUT = "biencoder"
os.makedirs(OUT, exist_ok=True)

# Configurations
BI_MODEL = "all-mpnet-base-v2"  # Replace with Qwen-compatible bi-encoder if available
BATCH = 16
EPOCHS = 2
LR = 2e-5
MAX_LEN = 256
HARD_NEGS = 3

# Helper function to load JSON lines
def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

# Load chunks and create training pairs
chunks = load_jsonl("prepared/chunked_docs.jsonl")

# Create mapping text->id
id2doc = {d["id"]: d for d in chunks}

# Creating training examples
train_examples = []
for d in chunks:
    if d.get("answer"):  # Check if the document has an answer
        # Create (query, positive) pairs
        train_examples.append(InputExample(texts=[d["text"], d["text"]]))

# Create some random negatives by shuffling
random.shuffle(train_examples)

# Model Initialization
model = SentenceTransformer(BI_MODEL)
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH)
train_loss = losses.MultipleNegativesRankingLoss(model)

# Training loop with checkpoint saving
for epoch in range(EPOCHS):
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=1,  # Train for one epoch at a time
        warmup_steps=100,
        output_path=os.path.join(OUT, "model_epoch_{}".format(epoch))
    )
    

model.save(os.path.join(OUT, "model"))

print("Final model saved to", os.path.join(OUT, "model"))
