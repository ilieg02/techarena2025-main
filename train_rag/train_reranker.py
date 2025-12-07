import os
import json
import random
import time
from tqdm import tqdm
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader

# -------------------------
# CONFIG
# -------------------------
OUT = "reranker"
os.makedirs(OUT, exist_ok=True)

MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EPOCHS = 2
BATCH_SIZE = 8
SUBSAMPLE_RATIO = 0.020        # Use 10% of dataset
NEGATIVE_SAMPLES = 5
BENCHMARK_SAMPLES = 2000       # For speed estimate

# -------------------------
# HELPERS
# -------------------------
def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

def is_valid_chunk(d):
    return d.get("answer") is not None

def create_training_samples(d, chunks):
    """Create InputExample training samples (1 positive + negatives)."""
    samples = []

    pool = [x for x in chunks if x["id"] != d["id"]]
    negs = random.sample(pool, k=min(NEGATIVE_SAMPLES, len(pool)))

    # negative samples
    for n in negs:
        samples.append(InputExample(
            texts=[d["text"], n["text"]],
            label=0.0
        ))

    # positive sample
    samples.append(InputExample(
        texts=[d["text"], d["text"]],
        label=1.0
    ))

    return samples

def benchmark_training_speed(ce, train_samples, batch_size):
    """Run a short benchmark (~2000 samples) to estimate training speed."""
    if len(train_samples) < BENCHMARK_SAMPLES:
        print("Not enough samples for benchmarking.")
        return None

    print(f"\nRunning benchmark with {BENCHMARK_SAMPLES} samples...")

    mini = random.sample(train_samples, BENCHMARK_SAMPLES)

    mini_loader = DataLoader(
        mini,
        batch_size=batch_size,
        shuffle=True
    )

    start = time.time()
    ce.fit(
        train_dataloader=mini_loader,
        epochs=1,
        show_progress_bar=True
    )
    elapsed = time.time() - start

    speed = BENCHMARK_SAMPLES / elapsed
    print(f"Benchmark speed: {speed:.1f} samples/sec")
    return speed

# -------------------------
# LOAD DATA
# -------------------------
print("Loading data...")
chunks = load_jsonl("prepared/chunked_docs.jsonl")
print(f"Loaded {len(chunks)} chunks total.")

# Subsample 10%
subset_size = int(len(chunks) * SUBSAMPLE_RATIO)
chunks = random.sample(chunks, subset_size)
print(f"Using {len(chunks)} chunks (10% subsample).")

# Filter valid chunks
valid_chunks = [d for d in chunks if is_valid_chunk(d)]
print(f"Found {len(valid_chunks)} valid chunks containing answers.")

if not valid_chunks:
    print("No valid chunks found. Exiting.")
    exit()

# -------------------------
# CREATE TRAINING SAMPLES
# -------------------------
print("\nCreating training samples...")
train_samples = []
for d in tqdm(valid_chunks, desc="Building samples"):
    train_samples.extend(create_training_samples(d, chunks))

total_samples = len(train_samples)
print(f"Generated {total_samples} training samples.")

random.shuffle(train_samples)

# -------------------------
# INITIALIZE MODEL
# -------------------------
print("\nInitializing CrossEncoder model...")
ce = CrossEncoder(
    MODEL,
    num_labels=1,
    default_activation_function="sigmoid"
)
print("Model initialized.")

# -------------------------
# BENCHMARK SPEED
# -------------------------
speed = benchmark_training_speed(ce, train_samples, BATCH_SIZE)
if speed:
    est_time = total_samples / speed
    print(f"\nEstimated training time per epoch: {est_time/60:.1f} minutes")

# -------------------------
# TRAINING LOOP
# -------------------------
print("\nStarting full training...")
train_loader = DataLoader(
    train_samples,
    batch_size=BATCH_SIZE,
    shuffle=True
)

for epoch in range(EPOCHS):
    print(f"\n===== Epoch {epoch+1}/{EPOCHS} =====")
    ce.fit(
        train_dataloader=train_loader,
        epochs=1,
        show_progress_bar=True,
        output_path=os.path.join(OUT, f"model_epoch_{epoch}"),
        save_best_model=True
    )
    print(f"Epoch {epoch+1} complete.")

# -------------------------
# SAVE FINAL MODEL
# -------------------------
ce.save(os.path.join(OUT, "model"))
print("\nTraining complete. Final model saved to:", os.path.join(OUT, "model"))
