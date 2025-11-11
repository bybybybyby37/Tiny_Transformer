#Important imports
import sys
import torch
import torchvision
import tensorboard
import numpy as np
import random
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

import torch.nn as nn
import torch.nn.functional as F
from models import TokenAndPositionEmbedding
from models.transformer import TransformerBlock

from models import tokenizer, embedding, transformer

import os
import requests
import pandas as pd
import time
import json
import math
from types import SimpleNamespace

pd.options.mode.chained_assignment = None  # default='warn'

os.environ['KMP_DUPLICATE_LIB_OK']='True' # To prevent the kernel from dying.


#--------------------------------------------------------------------------------------------#
with open("config\hyperparameters.json", "r") as f:
    cfg = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
data_path = cfg.data_path
save_path = cfg.save_path
split_ratio = tuple(cfg.split_ratio)
block_size = cfg.block_size
batch_size = cfg.batch_size
patience = cfg.patience
max_epochs = cfg.max_epochs 
eval_interval_epochs = cfg.eval_interval  
stride_overlap_ratio = cfg.stride_overlap_ratio

d_model = cfg.d_model
n_heads = cfg.n_heads
n_layers = cfg.n_layers
d_ff = cfg.d_ff
dropout = cfg.dropout

learning_rate = cfg.learning_rate
weight_decay = cfg.weight_decay
grad_clip = cfg.grad_clip
max_iters = cfg.max_iters
eval_interval = cfg.eval_interval
eval_iters = cfg.eval_iters
seed = cfg.seed

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)


#--------------------------------------------------------------------------------------------#
if data_path == "data/tiny_shakespeare.txt":
    if os.path.exists(data_path):
        print(f"'{data_path}' already exists, skipping download.")
    else:
        #Download dataset tiny shakespeare
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        text = requests.get(url).text
        with open("data/tiny_shakespeare.txt", "w", encoding="utf-8") as f:
            f.write(text)
        print("Tiny Shakespeare downloaded! File size:", len(text), "characters")
elif data_path == "data/full_shakespeare.txt":
    if os.path.exists(data_path):
        print(f"'{data_path}' already exists, skipping download.")
    else:
        #Download dataset full shakespeare
        os.makedirs("data", exist_ok=True)
        url = "https://www.gutenberg.org/files/100/100-0.txt"
        print("Downloading full Shakespeare from Project Gutenberg...")
        text = requests.get(url).text

        # Keep only the text
        if "*** START" in text:
            text = text.split("*** START")[1]
        if "*** END" in text:
            text = text.split("*** END")[0]

        with open(data_path, "w", encoding="utf-8") as f:
            f.write(text)
        print("Full Shakespeare downloaded! File size:", len(text), "characters")
else:
    print("Unexpected dataset, stop training.")
    sys.exit()


#--------------------------------------------------------------------------------------------#
#Tokenizer
'''with open("data/tiny_shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()'''

with open("data/full_shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

tok = tokenizer.CharTokenizer(text)
print(len(tok.chars), "unique chars")

ids = tok.encode(text)
data = torch.tensor(ids, dtype=torch.long)

#Calculate vocab_size using the actual character set of the tokenizer
vocab_size = getattr(tok, "vocab_size", len(tok.chars))
print("vocab_size =", vocab_size)
mx = int(max(ids)) if len(ids) > 0 else -1
assert mx < int(vocab_size), f"max id {mx} >= vocab_size {int(vocab_size)}"

# Split by split_ratio (can be adjusted in hyperparameter tuning)
n = len(data)
n_train = int(split_ratio[0] * n)
n_val = int(split_ratio[1] * n)
n_test = n - n_train - n_val

train_data = data[:n_train]
val_data = data[n_train:n_train + n_val]
test_data = data[n_train + n_val:]

print(f"Total tokens: {n:,}")
print(f"Train: {len(train_data):,}, Val: {len(val_data):,}, Test: {len(test_data):,}")

# define a dataset class
class CharDataset(torch.utils.data.Dataset):
    def __init__(self, data, block_size, stride=None):
        self.data = data
        self.block_size = block_size
        
        if stride is None or stride < 1:
            self.stride = self.block_size 
        else:
            self.stride = stride
        
        last_valid_start_idx = len(self.data) - self.block_size - 1
        
        if last_valid_start_idx < 0:
            self.num_samples = 0
        else:
            self.num_samples = (last_valid_start_idx // self.stride) + 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        actual_idx = idx * self.stride
        
        x = self.data[actual_idx : actual_idx + self.block_size]
        y = self.data[actual_idx + 1 : actual_idx + 1 + self.block_size]
        return x, y

my_stride = int (block_size * stride_overlap_ratio)

train_dataset = CharDataset(train_data, block_size, stride=my_stride)
val_dataset = CharDataset(val_data, block_size, stride=my_stride)
test_dataset = CharDataset(test_data, block_size, stride=my_stride)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,  
    pin_memory=True, 
)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)

print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
#--------------------------------------------------------------------------------------------#
#define module
class MiniTransformerLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = TokenAndPositionEmbedding(vocab_size, d_model, block_size)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, block_size)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx, targets=None):
        x = self.embed(idx)                      # (B,T,C)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)                    # (B,T,V)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=200, temperature=1.0, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, k=top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx
    

#--------------------------------------------------------------------------------------------#
#set up a new model
model = MiniTransformerLM().to(device)
print("Params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

#Tensorboard setup
run_dir = f"runs/tt_{time.strftime('%Y%m%d-%H%M%S')}"
writer = SummaryWriter(log_dir=run_dir)
print("TensorBoard logdir:", run_dir)
os.makedirs(os.path.dirname(save_path), exist_ok=True)


#--------------------------------------------------------------------------------------------#
#Start training process
best_val   = float("inf")
bad_epochs = 0
global_step = 0 

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
#scheduler = CosineAnnealingLR(optimizer, T_max=max_iters, eta_min=1e-6)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

@torch.no_grad()
def evaluate(loader):
    model.eval()  
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        _, loss = model(xb, yb)
        total_loss += loss.item()
    model.train() 
    return total_loss / len(loader) 

for epoch in range(1, max_epochs + 1):
    model.train()
    epoch_start_time = time.time()
    
    for batch_idx, (xb, yb) in enumerate(train_loader):
        xb, yb = xb.to(device), yb.to(device)
        
        # 1. Forward pass
        _, loss = model(xb, yb)

        # 2. Backward and optimize
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        # 3. add tensorboard log
        writer.add_scalar("loss/train_batch", loss.item(), global_step)
        global_step += 1

        if batch_idx % 100 == 0: 
             print(f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | Loss {loss.item():.4f}")


    # validation
    print(f"Epoch {epoch} finished in {time.time() - epoch_start_time:.2f}s. Running validation...")

    val_loss = evaluate(val_loader)
    print(f"Epoch {epoch:4d} | Val Loss {val_loss:.4f}")
    
    scheduler.step(val_loss) 

    # add tensorboard log
    writer.add_scalar("loss/val", val_loss, epoch) 
    writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch) 

    # early stopping
    if val_loss < best_val:
        best_val = val_loss
        bad_epochs = 0
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_val": best_val,
        }, save_path)
        print(f"Improved! Best val_loss={best_val:.4f} (saved)")
    else:
        bad_epochs += 1
        print(f"No improvement ({bad_epochs}/{patience})")
        if bad_epochs >= patience:
            print("Early stopping triggered.")
            break


#--------------------------------------------------------------------------------------------#
#Test result for best model
checkpoint = torch.load(save_path, map_location=device)
model.load_state_dict(checkpoint["model"])
model.to(device)
model.eval()

print(f"Loaded best model with best_val={checkpoint['best_val']:.4f} at epoch={checkpoint['epoch']}")

# Test on test dataset 
test_loss = evaluate(test_loader)

print(f"Final Test Loss: {test_loss:.4f}")
print(f"Perplexity (PPL): {torch.exp(torch.tensor(test_loss)):.2f}")


#test sample output text generation
model.eval()
# Test output: Generate from "ROMEO:"
start_ids = tok.encode("ROMEO:")
idx = torch.tensor([start_ids], dtype=torch.long, device=device)
out = model.generate(idx, max_new_tokens=400, temperature=0.9, top_k=50)
print(tok.decode(out[0].tolist()))
