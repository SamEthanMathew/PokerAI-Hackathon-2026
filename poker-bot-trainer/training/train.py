"""
Training script for PokerCloneNet.

Usage:
    python train.py <session.json> [--epochs 50] [--lr 1e-3] [--batch-size 64] [--out poker_clone.pt]

The script:
1. Extracts features from the session JSON
2. Trains PokerCloneNet on betting + discard data jointly
3. Saves the model checkpoint

Loss:
    L = CE(action) + 0.3 * CE(raise_bucket) + 0.2 * CE(discard)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from extract_features import process_session
from model import PokerCloneNet


def parse_args():
    p = argparse.ArgumentParser(description="Train PokerCloneNet")
    p.add_argument("session", help="Path to session JSON file")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--out", default="poker_clone.pt")
    p.add_argument("--val-split", type=float, default=0.1)
    return p.parse_args()


def train():
    args = parse_args()

    print(f"Loading session from {args.session}...")
    b_feats, b_labels, r_buckets, d_feats, d_labels = process_session(args.session)

    print(f"Betting samples: {len(b_labels)}, Discard samples: {len(d_labels)}")

    if len(b_labels) == 0:
        print("ERROR: No betting samples found. Play more hands first.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = PokerCloneNet(hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ce = nn.CrossEntropyLoss()

    # Build datasets
    B = torch.tensor(b_feats, dtype=torch.float32)
    BL = torch.tensor(b_labels, dtype=torch.long)
    BR = torch.tensor(r_buckets, dtype=torch.long)

    # Validation split
    n_val = max(1, int(len(B) * args.val_split))
    n_train = len(B) - n_val
    train_ds, val_ds = random_split(TensorDataset(B, BL, BR), [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # Discard dataset (smaller, trained jointly)
    D_feats = torch.tensor(d_feats, dtype=torch.float32) if len(d_labels) > 0 else None
    D_labels = torch.tensor(d_labels, dtype=torch.long) if len(d_labels) > 0 else None

    best_val_loss = float("inf")
    patience = 10
    patience_counter = 0

    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        correct_action = 0
        total = 0

        for batch_x, batch_action, batch_raise in train_loader:
            batch_x = batch_x.to(device)
            batch_action = batch_action.to(device)
            batch_raise = batch_raise.to(device)

            action_logits, raise_logits, _ = model(batch_x)

            loss = ce(action_logits, batch_action)
            loss += 0.3 * ce(raise_logits, batch_raise)

            # Add discard loss if we have discard data (sample a mini-batch)
            if D_feats is not None and len(D_labels) > 0:
                idx = torch.randint(len(D_labels), (min(args.batch_size, len(D_labels)),))
                d_x = D_feats[idx].to(device)
                d_y = D_labels[idx].to(device)
                _, _, discard_logits = model(d_x)
                loss += 0.2 * ce(discard_logits, d_y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * len(batch_x)
            correct_action += (action_logits.argmax(1) == batch_action).sum().item()
            total += len(batch_x)

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_x, batch_action, batch_raise in val_loader:
                batch_x = batch_x.to(device)
                batch_action = batch_action.to(device)
                batch_raise = batch_raise.to(device)
                action_logits, raise_logits, _ = model(batch_x)
                loss = ce(action_logits, batch_action)
                val_loss += loss.item() * len(batch_x)
                val_correct += (action_logits.argmax(1) == batch_action).sum().item()
                val_total += len(batch_x)

        avg_train = train_loss / total
        avg_val = val_loss / (val_total or 1)
        train_acc = correct_action / total
        val_acc = val_correct / (val_total or 1)

        print(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"Train Loss: {avg_train:.4f} Acc: {train_acc:.3f} | "
            f"Val Loss: {avg_val:.4f} Acc: {val_acc:.3f}"
        )

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), args.out)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"\nBest model saved to {args.out} (val loss: {best_val_loss:.4f})")
    print("\nTo export the bot:")
    print(f"  python export_bot.py {args.out} ../bot/bot.py")


if __name__ == "__main__":
    train()
