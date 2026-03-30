"""
Evaluate a trained PokerCloneNet model.

Usage:
    python evaluate.py <model.pt> <session.json> [--hidden-dim 256]

Reports:
  - Action type accuracy (fold/check/call/raise)
  - Raise bucket accuracy
  - Discard keep-pair accuracy
  - Per-class breakdown
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from extract_features import process_session
from model import PokerCloneNet, KEEP_COMBOS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_pt", help="Path to trained .pt file")
    parser.add_argument("session", help="Session JSON to evaluate on")
    parser.add_argument("--hidden-dim", type=int, default=256)
    args = parser.parse_args()

    model = PokerCloneNet(hidden_dim=args.hidden_dim)
    model.load_state_dict(torch.load(args.model_pt, map_location="cpu", weights_only=True))
    model.eval()

    b_feats, b_labels, r_buckets, d_feats, d_labels = process_session(args.session)

    if len(b_labels) == 0:
        print("No betting samples found")
        return

    B = torch.tensor(b_feats, dtype=torch.float32)
    BL = torch.tensor(b_labels, dtype=torch.long)
    BR = torch.tensor(r_buckets, dtype=torch.long)

    with torch.no_grad():
        a_logits, r_logits, _ = model(B)

    action_pred = a_logits.argmax(1)
    action_acc = (action_pred == BL).float().mean().item()
    print(f"\nAction accuracy: {action_acc:.3f} ({int(action_acc * len(BL))}/{len(BL)})")

    action_names = ["fold", "raise", "check", "call"]
    print("\nPer-class breakdown:")
    for i, name in enumerate(action_names):
        mask = BL == i
        if mask.sum() == 0:
            continue
        class_acc = (action_pred[mask] == BL[mask]).float().mean().item()
        print(f"  {name:6s}: {class_acc:.3f} ({mask.sum().item()} samples)")

    # Raise bucket accuracy (only when action=raise)
    raise_mask = BL == 1
    if raise_mask.sum() > 0:
        r_pred = r_logits[raise_mask].argmax(1)
        r_acc = (r_pred == BR[raise_mask]).float().mean().item()
        print(f"\nRaise bucket accuracy: {r_acc:.3f} ({raise_mask.sum().item()} samples)")

    # Discard accuracy
    if len(d_labels) > 0:
        D = torch.tensor(d_feats, dtype=torch.float32)
        DL = torch.tensor(d_labels, dtype=torch.long)
        with torch.no_grad():
            _, _, d_logits = model(D)
        d_pred = d_logits.argmax(1)
        d_acc = (d_pred == DL).float().mean().item()
        print(f"\nDiscard accuracy: {d_acc:.3f} ({len(d_labels)} samples)")
        print("Keep combo prediction breakdown:")
        for i, (k1, k2) in enumerate(KEEP_COMBOS):
            mask = DL == i
            if mask.sum() == 0:
                continue
            cacc = (d_pred[mask] == DL[mask]).float().mean().item()
            print(f"  combo ({k1},{k2}): {cacc:.3f} ({mask.sum().item()} samples)")
    else:
        print("\nNo discard samples found")


if __name__ == "__main__":
    main()
