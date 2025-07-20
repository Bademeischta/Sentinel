#!/usr/bin/env python
# scripts/eval_checkpoints.py

import os
import glob
import argparse
import torch
import numpy as np

from chess_ai.config import Config
from chess_ai.game_environment import GameEnvironment
from chess_ai.policy_value_net import PolicyValueNet
from chess_ai.evaluation import evaluate
from chess_ai.action_index import ACTION_SIZE

def load_network(ckpt_path: str) -> torch.nn.Module:
    net = PolicyValueNet(
        GameEnvironment.NUM_CHANNELS,
        ACTION_SIZE,
        num_blocks=Config.NUM_RES_BLOCKS,
        filters=Config.NUM_FILTERS
    ).to(Config.DEVICE)
    ckpt = torch.load(ckpt_path, map_location=Config.DEVICE, weights_only=False)
    state_dict = ckpt.get("model_state", ckpt)
    net.load_state_dict(state_dict)
    net.eval()
    return net

def main():
    parser = argparse.ArgumentParser(description="Round-robin Eval aller .pt-Checkpoints")
    parser.add_argument(
        "--ckpt-dir", "-d", default="checkpoints",
        help="Verzeichnis mit den .pt-Checkpoint-Dateien"
    )
    parser.add_argument(
        "--games", "-g", type=int, default=20,
        help="Anzahl Spiele pro Paarung"
    )
    parser.add_argument(
        "--sims", "-s", type=int, default=Config.NUM_SIMULATIONS,
        help="MCTS-Simulationen pro Zug"
    )
    parser.add_argument(
        "--max-moves", "-m", type=int, default=getattr(Config, "MAX_MOVES", 60),
        help="Maximale Züge pro Partie"
    )
    args = parser.parse_args()

    # Alle Checkpoint-Pfade sammeln
    paths = sorted(glob.glob(os.path.join(args.ckpt_dir, "*.pt")))
    if not paths:
        print(f"No checkpoints found in {args.ckpt_dir}")
        return

    # Netzwerke laden
    print("Lade Netzwerke …")
    nets = {os.path.basename(p): load_network(p) for p in paths}

    # Scoreboard initialisieren
    scores = {name: 0.0 for name in nets}

    # Round-Robin: alle Paare auswerten
    for i, name_i in enumerate(paths):
        for name_j in paths[i+1:]:
            basename_i = os.path.basename(name_i)
            basename_j = os.path.basename(name_j)
            print(f"\n▶ Eval: {basename_i} vs {basename_j}")
            stats = evaluate(
                nets[basename_i],
                nets[basename_j],
                num_games=args.games,
                num_simulations=args.sims,
                max_moves=args.max_moves
            )
            wins_i = stats["wins"]
            wins_j = stats["losses"]
            draws  = stats["draws"]
            print(f"  {basename_i}: W{wins_i}  {basename_j}: W{wins_j}  D{draws}")

            # Punkte: Win=1, Draw=0.5
            scores[basename_i] += wins_i + draws * 0.5
            scores[basename_j] += wins_j + draws * 0.5

    # Endstand ausgeben
    print("\n=== Leaderboard ===")
    sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    for name, score in sorted_scores:
        print(f"{name:25s}  {score:.1f}")

if __name__ == "__main__":
    main()
