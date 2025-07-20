#!/usr/bin/env python
# scripts/bench_self_play.py

import time
import argparse

import torch
from chess_ai.game_environment import GameEnvironment
from chess_ai.self_play import run_self_play
from chess_ai.policy_value_net import PolicyValueNet
from chess_ai.config import Config
from chess_ai.action_index import ACTION_SIZE  # ← hier importieren

def benchmark_self_play(games: int, sims: int):
    """
    Spielt `games` Self-Play-Partien mit `sims` MCTS-Simulationen pro Zug
    und misst die dafür benötigte Zeit. Rechnet hoch auf 1000 Spiele.
    """
    # Netzwerk aufbauen
    net = PolicyValueNet(
        GameEnvironment.NUM_CHANNELS,
        ACTION_SIZE,                  # ← korrekt verwenden
        num_blocks=Config.NUM_RES_BLOCKS,
        filters=Config.NUM_FILTERS
    ).to(Config.DEVICE)
    net.eval()

    # Warmup
    _ = next(run_self_play(net, num_simulations=1))

    start = time.perf_counter()
    for i in range(1, games + 1):
        # jede Partie komplett durchselfplayen
        for _ in run_self_play(net, num_simulations=sims):
            pass
        print(f"\rPlayed {i}/{games} games…", end="", flush=True)
    total = time.perf_counter() - start

    avg_per_game = total / games
    eta_1000 = avg_per_game * 1000

    print(f"\n\n-- Benchmark Results --")
    print(f"Total time for {games} games @ {sims} sims: {total:.1f} s")
    print(f"Average per game: {avg_per_game:.2f} s")
    print(f"Extrapolated for 1000 games: {eta_1000/3600:.2f} h ({eta_1000:.0f} s)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark Self-Play in reinem Python-MCTS"
    )
    parser.add_argument(
        "--games", "-g", type=int, default=10,
        help="Anzahl Self-Play-Spiele zum Messen"
    )
    parser.add_argument(
        "--sims", "-s", type=int, default=100,
        help="MCTS-Simulationen pro Zug"
    )
    args = parser.parse_args()
    benchmark_self_play(args.games, args.sims)
