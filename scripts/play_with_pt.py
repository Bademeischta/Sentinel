#!/usr/bin/env python
# scripts/play_with_pt.py

import argparse
import time
import torch
import chess

from chess_ai.config        import Config
from chess_ai.game_environment import GameEnvironment
from chess_ai.policy_value_net import PolicyValueNet
from chess_ai.mcts         import MCTS
from chess_ai.action_index import index_to_move

def select_move_from_visits(visits: dict[int,int]) -> chess.Move | None:
    if not visits:
        return None
    best_idx, _ = max(visits.items(), key=lambda kv: kv[1])
    return index_to_move(best_idx)

def main():
    parser = argparse.ArgumentParser(description="Play interactively against a .pt checkpoint")
    parser.add_argument("-c","--ckpt", required=True, help="Pfad zum .pt-Checkpoint")
    parser.add_argument("-s","--simulations", type=int, default=Config.NUM_SIMULATIONS,
                        help="MCTS-Simulationen pro Zug")
    parser.add_argument("--c_puct", type=float, default=Config.C_PUCT,
                        help="PUCT-Parameter für MCTS")
    args = parser.parse_args()

    # 1) Netzwerk laden
    from chess_ai.action_index import ACTION_SIZE
    net = PolicyValueNet(
        GameEnvironment.NUM_CHANNELS,
        ACTION_SIZE,                             # ✔️ korrekt
        num_blocks=Config.NUM_RES_BLOCKS,
        filters=Config.NUM_FILTERS
    ).to(Config.DEVICE)
    ckpt = torch.load(args.ckpt, map_location=Config.DEVICE, weights_only=False)
    state_dict = ckpt.get("model_state", ckpt)
    net.load_state_dict(state_dict)
    net.eval()

    # 2) MCTS-Instanz
    mcts = MCTS(network=net,
                c_puct=args.c_puct,
                num_simulations=args.simulations)

    # 3) Spiel-Setup
    env = GameEnvironment()
    board = chess.Board()
    state = env.reset()
    print(board)

    # 4) Interaktiver Loop
    while True:
        # Spielerzug
        uci = input("Your move (UCI): ")
        try:
            mv = chess.Move.from_uci(uci)
            assert mv in board.legal_moves
        except:
            print("Ungültiger Zug, nochmal.")
            continue

        state, reward, done = env.step(mv)
        board.push(mv)
        print(f"\nNach deinem Zug:\n{board}")
        if done:
            break

        # Engine-Zug
        t0 = time.perf_counter()
        visits = mcts.run(env.board)
        dt = (time.perf_counter()-t0)*1000
        print(f"\n⏱️ Engine denkt: {dt:.1f} ms für {args.simulations} Sims")

        mv2 = select_move_from_visits(visits)
        if mv2 is None:
            print("Engine hat keine legalen Züge mehr. Partie beendet.")
            break

        state, reward, done = env.step(mv2)
        board.push(mv2)
        print(f"\nEngine spielt: {mv2}\n{board}")
        if done:
            break

    print(f"\nErgebnis: {board.result()}")

if __name__ == "__main__":
    main()
