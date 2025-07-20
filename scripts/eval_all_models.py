#!/usr/bin/env python
# scripts/eval_all_models.py

import os
import glob
import argparse
import time
import torch
import numpy as np
import chess
import chess.svg
import cairosvg
import onnxruntime as ort  # ← hinzugefügt

from chess_ai.config           import Config
from chess_ai.game_environment import GameEnvironment
from chess_ai.policy_value_net import PolicyValueNet
from chess_ai.action_index     import ACTION_SIZE, index_to_move
from chess_ai.mcts             import MCTS

# ─── Helpers ────────────────────────────────────────────────────────────────────

def load_pt_model(path: str):
    net = PolicyValueNet(
        GameEnvironment.NUM_CHANNELS,
        ACTION_SIZE,
        num_blocks=Config.NUM_RES_BLOCKS,
        filters=Config.NUM_FILTERS
    ).to(Config.DEVICE)
    ckpt = torch.load(path, map_location=Config.DEVICE, weights_only=False)
    state_dict = ckpt.get("model_state", ckpt)
    net.load_state_dict(state_dict)
    net.eval()
    return net

class OnnxNetWrapper:
    """Wrappt eine ONNX InferenceSession als PolicyValueNet-like Objekt."""
    def __init__(self, session: ort.InferenceSession):  # ← Ort ist nun importiert
        self.sess = session

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, x: torch.Tensor):
        x_np = x.detach().cpu().numpy().astype(np.float32)
        policy_np, value_np = self.sess.run(
            ["policy", "value"], {"input": x_np}
        )
        device = x.device
        policy_t = torch.from_numpy(policy_np).to(device)
        v_arr = value_np.reshape(policy_np.shape[0])
        value_t = torch.from_numpy(v_arr).to(device)
        return torch.log(policy_t + 1e-9), value_t

def load_onnx_model(path: str):
    providers = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers.insert(0, "CUDAExecutionProvider")
    sess = ort.InferenceSession(path, providers=providers)
    return OnnxNetWrapper(sess)

def select_move_from_visits(visits: dict[int,int]) -> chess.Move | None:
    if not visits:
        return None
    idx, _ = max(visits.items(), key=lambda kv: kv[1])
    return index_to_move(idx)

def save_board_png(board: chess.Board, out_path: str):
    svg = chess.svg.board(board=board)
    cairosvg.svg2png(bytestring=svg.encode("utf-8"), write_to=out_path)

# ─── Evaluation ────────────────────────────────────────────────────────────────

def play_game(white_net, black_net, sims: int, png_dir: str, basename: str, game_idx: str):
    env = GameEnvironment()
    board = chess.Board()
    state = env.reset()

    mcts_white = MCTS(white_net, c_puct=Config.C_PUCT, num_simulations=sims)
    mcts_black = MCTS(black_net, c_puct=Config.C_PUCT, num_simulations=sims)

    while True:
        net = mcts_white if board.turn == chess.WHITE else mcts_black
        visits = net.run(env.board)
        mv = select_move_from_visits(visits)
        if mv is None:
            break
        state, _, done = env.step(mv)
        board.push(mv)
        if done:
            break

    result = board.result()  # '1-0', '0-1' or '1/2-1/2'
    fname = f"{basename}_game{game_idx}_{result.replace('/','-')}.png"
    save_board_png(board, os.path.join(png_dir, fname))
    return result

def main():
    parser = argparse.ArgumentParser(
        description="Evaluiere alle .pt- und .onnx-Modelle im Round-Robin und speichere Endstellungen als PNG"
    )
    parser.add_argument("--pt-dir",   default="checkpoints", help="Ordner mit .pt Checkpoints")
    parser.add_argument("--onnx-dir", default="nets",        help="Ordner mit .onnx Modellen")
    parser.add_argument("--games", "-g", type=int, default=10, help="Spiele pro Paarung")
    parser.add_argument("--sims",  "-s", type=int, default=50, help="MCTS-Simulationen/Move")
    parser.add_argument("--png-dir",    default="games_png",   help="Ausgabeordner für PNGs")
    args = parser.parse_args()

    os.makedirs(args.png_dir, exist_ok=True)

    # Modelle sammeln
    pt_paths   = sorted(glob.glob(os.path.join(args.pt_dir,  "*.pt")))
    onnx_paths = sorted(glob.glob(os.path.join(args.onnx_dir,"*.onnx")))
    all_models = []

    for p in pt_paths:
        name = os.path.basename(p)
        all_models.append((name, load_pt_model(p)))
    for o in onnx_paths:
        name = os.path.basename(o)
        all_models.append((name, load_onnx_model(o)))

    if len(all_models) < 2:
        print("Braucht mindestens 2 Modelle (.pt oder .onnx).")
        return

    # Round-Robin-Evaluation
    results = {}
    for i in range(len(all_models)):
        name_i, net_i = all_models[i]
        for j in range(i+1, len(all_models)):
            name_j, net_j = all_models[j]
            basename = f"{name_i}_vs_{name_j}"
            print(f"\n▶ Eval: {basename}")
            res_list = []
            for k in range(1, args.games + 1):
                res1 = play_game(net_i, net_j, args.sims,   args.png_dir, basename, str(k))
                res2 = play_game(net_j, net_i, args.sims,   args.png_dir, basename, f"{k}b")
                res_list += [res1, res2]
                print(f"  Spiel {2*k-1}: {res1},   Spiel {2*k}: {res2}")
            results[basename] = res_list

    # Zusammenfassung
    print("\n=== Summary ===")
    for basename, res_list in results.items():
        wins_i = res_list.count("1-0")
        wins_j = res_list.count("0-1")
        draws  = res_list.count("1/2-1/2")
        print(f"{basename}: A_Wins={wins_i}, B_Wins={wins_j}, Draws={draws}")

if __name__ == "__main__":
    main()
