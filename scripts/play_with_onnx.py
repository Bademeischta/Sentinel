#!/usr/bin/env python
# scripts/play_with_onnx.py

import time
import onnxruntime as ort
import numpy as np
import torch
import chess

from chess_ai.config import Config
from chess_ai.game_environment import GameEnvironment
from chess_ai.action_index import index_to_move

# 1) Wir definieren zuerst unseren Wrapper
class OnnxNetWrapper:
    """
    Wrappt eine ONNX InferenceSession so, dass sie wie ein PyTorch-Modul
    mit .to(), .eval() und __call__ funktioniert.
    """
    def __init__(self, session: ort.InferenceSession):
        self.sess = session

    def to(self, device):
        # no-op, damit MCTS.__init__ network.to(...) klappt
        return self

    def eval(self):
        # no-op, falls MCTS eval() aufruft
        return self

    def __call__(self, x: torch.Tensor):
        """
        Erwartet x: torch.Tensor, Shape (N, C, 8, 8), dtype=float32.
        Gibt zurück:
            log_p: torch.Tensor, Shape (N, ACTION_SIZE)
            v:     torch.Tensor, Shape (N,)
        """
        # 1) Tensor → NumPy (CPU)
        x_np = x.detach().cpu().numpy().astype(np.float32)
        # 2) ONNX-Runtime-Inferenz
        policy_np, value_np = self.sess.run(
            ["policy", "value"],
            {"input": x_np}
        )
        # 3) NumPy → torch.Tensor
        device = x.device
        policy_t = torch.from_numpy(policy_np).to(device)
        value_arr = value_np.reshape(policy_np.shape[0])
        value_t = torch.from_numpy(value_arr).to(device)
        # 4) log-policy zurückgeben
        log_p = torch.log(policy_t + 1e-9)
        return log_p, value_t

# 2) Jetzt MCTS importieren, das in __init__ network.to(...) aufruft
from chess_ai.mcts import MCTS

def select_move_from_visits(visits: dict[int, int]) -> chess.Move | None:
    """
    Wählt den Zug mit den meisten Visits.
    """
    if not visits:
        return None
    best_idx, _ = max(visits.items(), key=lambda kv: kv[1])
    return index_to_move(best_idx)

def main():
    # ONNX-Session initialisieren
    providers = []
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    sess = ort.InferenceSession("nets/final_model.onnx", providers=providers)

    # Wrapper instanziieren
    onnx_net = OnnxNetWrapper(sess)

    # Environment & Board
    env = GameEnvironment()
    board = chess.Board()
    state = env.reset()  # erstes Feature-Array
    print(board)

    # MCTS mit ONNX-Netz und z.B. 200 Simulationen
    mcts = MCTS(network=onnx_net, c_puct=Config.C_PUCT, num_simulations=200)

    # Spielschleife
    while True:
        # Menschlicher Zug
        uci = input("Your move (UCI): ")
        try:
            mv = chess.Move.from_uci(uci)
            assert mv in board.legal_moves
        except:
            print("Ungültiger Zug, bitte erneut.")
            continue

        state, reward, done = env.step(mv)
        board.push(mv)
        print(f"\nNach deinem Zug:\n{board}")
        if done:
            break

        # Engine denkt
        t0 = time.perf_counter()
        visits = mcts.run(env.board)
        dt = (time.perf_counter() - t0) * 1000
        print(f"\n⏱️ Engine benötigte {dt:.1f} ms für {mcts.num_simulations} Simulationen")

        # Zug wählen und spielen
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
