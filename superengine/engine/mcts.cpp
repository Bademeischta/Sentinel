#include "mcts.h"
#include "movegen.h"
#include "position.h"

MCTS::MCTS(const std::string &model_path, int num_simulations, int threads)
#ifdef USE_ONNXRUNTIME
    : env_(ORT_LOGGING_LEVEL_WARNING, "mcts"),
      session_(env_, model_path.c_str(), Ort::SessionOptions{})
#endif
{
    num_simulations_ = num_simulations;
    threads_ = threads;
}

std::vector<int> MCTS::run(const Position &pos) {
    std::vector<int> visits(4096, 0);
    // NOTE: This is a placeholder implementation. A real version
    // would perform Monte Carlo tree search using the engine's
    // bitboard move generation and evaluate leaf nodes with the
    // ONNX Runtime session.
    auto moves = movegen::generate_moves(pos);
    for (const auto &m : moves) {
        int idx = (m.from << 6) | m.to;
        visits[idx] = 1; // indicate legal move
    }
    return visits;
}