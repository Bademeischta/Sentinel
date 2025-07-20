#pragma once
#include "position.h"
#include <memory>
#include <string>
#include <vector>

#ifdef USE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

class MCTS {
public:
    MCTS(const std::string &model_path, int num_simulations = 100, int threads = 1);
    // Returns visit counts per move index (0..4095)
    std::vector<int> run(const Position &pos);
private:
#ifdef USE_ONNXRUNTIME
    Ort::Env env_;
    Ort::Session session_{nullptr};
#endif
    int num_simulations_;
    int threads_;
};