#include "engine/uci.cpp"
#include "engine/movegen.h"
#include "engine/position.h"
#include "engine/search.h"
#include "engine/nnue_eval.h"

#include <cstring>
#include <iostream>
#include <random>
#include <sstream>

using namespace movegen;

static std::string move_to_uci(const Move &m) {
    std::string s;
    s += char('a' + m.from % 8);
    s += char('1' + m.from / 8);
    s += char('a' + m.to % 8);
    s += char('1' + m.to / 8);
    if (m.promo) s += " nbrq"[m.promo];
    return s;
}

static void run_selfplay(unsigned seed, const std::string &net, int games) {
    if (!net.empty()) nnue::load_network(net);
    init_attack_tables();
    std::mt19937 rng(seed);
    for (int g = 0; g < games; ++g) {
        Position pos("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        Search search;
        std::ostringstream moves;
        int move_no = 1;
        while (true) {
            auto legal = generate_moves(pos);
            if (legal.empty()) break;
            Move best = legal[0];
            int bestScore = -32000;
            for (const auto &m : legal) {
                Position next = pos;
                next.do_move(m);
                int score = -search.search(next, 2);
                if (score > bestScore || (score == bestScore && rng() % 2)) {
                    bestScore = score;
                    best = m;
                }
            }
            if (pos.side_to_move() == WHITE) moves << move_no << ". ";
            moves << move_to_uci(best) << ' ';
            pos.do_move(best);
            if (pos.side_to_move() == WHITE) ++move_no;
        }

        std::string result;
        auto legal = generate_moves(pos);
        if (legal.empty() && pos.in_check(pos.side_to_move()))
            result = pos.side_to_move() == WHITE ? "0-1" : "1-0";
        else
            result = "1/2-1/2";

        std::cout << "[Result \"" << result << "\"]\n\n";
        std::cout << moves.str() << result << "\n";
    }
}

int main(int argc, char **argv) {
    if (argc > 1 && std::strcmp(argv[1], "selfplay") == 0) {
        unsigned seed = 0;
        std::string net;
        int games = 1;
        for (int i = 2; i < argc; ++i) {
            if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc)
                seed = std::stoul(argv[++i]);
            else if (std::strcmp(argv[i], "--net") == 0 && i + 1 < argc)
                net = argv[++i];
            else if (std::strcmp(argv[i], "--games") == 0 && i + 1 < argc)
                games = std::stoi(argv[++i]);
        }
        run_selfplay(seed, net, games);
        return 0;
    }
    init_attack_tables();
    uci_loop();
    return 0;
}
