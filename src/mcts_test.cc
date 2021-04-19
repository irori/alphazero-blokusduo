#include "mcts.h"

#include <gtest/gtest.h>

#include "connect4_gs.h"

namespace alphazero {
namespace {

// NOLINTNEXTLINE
TEST(Node, Basic) {
  auto gs = connect4_gs::Connect4GS{};
  auto root = Node{};
  root.add_children(gs.valid_moves());
  EXPECT_EQ(7, root.children.size());

  const float CPUCT = 2.0;
  auto pi = SizedVector<float, 7>{};
  pi << 0.1F, 1.2F, 0.3F, 0.4F, 0.5F, 0.6F, 0.7F;
  root.update_policy(pi);
  for (const auto& c : root.children) {
    if (c.move == 5) {
      EXPECT_EQ(0.6F, c.policy);
      EXPECT_EQ(1.2F, c.uct(1, CPUCT));
      EXPECT_EQ(2.4F, c.uct(2, CPUCT));
    }
  }
  root.n = 1;
  auto* n = root.best_child(CPUCT);
  EXPECT_EQ(1, n->move);
}

// NOLINTNEXTLINE
TEST(MCTS, Basic) {
  auto gs = connect4_gs::Connect4GS{};
  gs.play_move(1);
  gs.play_move(6);
  gs.play_move(3);
  gs.play_move(6);
  auto mcts = MCTS{2, gs.num_moves()};
  while (mcts.depth() < 800) {
    auto leaf = mcts.find_leaf(gs);
    auto [value, pi] = dumb_eval(*leaf);
    mcts.process_result(value, pi);
  }
  EXPECT_EQ(mcts.pick_move(0, gs.num_moves()), 2);
}

}  // namespace
}  // namespace alphazero