#include "blokusduo_gs.h"

#include <cassert>
#include <unordered_map>
#include <vector>
#include "blokusduo.h"

namespace alphazero::blokusduo_gs {

namespace {
using blokusduo::Move;

std::vector<Move> all_possible_moves;
std::unordered_map<Move, int, Move::Hash> move_to_index;

void compute_possible_moves() {
  all_possible_moves = Board::all_possible_moves();
  for (size_t i = 0; i < all_possible_moves.size(); ++i) {
    move_to_index[all_possible_moves[i]] = i;
  }
}

}  // namespace

// static
[[nodiscard]] uint32_t BlokusDuoGS::num_possible_moves() noexcept {
  if (all_possible_moves.empty()) compute_possible_moves();
  return all_possible_moves.size();
}

[[nodiscard]] std::unique_ptr<GameState> BlokusDuoGS::copy() const noexcept {
  return std::make_unique<BlokusDuoGS>(*this);
}

[[nodiscard]] bool BlokusDuoGS::operator==(
    const GameState& other) const noexcept {
  const auto* other_gs = dynamic_cast<const BlokusDuoGS*>(&other);
  if (other_gs == nullptr) {
    return false;
  }
  return board_.key() == other_gs->board_.key();
}

void BlokusDuoGS::hash(absl::HashState h) const {
  absl::HashState::combine(std::move(h), board_.key().string_view());
}

[[nodiscard]] uint32_t BlokusDuoGS::num_moves() const noexcept {
  return num_possible_moves();
}

[[nodiscard]] Vector<uint8_t> BlokusDuoGS::valid_moves() const noexcept {
  auto valids = Vector<uint8_t>{num_moves()};
  valids.setZero();
  for (auto m : board_.valid_moves()) {
    valids(move_to_index[m]) = 1;
  }
  return valids;
}

void BlokusDuoGS::play_move(uint32_t move) {
  Move m = all_possible_moves[move];
  assert(board_.is_valid_move(m));
  board_.play_move(m);
}

[[nodiscard]] std::optional<Vector<float>> BlokusDuoGS::scores() const noexcept {
  if (!board_.is_game_over()) return std::nullopt;
  auto scores = SizedVector<float, 3>{};
  scores.setZero();
  int v = board_.score(0) - board_.score(1);
  if (v > 0) {
    scores(0) = 1;
  } else if (v < 0) {
    scores(1) = 1;
  } else {
    scores(2) = 1;
  }
  return scores;
}

[[nodiscard]] Tensor<float, 3> BlokusDuoGS::canonicalized() const noexcept {
  auto out = CanonicalTensor{};
  for (int p = 0; p < 2; p++) {
    int ch = board_.is_violet_turn() ? p : 1 - p;
    const uint8_t block =
        p == 0 ? Board::VIOLET_TILE : Board::ORANGE_TILE;
    for (int y = 0; y < Board::YSIZE; y++) {
      for (int x = 0; x < Board::XSIZE; x++) {
        out(ch, y, x) = board_.at(x, y) & block ? 1.0f : 0.0f;
      }
    }
  }
  return out;
}

[[nodiscard]] std::vector<PlayHistory> BlokusDuoGS::symmetries(
    const PlayHistory& base) const noexcept {
  std::vector<PlayHistory> syms(8);
  syms[0] = base;

  const Eigen::array<int, 3> shuffle_dims = {0, 2, 1};
  const Eigen::array<bool, 3> reverse_dims = {false, false, true};
  syms[3].canonical = syms[0].canonical.shuffle(shuffle_dims);
  syms[2].canonical = syms[3].canonical.reverse(reverse_dims);
  syms[5].canonical = syms[2].canonical.shuffle(shuffle_dims);
  syms[4].canonical = syms[5].canonical.reverse(reverse_dims);
  syms[7].canonical = syms[4].canonical.shuffle(shuffle_dims);
  syms[6].canonical = syms[7].canonical.reverse(reverse_dims);
  syms[1].canonical = syms[6].canonical.shuffle(shuffle_dims);

  int n = num_moves();
  assert(n == base.pi.size());
  for (int r = 1; r < 8; r++) {
    syms[r].v = base.v;
    syms[r].pi = Vector<float>(n);
    for (auto i = 0; i < n; ++i) {
      Move m = Board::rotate_move(all_possible_moves[i], r);
      syms[r].pi(move_to_index[m]) = base.pi(i);
    }
  }

  return syms;
}

[[nodiscard]] std::string BlokusDuoGS::dump() const noexcept {
  return "Turn " + std::to_string(board_.turn()) + ":\n" + board_.to_string();
}

}  // namespace alphazero::blokusduo_gs
