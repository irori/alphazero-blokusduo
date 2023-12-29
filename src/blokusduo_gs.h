#pragma once

#include "dll_export.h"
#include "game_state.h"
#include "blokusduo.h"

namespace alphazero::blokusduo_gs {

using Board = blokusduo::mini::Board;

constexpr const int NUM_PLAYERS = 2;
constexpr const int NUM_SYMMETRIES = 8;
constexpr const std::array<int, 3> CANONICAL_SHAPE = {2, Board::YSIZE, Board::XSIZE};

using CanonicalTensor =
    SizedTensor<float, Eigen::Sizes<CANONICAL_SHAPE[0], CANONICAL_SHAPE[1],
                                    CANONICAL_SHAPE[2]>>;

class DLLEXPORT BlokusDuoGS : public GameState {
 public:
  [[nodiscard]] std::unique_ptr<GameState> copy() const noexcept override;
  [[nodiscard]] bool operator==(const GameState& other) const noexcept override;

  void hash(absl::HashState h) const override;

  // Returns the current player. Players must be 0 indexed.
  [[nodiscard]] uint8_t current_player() const noexcept override {
    return board_.player();
  };

  // Returns the current turn.
  [[nodiscard]] uint32_t current_turn() const noexcept override {
    return board_.turn();
  }

  // Returns the number of possible moves.
  [[nodiscard]] uint32_t num_moves() const noexcept override;

  // Returns the number of players.
  [[nodiscard]] uint8_t num_players() const noexcept override { return 2; };

  // Returns a bool for all moves. The value is true if the move is playable
  // from this GameState.
  [[nodiscard]] Vector<uint8_t> valid_moves() const noexcept override;

  // Plays a move, modifying the current GameState.
  void play_move(uint32_t move) override;

  // Returns nullopt if the game isn't over.
  // Returns a one hot encode result of the game.
  // The first num player positions are set to 1 if that player won and 0
  // otherwise. The last position is set to 1 if the game was a draw and 0
  // otherwise.
  [[nodiscard]] std::optional<Vector<float>> scores() const noexcept override;

  // Returns the canonicalized form of the board, ready for feeding to a NN.
  [[nodiscard]] Tensor<float, 3> canonicalized() const noexcept override;

  // Returns the number of symmetries the game has.
  [[nodiscard]] uint8_t num_symmetries() const noexcept override {
    return NUM_SYMMETRIES;
  }

  // Returns an list of all symetrical game states (including the base state).
  [[nodiscard]] std::vector<PlayHistory> symmetries(
      const PlayHistory& base) const noexcept override;

  // Returns a string representation of the game state.
  [[nodiscard]] std::string dump() const noexcept override;

  // Deletes all data that is not necessary for storing as a hash key.
  // This avoids wasting tons of space when caching states.
  void minimize_storage() override {}

  [[nodiscard]] static uint32_t num_possible_moves() noexcept;

 private:
  // Board contains a layer for each player.
  // A 0 means no piece, a 1 means a piece for that player.
  Board board_;
};

}  // namespace alphazero::blokusduo_gs