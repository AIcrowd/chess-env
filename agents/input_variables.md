#### Board Representation

- `{{ board_utf }}` - Visual board representation with Unicode chess pieces (♔♕♖♗♘♙ for White, ♚♛♜♝♞♟ for Black)
- `{{ board_ascii }}` - ASCII board representation
- `{{ FEN }}` - The current board position in Forsyth-Edwards Notation

#### Game State

- `{{ side_to_move }}` - Which side is to move ("White" or "Black")
- `{{ last_move }}` - Description of the last move played (e.g., "White played e4")

#### Legal Moves (String Format)

- `{{ legal_moves_uci }}` - Legal moves as space-separated string in UCI notation (e.g., "e2e4 g1f3 d2d4")
- `{{ legal_moves_san }}` - Legal moves as space-separated string in SAN notation (e.g., "e4 Nf3 d4")

#### Legal Moves (List Format)

- `{{ legal_moves_uci_list }}` - Legal moves as a Python list in UCI notation (e.g., ["e2e4", "g1f3", "d2d4"])
- `{{ legal_moves_san_list }}` - Legal moves as a Python list in SAN notation (e.g., ["e4", "Nf3", "d4"])
- `{{ first_legal_move }}` - The first legal move in UCI notation (useful for showing format examples)

#### Move History (String Format)

- `{{ move_history_uci }}` - Game history as space-separated string in UCI notation (e.g., "e2e4 e7e5 g1f3")
- `{{ move_history_san }}` - Game history as space-separated string in SAN notation (e.g., "e4 e5 Nf3")

#### Move History (List Format)

- `{{ move_history_uci_list }}` - Game history as a Python list in UCI notation (e.g., ["e2e4", "e7e5", "g1f3"])
- `{{ move_history_san_list }}` - Game history as a Python list in SAN notation (e.g., ["e4", "e5", "Nf3"])
