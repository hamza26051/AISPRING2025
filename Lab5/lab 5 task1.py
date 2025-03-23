import chess
import chess.engine

def evaluate_board(board):
    """
    Evaluates the board position using a simple material evaluation.
    Higher values favor White, lower values favor Black.
    """
    piece_values = {
        chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
        chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0  # King is invaluable
    }
    
    score = 0
    for piece_type, value in piece_values.items():
        score += len(board.pieces(piece_type, chess.WHITE)) * value
        score -= len(board.pieces(piece_type, chess.BLACK)) * value
    
    return score

def beam_search(board, beam_width=3, depth_limit=3):
    """
    Perform beam search to find the best sequence of moves.
    
    Args:
        board (chess.Board): Current chess board state.
        beam_width (int): Number of top moves to explore at each level.
        depth_limit (int): Maximum depth to search.

    Returns:
        best_sequence (list): Best move sequence.
        best_score (float): Evaluation score of the best move sequence.
    """
    # Priority queue for beam search
    beam = [(board, [], evaluate_board(board))]  # (board_state, move_sequence, score)

    for depth in range(depth_limit):
        candidates = []
        
        for current_board, move_seq, score in beam:
            moves = list(current_board.legal_moves)
            
            # Evaluate all moves and store best candidates
            move_evaluations = []
            for move in moves:
                new_board = current_board.copy()
                new_board.push(move)
                new_score = evaluate_board(new_board)
                move_evaluations.append((new_board, move_seq + [move], new_score))
            
            # Select top-k moves based on evaluation
            move_evaluations.sort(key=lambda x: x[2], reverse=True)  # Maximize score
            candidates.extend(move_evaluations[:beam_width])
        
        # Update the beam with the best candidates
        beam = candidates
    
    # Return the best sequence and score
    best_sequence = max(beam, key=lambda x: x[2])
    return best_sequence[1], best_sequence[2]

# Example usage
board = chess.Board()  # Start from the initial position
best_moves, best_score = beam_search(board, beam_width=3, depth_limit=3)

print("Best Move Sequence:", best_moves)
print("Evaluation Score:", best_score)
