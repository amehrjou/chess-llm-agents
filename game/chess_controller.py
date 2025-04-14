# Copyright (c) 2025 Arash Mehrjou
# All rights reserved.

import chess
import chess.pgn
from datetime import datetime
from loguru import logger
from typing import Tuple, Optional
from agents.base_agent import BaseAgent
from utils.visualization import ChessVisualizer
from pathlib import Path

class ChessController:
    """Controller for managing the chess game between two agents."""
    
    def __init__(self, white_agent: BaseAgent, black_agent: BaseAgent):
        self.board = chess.Board()
        self.white_agent = white_agent
        self.black_agent = black_agent
        self.initial_board = self.board.copy()
        self.move_count = 0  # Add move counter
        self.max_moves = 14  # Set maximum moves to 14 (7 moves per player)
        self.move_history: list[Tuple[str, str]] = []  # List of (player, move) tuples
        self.visualizer = ChessVisualizer(Path("game_visualizations"))
        self.game = chess.pgn.Game()
        
        # Set up game metadata
        self.game.headers["Event"] = "AI Chess Game"
        self.game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
        self.game.headers["White"] = white_agent.name
        self.game.headers["Black"] = black_agent.name
        self.game.headers["Result"] = "*"  # Ongoing game
        
        # Initialize game node
        self.current_node = self.game
        
        # Save initial board state
        self.save_board_visualization()
    
    def get_current_agent(self) -> BaseAgent:
        """Get the agent whose turn it is to move.
        
        Returns:
            BaseAgent: The agent that should make the next move
        """
        return self.white_agent if self.board.turn == chess.WHITE else self.black_agent
    
    def is_game_over(self) -> bool:
        """Check if the game is over.
        
        Returns:
            bool: True if the game is over, False otherwise
        """
        return self.board.is_game_over() or self.move_count >= self.max_moves  # Add move limit check
    
    def get_game_result(self) -> Optional[str]:
        """Get the result of the game if it's over.
        
        Returns:
            Optional[str]: The game result or None if the game is not over
        """
        if self.move_count >= self.max_moves:
            result = "Game ended after 7 moves (14 half-moves)"
            self.game.headers["Result"] = "1/2-1/2"  # Draw when max moves reached
        elif self.board.is_checkmate():
            winner = "Black" if self.board.turn == chess.WHITE else "White"
            result = f"Checkmate - {winner} wins"
            self.game.headers["Result"] = "0-1" if winner == "Black" else "1-0"
        elif self.board.is_stalemate():
            result = "Stalemate"
            self.game.headers["Result"] = "1/2-1/2"
        elif self.board.is_insufficient_material():
            result = "Draw - Insufficient material"
            self.game.headers["Result"] = "1/2-1/2"
        elif self.board.is_seventyfive_moves():
            result = "Draw - 75-move rule"
            self.game.headers["Result"] = "1/2-1/2"
        elif self.board.is_fivefold_repetition():
            result = "Draw - Fivefold repetition"
            self.game.headers["Result"] = "1/2-1/2"
        else:
            result = "Draw"
            self.game.headers["Result"] = "1/2-1/2"
            
        # Save the game to PGN file
        self.save_game()
        
        # Create GIF of the game
        try:
            logger.info("Generating game visualization...")
            gif_path = self.visualizer.create_game_gif(list(self.board.move_stack))
            logger.info(f"Game visualization saved to: {gif_path}")
        except Exception as e:
            logger.error(f"Failed to generate game visualization: {str(e)}")
        
        return result
    
    def _is_valid_move(self, move: str) -> bool:
        """
        Check if a move is valid in both format and legality.
        Args:
            move: Move in UCI format (e.g., 'e2e4', 'e7e8q')
        Returns:
            bool: True if move is valid and legal, False otherwise
        """
        try:
            # Check UCI format
            if not isinstance(move, str):
                return False
            if len(move) not in [4, 5]:  # Standard moves are 4 chars, promotions are 5
                return False
                
            # Try to parse the move
            move_obj = chess.Move.from_uci(move)
            
            # Check if move is legal in current position
            return move_obj in self.board.legal_moves
            
        except ValueError:
            return False
        except Exception as e:
            logger.error(f"Error validating move: {str(e)}")
            return False
            
    def make_move(self, agent) -> bool:
        """
        Get a move from the agent and apply it to the board if valid.
        Returns True if the move was successfully made, False otherwise.
        """
        try:
            move = agent.get_move(self.get_board_state())
            if not move:
                logger.error(f"[Move {(self.move_count + 2) // 2}] {agent.name} returned an empty move")
                return False
                
            if not self._is_valid_move(move):
                logger.error(f"[Move {(self.move_count + 2) // 2}] {agent.name} returned invalid move: {move}")
                return False
                
            # Handle promotion moves
            if len(move) == 5:
                promotion_piece = move[4].lower()
                if promotion_piece not in ['n', 'b', 'r', 'q']:
                    logger.error(f"[Move {(self.move_count + 2) // 2}] Invalid promotion piece: {promotion_piece}")
                    return False
                move_obj = chess.Move.from_uci(move)
            else:
                move_obj = chess.Move.from_uci(move)
                
            # Make the move
            self.board.push(move_obj)
            self.move_history.append(move)
            self.move_count += 1  # Increment move count
            
            # Save board state visualization
            self.save_board_visualization()
            
            # Update game tree
            self.update_game_tree(move)
            
            # Calculate actual move number and color
            move_number = (self.move_count + 1) // 2
            color = "White" if self.move_count % 2 == 1 else "Black"
            logger.info(f"╔════ Move {move_number} ════╗")
            logger.info(f"║ {color}: {move}")
            logger.info(f"╚{'═' * (14 + len(str(move_number)))}╝")
            
            return True
            
        except ValueError as e:
            logger.error(f"[Move {(self.move_count + 2) // 2}] Invalid move format from {agent.name}: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"[Move {(self.move_count + 2) // 2}] Error making move for {agent.name}: {str(e)}")
            return False
    
    def get_board_state(self) -> str:
        """Get the current board state in FEN notation.
        
        Returns:
            str: The current board state in FEN notation
        """
        return self.board.fen()
    
    def get_move_history(self) -> list[Tuple[str, str]]:
        """Get the history of moves made in the game.
        
        Returns:
            list[Tuple[str, str]]: List of (player, move) tuples
        """
        return self.move_history
        
    def save_game(self, filename: str = "game.pgn") -> None:
        """Save the game to a PGN file.
        
        Args:
            filename: Name of the file to save the game to
        """
        # Save to game_visualizations directory
        pgn_path = self.visualizer.output_dir / filename
        with open(pgn_path, "w") as f:
            print(self.game, file=f, end="\n\n")
            
    def run(self) -> None:
        """Run the chess game between the two agents."""
        logger.info("╔═══════════════════════╗")
        logger.info("║    Starting Game      ║")
        logger.info("╚═══════════════════════╝")
        
        while not self.is_game_over():
            current_agent = self.get_current_agent()
            move_number = (self.move_count + 2) // 2
            color = "White" if self.board.turn == chess.WHITE else "Black"
            logger.info(f"[Move {move_number}] Waiting for {color}'s move...")
            
            if not self.make_move(current_agent):
                logger.error(f"[Move {move_number}] Failed to make move for {current_agent.name}")
                break
                
        result = self.get_game_result()
        logger.info("╔═══════════════════════╗")
        logger.info(f"║      Game Over       ║")
        logger.info(f"║ Moves played: {(self.move_count + 1) // 2}      ║")
        logger.info(f"║ Result: {result[:15]} {'.' * (15 - len(result[:15]))} ║")
        logger.info("╚═══════════════════════╝")
    
    def save_board_visualization(self) -> None:
        """Save the current board state as an image."""
        self.visualizer.create_board_image(self.board)
        
    def update_game_tree(self, move: str) -> None:
        """Update the game tree with a new move.
        
        Args:
            move: The move in UCI notation
        """
        move_obj = chess.Move.from_uci(move)
        self.current_node = self.current_node.add_variation(move_obj) 