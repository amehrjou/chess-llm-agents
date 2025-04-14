# Copyright (c) 2025 Arash Mehrjou
# All rights reserved.

import os
from typing import Optional
import chess
from llama_cpp import Llama
from loguru import logger
from .base_agent import BaseAgent

class LLMChessAgent(BaseAgent):
    """Chess agent that uses a local LLM to make moves."""
    
    def __init__(self, name: str, color: chess.Color, model_path: str = "~/LLMModels/mistral-7b-instruct-v0.2.Q4_K_M.gguf"):
        super().__init__(name)
        self.color = color
        self.model_path = os.path.expanduser(model_path)  # Expand ~ to home directory
        self.llm = None
        self._initialize_llm()
        
        # System prompt for chess
        self.system_prompt = """You are a chess player. You will be given:
1. The current board state in FEN notation
2. A list of all legal moves in UCI notation
3. The move history (if any)

Your task is to choose ONE move from the list of legal moves.
Respond ONLY with the chosen move in UCI notation.

UCI notation uses the format 'source square + target square', e.g.:
- e2e4 (pawn moves from e2 to e4)
- g1f3 (knight moves from g1 to f3)
- e1g1 (castling kingside)
- a7a8q (pawn promotion to queen)
- h2h1n (pawn promotion to knight)

IMPORTANT:
- Only respond with a move from the provided legal moves list
- Do not add any explanation or commentary
- Do not use algebraic notation (like 1.e4 or Nf3)
- Do not use descriptive notation (like P-K4)
- Do not add move numbers or dots
- Choose a move that develops your pieces and controls the center"""

        # Initialize game memory
        self.move_history = []
        self.failed_moves = set()  # Keep track of moves that failed
        
    def _initialize_llm(self):
        """Initialize the LLM with the specified model."""
        try:
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=2048,
                n_threads=4,
                n_gpu_layers=0
            )
            logger.info(f"Initialized LLM with model: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            raise

    def get_move(self, board_state: str, max_retries: int = 3) -> str:
        """Get the next move from the agent.
        
        Args:
            board_state: The current board state in FEN notation
            max_retries: Maximum number of attempts to get a valid move
            
        Returns:
            str: The move in UCI notation
            
        Raises:
            ValueError: If no valid move is generated after max_retries attempts
        """
        # Create a temporary board to validate moves
        temp_board = chess.Board(board_state)
        legal_moves = [move.uci() for move in temp_board.legal_moves]
        
        for attempt in range(max_retries):
            try:
                # Get move from LLM
                move = self._get_llm_move(board_state)
                
                # Validate move format
                if not self._is_valid_uci_format(move):
                    logger.warning(f"Invalid UCI format: {move}")
                    continue
                    
                # Check if move is legal
                if move not in legal_moves:
                    logger.warning(f"Move not in legal moves: {move}. Legal moves: {legal_moves}")
                    continue
                    
                logger.info(f"Agent {self.name} chose move: {move}")
                return move
                
            except Exception as e:
                logger.error(f"Error getting move (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt == max_retries - 1:
                    raise ValueError(f"Failed to generate valid move after {max_retries} attempts")
                    
        # If we get here, all attempts failed
        raise ValueError(f"Failed to generate valid move after {max_retries} attempts")
        
    def _is_valid_uci_format(self, move: str) -> bool:
        """Check if a move string is in valid UCI format.
        
        Args:
            move: The move string to check
            
        Returns:
            bool: True if the move is in valid UCI format
        """
        # Regular moves (e.g., "e2e4")
        if len(move) == 4:
            return all(c in 'abcdefgh' for c in [move[0], move[2]]) and \
                   all(c in '12345678' for c in [move[1], move[3]])
                   
        # Promotion moves (e.g., "e7e8q")
        if len(move) == 5:
            return all(c in 'abcdefgh' for c in [move[0], move[2]]) and \
                   all(c in '12345678' for c in [move[1], move[3]]) and \
                   move[4] in ['q', 'r', 'b', 'n']
                   
        return False

    def _get_llm_move(self, board_state: str) -> str:
        """Get a move from the LLM based on the current board state.
        
        Args:
            board_state: The current state of the chess board in FEN notation
            
        Returns:
            str: The move in UCI notation
        """
        # Create a chess board from FEN
        board = chess.Board(board_state)
        
        # Get list of legal moves in UCI notation
        legal_moves = [move.uci() for move in board.legal_moves]
        if not legal_moves:
            raise ValueError("No legal moves available")
            
        # Filter out moves that have failed before
        available_moves = [move for move in legal_moves if move not in self.failed_moves]
        if not available_moves:
            # If all moves have been tried, reset failed moves and try again
            self.failed_moves.clear()
            available_moves = legal_moves
            
        # Create the prompt with board state and legal moves
        prompt = f"{self.system_prompt}\n\nCurrent board state: {board_state}\n"
        prompt += f"\nLegal moves: {', '.join(available_moves)}\n"
        
        # Add move history if available
        if self.move_history:
            prompt += f"\nPrevious moves: {', '.join(self.move_history[-5:])}\n"  # Only show last 5 moves
            
        prompt += "\nYour move (choose from legal moves):"
        
        response = self.llm(
            prompt,
            max_tokens=10,
            stop=["\n"],
            temperature=0.7,  # Increase temperature for more move variety
        )
        
        # Extract just the move part
        move = response["choices"][0]["text"].strip()
        # Remove any extra text after the move
        move = move.split()[0]  # Take only the first word
        logger.info(f"Agent {self.name} chose move: {move}")
        
        # Validate the move is in the legal moves list
        if move in legal_moves:
            self.move_history.append(move)
            return move
            
        logger.warning(f"Move {move} not in legal moves list")
        self.failed_moves.add(move)  # Add to failed moves
        
        # If we get here, make a random legal move
        move = available_moves[0]  # Take the first available move
        logger.warning(f"Using fallback move: {move}")
        self.move_history.append(move)
        return move 