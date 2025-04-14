# Copyright (c) 2025 Arash Mehrjou
# All rights reserved.

import os
from dotenv import load_dotenv
from loguru import logger
from agents.llm_chess_agent import LLMChessAgent
from game.chess_controller import ChessController
from game.game_graph import run_game
import chess

def main():
    # Initialize the chess game
    logger.info("Starting chess game between White and Black")
    
    # Load environment variables
    load_dotenv()
    
    # Initialize agents with centralized model paths
    white_agent = LLMChessAgent(
        name="White",
        color=chess.WHITE,
        model_path="~/LLMModels/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    )
    
    black_agent = LLMChessAgent(
        name="Black",
        color=chess.BLACK,
        model_path="~/LLMModels/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    )
    
    # Create and run the game
    game = ChessController(white_agent, black_agent)
    game.run()

if __name__ == "__main__":
    main() 