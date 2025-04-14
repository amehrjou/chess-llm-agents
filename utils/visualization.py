# Copyright (c) 2025 Arash Mehrjou
# All rights reserved.

import os
import chess
import chess.svg
from pathlib import Path
from cairosvg import svg2png
from PIL import Image
import tempfile
import matplotlib.pyplot as plt

class ChessVisualizer:
    """Helper class for visualizing chess games."""
    
    def __init__(self, output_dir: Path):
        """Initialize the visualizer.
        
        Args:
            output_dir: Directory to save visualization files
        """
        self.output_dir = output_dir
        self.move_count = 0
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def create_board_image(self, board: chess.Board, move: chess.Move | None = None) -> str:
        """Create an image of the current board state.
        
        Args:
            board: The chess board to visualize
            move: The last move made (optional)
            
        Returns:
            str: Path to the generated PNG file
        """
        # Generate SVG
        size = 400  # Size of the board in pixels
        svg_data = chess.svg.board(
            board=board,
            size=size,
            lastmove=move if move else None,
            coordinates=True
        )
        
        # Save PNG to temp directory
        png_path = self.temp_dir / f"frame_{self.move_count:03d}.png"
        
        # Convert to PNG
        svg2png(
            bytestring=svg_data.encode('utf-8'),
            write_to=str(png_path),
            output_width=size,
            output_height=size
        )
        
        return str(png_path)
        
    def create_game_gif(self, moves: list[chess.Move], duration: int = 1000) -> str:
        """Create a GIF animation of the game.
        
        Args:
            moves: List of moves in the game
            duration: Duration for each frame in milliseconds
            
        Returns:
            str: Path to the generated GIF file
        """
        # Reset move counter
        self.move_count = 0
        
        # Create a new board
        board = chess.Board()
        
        # Create first frame
        frames = [Image.open(self.create_board_image(board))]
        self.move_count += 1
        
        # Create a frame for each move
        for move in moves:
            board.push(move)
            frames.append(Image.open(self.create_board_image(board, move)))
            self.move_count += 1
        
        # Save the GIF
        gif_path = self.output_dir / "game.gif"
        frames[0].save(
            str(gif_path),
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0
        )
        
        # Clean up temp directory
        for file in self.temp_dir.glob("*.png"):
            file.unlink()
        self.temp_dir.rmdir()
        
        return str(gif_path) 