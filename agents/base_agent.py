# Copyright (c) 2025 Arash Mehrjou
# All rights reserved.

from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseAgent(ABC):
    """Base class for all agents in the chess game."""
    
    def __init__(self, name: str):
        self.name = name
        self.memory: Dict[str, Any] = {}
    
    @abstractmethod
    def get_move(self, board_state: str) -> str:
        """Get the next move from the agent based on the current board state.
        
        Args:
            board_state: The current state of the chess board in FEN notation
            
        Returns:
            str: The move in algebraic notation (e.g., "e2e4")
        """
        pass
    
    def update_memory(self, key: str, value: Any) -> None:
        """Update the agent's memory with new information.
        
        Args:
            key: The key to store the value under
            value: The value to store
        """
        self.memory[key] = value
    
    def get_memory(self, key: str) -> Any:
        """Retrieve a value from the agent's memory.
        
        Args:
            key: The key to retrieve the value for
            
        Returns:
            Any: The stored value, or None if not found
        """
        return self.memory.get(key) 