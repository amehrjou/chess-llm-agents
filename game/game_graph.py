# Copyright (c) 2025 Arash Mehrjou
# All rights reserved.

from typing import TypedDict, Annotated, Sequence
from langgraph.graph import Graph, StateGraph
from langgraph.prebuilt import ToolNode
from loguru import logger
from game.chess_controller import ChessController
import chess
from agents.base_agent import BaseAgent as ChessAgent

class GameState(TypedDict):
    """State of the chess game."""
    controller: ChessController
    current_agent: ChessAgent
    game_over: bool
    move_count: int
    result: str | None
    winner: str | None
    reason: str | None

def create_game_graph(controller: ChessController) -> Graph:
    """Create the LangGraph for managing the chess game.
    
    Args:
        controller: The chess game controller
        
    Returns:
        Graph: The configured LangGraph
    """
    def get_agent_move(state: GameState) -> GameState:
        """Get the next move from the current agent."""
        # Extract current state
        controller = state["controller"]
        current_agent = state["current_agent"]
        move_count = state.get("move_count", 0)
        
        # Check if we've reached the maximum number of moves
        if move_count >= controller.max_moves:
            logger.warning(f"Game terminated after reaching maximum moves: {controller.max_moves}")
            return {
                **state,
                "game_over": True,
                "result": "draw",
                "reason": "maximum_moves_reached",
                "winner": None
            }
        
        # Try to make a move
        success = controller.make_move(current_agent)
        
        if not success:
            logger.error(f"Game terminated due to invalid move by {current_agent.name}")
            return {
                **state,
                "game_over": True,
                "result": "forfeit",
                "winner": "black" if current_agent.color == "white" else "white",
                "reason": "invalid_move"
            }
        
        # Check game ending conditions
        board = controller.board
        
        if board.is_checkmate():
            winner = "black" if board.turn == chess.WHITE else "white"
            logger.info(f"Checkmate! {winner.capitalize()} wins!")
            return {
                **state,
                "game_over": True,
                "result": "checkmate",
                "winner": winner,
                "reason": "checkmate"
            }
        
        if board.is_stalemate() or board.is_insufficient_material() or board.is_fifty_moves():
            reason = ("stalemate" if board.is_stalemate() else 
                     "insufficient_material" if board.is_insufficient_material() else 
                     "fifty_moves")
            logger.info(f"Game drawn due to {reason}!")
            return {
                **state,
                "game_over": True,
                "result": "draw",
                "winner": None,
                "reason": reason
            }
        
        # Update the current agent to the next player
        next_agent = controller.white_agent if current_agent == controller.black_agent else controller.black_agent
        
        # Game continues
        return {
            **state,
            "game_over": False,
            "move_count": move_count + 1,
            "current_agent": next_agent
        }
    
    # Create the graph
    workflow = StateGraph(GameState)
    
    # Add the nodes
    workflow.add_node("agent_move", get_agent_move)
    
    # Define the edges
    workflow.add_edge("agent_move", "agent_move")
    
    # Set the entry point
    workflow.set_entry_point("agent_move")
    
    # Compile the graph
    return workflow.compile()

def run_game(controller: ChessController, moves_per_turn: int = 5) -> None:
    """Run the chess game between the two agents.
    
    Args:
        controller: The chess game controller
        moves_per_turn: Maximum number of move attempts per turn
    """
    # Create the game graph
    graph = create_game_graph(controller)
    
    # Initial state
    initial_state = {
        "controller": controller,
        "current_agent": controller.white_agent,  # White moves first
        "game_over": False,
        "move_count": 0,
        "result": None,
        "winner": None,
        "reason": None
    }
    
    # Run the game
    current_state = initial_state
    
    while not current_state["game_over"]:
        try:
            current_state = graph.invoke(current_state)
            
            # Log the current state
            logger.info(f"Move #{current_state['move_count']}: {current_state['current_agent'].name}'s turn completed")
            logger.info(f"Board state: {controller.board.fen()}")
            
            if current_state["game_over"]:
                if current_state["winner"]:
                    logger.info(f"Game over: {current_state['winner'].capitalize()} wins by {current_state['reason']}")
                else:
                    logger.info(f"Game over: Draw by {current_state['reason']}")
                break
                
        except Exception as e:
            logger.error(f"Unexpected error during game: {str(e)}")
            current_state = {
                **current_state,
                "game_over": True,
                "result": "error",
                "reason": str(e)
            }
            break 