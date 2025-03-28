"""
Base agent class that all specialized SMOL agents will inherit from
"""

import logging
import time
from typing import Dict, List, Any, Optional, Union
import json
import uuid

from transformers import pipeline

class BaseAgent:
    """Base agent class with core functionalities for SMOL agents"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize the base agent
        
        Args:
            name: Unique name for the agent
            config: Configuration dictionary for the agent
        """
        self.name = name
        self.id = str(uuid.uuid4())
        self.config = config
        self.memory = []
        self.max_memory_size = config.get("memory_size", 10)
        self.verbose = config.get("verbose", True)
        self.max_iterations = config.get("max_iterations", 5)
        self.thinking_style = config.get("thinking_style", "step_by_step")
        
        # Set up logging
        self.logger = logging.getLogger(f"agent.{name}")
        self.logger.setLevel(logging.INFO if self.verbose else logging.WARNING)
        
        # Load small language model for agent's reasoning
        self.model_name = config.get("model_name", "distilbert-base-uncased")
        self.model = None  # Lazy loading when needed
        
        self.logger.info(f"Agent {name} initialized with ID {self.id}")
    
    def _load_model(self):
        """Load the language model if not already loaded"""
        if self.model is None:
            self.logger.info(f"Loading model {self.model_name}")
            try:
                self.model = pipeline("text-generation", model=self.model_name)
                self.logger.info(f"Model {self.model_name} loaded successfully")
            except Exception as e:
                self.logger.error(f"Error loading model: {str(e)}")
                raise
    
    def think(self, input_text: str) -> str:
        """
        Generate thinking steps based on input
        
        Args:
            input_text: The text to think about
            
        Returns:
            Generated thinking process as text
        """
        self._load_model()
        
        if self.thinking_style == "step_by_step":
            prompt = f"Think through this problem step by step: {input_text}"
        else:
            prompt = f"Think about: {input_text}"
            
        try:
            result = self.model(prompt, max_length=200, num_return_sequences=1)
            thinking = result[0]['generated_text'].replace(prompt, "").strip()
            
            if self.verbose:
                self.logger.info(f"Thinking process: {thinking}")
                
            return thinking
        except Exception as e:
            self.logger.error(f"Error in thinking process: {str(e)}")
            return f"I need to address: {input_text}"
    
    def act(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform actions based on a task
        
        Args:
            task: Dictionary containing task details
            
        Returns:
            Dictionary with results of the action
        """
        self.logger.info(f"Acting on task: {task.get('task_type', 'unknown')}")
        
        # Record the task in memory
        self._update_memory({"type": "task", "content": task})
        
        # This should be implemented by subclasses
        result = {"status": "not_implemented", "agent": self.name}
        
        # Record the result in memory
        self._update_memory({"type": "result", "content": result})
        
        return result
    
    def _update_memory(self, entry: Dict[str, Any]):
        """Update the agent's memory with a new entry"""
        entry["timestamp"] = time.time()
        self.memory.append(entry)
        
        # Trim memory if it exceeds the maximum size
        if len(self.memory) > self.max_memory_size:
            self.memory = self.memory[-self.max_memory_size:]
    
    def get_memory(self) -> List[Dict[str, Any]]:
        """Get the agent's memory"""
        return self.memory
    
    def communicate(self, message: Dict[str, Any], recipient: str = "broadcast") -> bool:
        """
        Send a message to another agent or broadcast to all agents
        
        Args:
            message: Message content
            recipient: Target agent ID or "broadcast"
            
        Returns:
            Boolean indicating if the message was sent successfully
        """
        message["sender"] = self.id
        message["sender_name"] = self.name
        message["timestamp"] = time.time()
        message["recipient"] = recipient
        
        # In a real implementation, this would use a message broker
        # For now, we just log the message
        self.logger.info(f"Sending message to {recipient}: {json.dumps(message, default=str)[:100]}...")
        
        return True
    
    def serialize(self) -> Dict[str, Any]:
        """Convert agent state to serializable dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.__class__.__name__,
            "config": self.config,
            "memory_size": len(self.memory)
        }
    
    def __str__(self) -> str:
        """String representation of the agent"""
        return f"{self.__class__.__name__}(name={self.name}, id={self.id})"