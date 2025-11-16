"""
Memory management for the Model-Based Software Engineering Assistant.

Handles both short-term session memory and long-term project memory.
Provides storage and retrieval of project state, analyses, and artifacts.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.config import PROJECTS_DIR, MAX_PROJECT_MEMORY_SIZE

logger = logging.getLogger(__name__)


class ProjectMemory:
    """
    Manages long-term storage of project state, generated artifacts,
    and analysis history.
    """

    def __init__(self, project_id: str):
        """
        Initialize project memory.

        Args:
            project_id: Unique identifier for the project.
        """
        self.project_id = project_id
        self.project_dir = PROJECTS_DIR / project_id
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.memory_file = self.project_dir / "memory.json"

    def load(self) -> Dict[str, Any]:
        """
        Load project memory from disk.

        Returns:
            Dictionary containing project state or empty dict if not found.
        """
        if self.memory_file.exists():
            try:
                with open(self.memory_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load memory for project {self.project_id}: {e}")
                return self._default_memory()
        return self._default_memory()

    def save(self, state: Dict[str, Any]) -> None:
        """
        Save project memory to disk.

        Args:
            state: Dictionary containing project state to save.
        """
        try:
            state["last_updated"] = datetime.utcnow().isoformat()
            with open(self.memory_file, "w") as f:
                json.dump(state, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save memory for project {self.project_id}: {e}")

    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update specific fields in project memory.

        Args:
            updates: Dictionary of fields to update.
        """
        state = self.load()
        state.update(updates)
        self.save(state)

    def add_analysis(self, analysis: Dict[str, Any]) -> None:
        """
        Add an analysis report to the project history.

        Args:
            analysis: Dictionary containing analysis results.
        """
        state = self.load()
        if "analysis_history" not in state:
            state["analysis_history"] = []
        state["analysis_history"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "data": analysis
        })
        # Keep only recent analyses
        if len(state["analysis_history"]) > MAX_PROJECT_MEMORY_SIZE:
            state["analysis_history"] = state["analysis_history"][-MAX_PROJECT_MEMORY_SIZE:]
        self.save(state)

    def get_model_ir(self) -> Optional[Dict[str, Any]]:
        """
        Retrieve the current intermediate representation of the model.

        Returns:
            Dictionary containing the model IR or None if not available.
        """
        state = self.load()
        return state.get("model_ir")

    def set_model_ir(self, ir: Dict[str, Any]) -> None:
        """
        Store the intermediate representation of the model.

        Args:
            ir: Dictionary containing the model IR.
        """
        self.update({"model_ir": ir, "model_ir_updated": datetime.utcnow().isoformat()})

    def _default_memory(self) -> Dict[str, Any]:
        """Create a default memory structure."""
        return {
            "project_id": self.project_id,
            "created_at": datetime.utcnow().isoformat(),
            "last_updated": datetime.utcnow().isoformat(),
            "model_ir": None,
            "generated_files": [],
            "analysis_history": [],
            "metadata": {}
        }


class SessionMemory:
    """
    Manages short-term session memory for a conversation or run.
    Stores messages, intermediate results, and context.
    """

    def __init__(self):
        """Initialize session memory."""
        self.messages: List[Dict[str, Any]] = []
        self.intermediate_results: Dict[str, Any] = {}
        self.context: Dict[str, Any] = {}

    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a message to session memory.

        Args:
            role: Role of the message sender (e.g., "user", "assistant", "system").
            content: Message content.
            metadata: Optional metadata associated with the message.
        """
        message = {
            "timestamp": datetime.utcnow().isoformat(),
            "role": role,
            "content": content,
            "metadata": metadata or {}
        }
        self.messages.append(message)

    def get_messages(self) -> List[Dict[str, Any]]:
        """Retrieve all messages in the session."""
        return self.messages

    def set_intermediate_result(self, key: str, value: Any) -> None:
        """Store an intermediate result."""
        self.intermediate_results[key] = value

    def get_intermediate_result(self, key: str) -> Optional[Any]:
        """Retrieve an intermediate result."""
        return self.intermediate_results.get(key)

    def set_context(self, key: str, value: Any) -> None:
        """Store a context value."""
        self.context[key] = value

    def get_context(self, key: str) -> Optional[Any]:
        """Retrieve a context value."""
        return self.context.get(key)

    def clear(self) -> None:
        """Clear all session memory."""
        self.messages.clear()
        self.intermediate_results.clear()
        self.context.clear()
