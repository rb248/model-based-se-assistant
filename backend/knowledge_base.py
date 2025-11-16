"""
Knowledge base and RAG setup for the Model-Based Software Engineering Assistant.

Provides design principles, patterns, and best practices for the analysis agent
to use when evaluating models and generating recommendations.
"""

import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional

from backend.config import KNOWLEDGE_BASE_DIR
from backend.llms import get_embeddings

logger = logging.getLogger(__name__)


# Curated knowledge base documents
KNOWLEDGE_DOCUMENTS = {
    "SOLID_PRINCIPLES": [
        {
            "title": "Single Responsibility Principle (SRP)",
            "content": """A class should have only one reason to change. Each class should encapsulate a single responsibility or concern.
            
Benefits:
- Easier to understand and test
- More flexible and maintainable
- Reduces coupling between classes

Example: A UserRepository class should handle database operations, not validation or business logic.""",
            "category": "SOLID",
        },
        {
            "title": "Open/Closed Principle (OCP)",
            "content": """Software entities should be open for extension but closed for modification.
            
Use inheritance, composition, and polymorphism to extend behavior without changing existing code.

Benefits:
- Reduces risk of breaking existing functionality
- Promotes reusability
- Makes code more maintainable

Example: Use abstract base classes and interfaces instead of modifying concrete implementations.""",
            "category": "SOLID",
        },
        {
            "title": "Liskov Substitution Principle (LSP)",
            "content": """Subclasses should be substitutable for their base classes without breaking the application logic.
            
If S is a subtype of T, then objects of type S may be substituted for objects of type T.

Benefits:
- Ensures proper inheritance hierarchies
- Prevents subtle bugs
- Enables true polymorphism

Example: A Square class should not override setWidth() and setHeight() differently than Rectangle.""",
            "category": "SOLID",
        },
        {
            "title": "Interface Segregation Principle (ISP)",
            "content": """Clients should not be forced to depend on interfaces they do not use.
            
Create specific, focused interfaces rather than one large "fat" interface.

Benefits:
- Reduces coupling
- Makes code more reusable
- Improves code clarity

Example: Don't create a Worker interface with unrelated methods like work() and eat(); split them.""",
            "category": "SOLID",
        },
        {
            "title": "Dependency Inversion Principle (DIP)",
            "content": """High-level modules should not depend on low-level modules. Both should depend on abstractions.

Abstractions should not depend on details. Details should depend on abstractions.

Benefits:
- Enables testability through dependency injection
- Reduces tight coupling
- Makes systems more flexible

Example: Depend on DatabaseInterface, not on PostgresDatabase directly.""",
            "category": "SOLID",
        },
    ],
    "DESIGN_PATTERNS": [
        {
            "title": "Factory Pattern",
            "content": """Create objects without specifying their concrete classes.
            
Use when:
- Creating objects of different types based on input
- Want to hide implementation details
- Need centralized object creation logic

Benefits:
- Decouples object creation from usage
- Makes code more flexible and maintainable
- Easier to add new types

Example: ShapeFactory.create('circle') instead of new Circle().""",
            "category": "Creational",
        },
        {
            "title": "Strategy Pattern",
            "content": """Define a family of algorithms, encapsulate each one, and make them interchangeable.
            
Use when:
- Multiple algorithms for a task exist
- Need to switch algorithms at runtime
- Want to avoid conditional statements

Benefits:
- Eliminates switch/if statements
- Makes code more maintainable
- Enables runtime algorithm selection

Example: PaymentStrategy interface with CreditCardStrategy, PayPalStrategy, etc.""",
            "category": "Behavioral",
        },
        {
            "title": "Observer Pattern",
            "content": """Define a one-to-many dependency where when one object changes state, all dependents are notified automatically.
            
Use when:
- One object's state change should trigger updates in multiple objects
- Don't want tight coupling between observer and observable
- Need loose coupling between components

Benefits:
- Loose coupling
- Dynamic subscriber relationships
- Natural for event-driven architectures

Example: Event listeners, MVC model updates, pub/sub systems.""",
            "category": "Behavioral",
        },
        {
            "title": "Singleton Pattern",
            "content": """Ensure a class has only one instance and provide a global point of access.
            
Use carefully for:
- Database connections
- Logger instances
- Configuration managers

Caution: Can hide dependencies and make testing difficult. Prefer dependency injection.""",
            "category": "Creational",
        },
    ],
    "UML_BEST_PRACTICES": [
        {
            "title": "Avoid God Objects",
            "content": """A class that knows too much or does too much is called a "God Object" or "God Class".
            
Red flags:
- Class with >10 methods
- Class with >5 responsibilities
- Class with many dependencies
- High cohesion between unrelated methods

Solution: Break the class into smaller, focused classes following SRP.""",
            "category": "AntiPatterns",
        },
        {
            "title": "Clear Association Semantics",
            "content": """Model associations clearly in UML diagrams.
            
Good practices:
- Use role names to clarify relationships
- Specify multiplicity (1..1, 1..*, 0..*)
- Distinguish aggregation (open diamond) vs composition (filled diamond)
- Add navigability arrows when needed

Bad: Unlabeled lines between classes
Good: "author: User" with multiplicity "1..*" on Post class""",
            "category": "UML",
        },
        {
            "title": "Proper Inheritance Hierarchy",
            "content": """Design inheritance hierarchies that make sense logically.
            
Guidelines:
- Use "is-a" relationship for inheritance
- Ensure LSP is satisfied
- Prefer composition over inheritance when possible
- Keep hierarchies shallow (prefer depth <= 3)

Example: Dog IS-A Animal (good) vs Dog HAS-A Dog (bad)""",
            "category": "UML",
        },
    ],
    "CODE_QUALITY": [
        {
            "title": "DRY - Don't Repeat Yourself",
            "content": """Avoid duplicating code. Extract common logic into reusable methods or classes.
            
Benefits:
- Easier maintenance
- Reduces bugs
- Improves readability

When you need to change logic, update it in one place.""",
            "category": "Principles",
        },
        {
            "title": "KISS - Keep It Simple, Stupid",
            "content": """Prefer simple, straightforward solutions over complex ones.
            
Guidelines:
- Avoid premature optimization
- Use clear variable names
- Break down complex logic into smaller functions
- Add comments for "why", not "what"

A simple solution is easier to understand, test, and maintain.""",
            "category": "Principles",
        },
        {
            "title": "YAGNI - You Aren't Gonna Need It",
            "content": """Don't add features or code you don't currently need.
            
Anti-pattern: Building abstract frameworks for hypothetical future features

Better approach:
- Implement what you need now
- Refactor when you need to extend
- Keep code simple and maintainable for current requirements""",
            "category": "Principles",
        },
    ],
}


class KnowledgeBase:
    """Manages the knowledge base with RAG capabilities."""

    def __init__(self, use_faiss: bool = True, embedding_provider: str = "google"):
        """
        Initialize knowledge base.

        Args:
            use_faiss: Whether to use FAISS for vector storage.
            embedding_provider: Provider for embeddings ("google" or "ollama").
        """
        self.use_faiss = use_faiss
        self.embedding_provider = embedding_provider
        self.documents = []
        self.embeddings = None
        self.vector_store = None
        self.retriever = None
        self.logger = logger

    def setup(self) -> None:
        """Set up the knowledge base with embeddings and vector store."""
        self.logger.info("Setting up knowledge base")

        try:
            # Get embeddings
            self.embeddings = get_embeddings(provider=self.embedding_provider)
            self.logger.info(f"Embeddings initialized with {self.embedding_provider}")

            # Flatten documents
            self._flatten_documents()

            # Try FAISS first
            if self.use_faiss:
                self._setup_faiss()
            else:
                self._setup_chroma()

            self.logger.info(f"Knowledge base ready with {len(self.documents)} documents")

        except Exception as e:
            self.logger.error(f"Error setting up knowledge base: {e}")
            raise

    def _flatten_documents(self) -> None:
        """Flatten the hierarchical document structure."""
        self.documents = []
        for category, docs in KNOWLEDGE_DOCUMENTS.items():
            for doc in docs:
                self.documents.append({
                    "title": doc["title"],
                    "content": doc["content"],
                    "category": doc.get("category", category),
                })

    def _setup_faiss(self) -> None:
        """Set up FAISS vector store."""
        try:
            from langchain_community.vectorstores import FAISS
            from langchain_core.documents import Document

            # Create LangChain documents
            docs = [
                Document(
                    page_content=f"{doc['title']}\n{doc['content']}",
                    metadata={"title": doc["title"], "category": doc["category"]},
                )
                for doc in self.documents
            ]

            # Create FAISS vector store
            self.vector_store = FAISS.from_documents(docs, self.embeddings)
            self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
            self.logger.info("FAISS vector store created successfully")

            # Save to disk for persistence
            index_path = KNOWLEDGE_BASE_DIR / "faiss_index"
            self.vector_store.save_local(str(index_path))
            self.logger.info(f"FAISS index saved to {index_path}")

        except ImportError:
            self.logger.warning("FAISS not available, falling back to Chroma")
            self._setup_chroma()

    def _setup_chroma(self) -> None:
        """Set up Chroma vector store as fallback."""
        try:
            from langchain_community.vectorstores import Chroma
            from langchain_core.documents import Document

            # Create LangChain documents
            docs = [
                Document(
                    page_content=f"{doc['title']}\n{doc['content']}",
                    metadata={"title": doc["title"], "category": doc["category"]},
                )
                for doc in self.documents
            ]

            # Create Chroma vector store
            self.vector_store = Chroma.from_documents(
                docs,
                self.embeddings,
                persist_directory=str(KNOWLEDGE_BASE_DIR / "chroma_db"),
            )
            self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
            self.logger.info("Chroma vector store created successfully")

        except ImportError:
            self.logger.error("Both FAISS and Chroma unavailable")
            raise

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, str]]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Search query.
            top_k: Number of results to return.

        Returns:
            List of relevant documents.
        """
        if not self.retriever:
            self.setup()

        try:
            results = self.retriever.invoke(query)
            return [
                {
                    "title": doc.metadata.get("title", "Unknown"),
                    "content": doc.page_content[:200],
                    "category": doc.metadata.get("category", "Unknown"),
                }
                for doc in results[:top_k]
            ]
        except Exception as e:
            self.logger.error(f"Retrieval error: {e}")
            return []

    def get_simple_retriever(self):
        """Get the retriever object for use in agents."""
        if not self.retriever:
            self.setup()
        return self.retriever


# Global knowledge base instance
_kb_instance: Optional[KnowledgeBase] = None



def get_knowledge_base() -> KnowledgeBase:
    """Get or create the global knowledge base instance."""
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = KnowledgeBase(use_faiss=True)
        _kb_instance.setup()
    return _kb_instance


def retrieve_design_knowledge(query: str, top_k: int = 3) -> List[Dict[str, str]]:
    """
    Convenience function to retrieve design knowledge.

    Args:
        query: Search query.
        top_k: Number of results.

    Returns:
        List of relevant documents.
    """
    kb = get_knowledge_base()
    return kb.retrieve(query, top_k)


# Main block for direct execution
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Index the knowledge base for RAG.")
    parser.add_argument("--provider", type=str, default="google", help="Embedding provider (google or ollama)")
    parser.add_argument("--faiss", action="store_true", help="Use FAISS (default: True)")
    args = parser.parse_args()

    print("[KnowledgeBase] Starting setup...")
    kb = KnowledgeBase(use_faiss=args.faiss, embedding_provider=args.provider)
    kb.setup()
    print(f"[KnowledgeBase] Indexed {len(kb.documents)} documents.")
    print("[KnowledgeBase] Setup complete.")
