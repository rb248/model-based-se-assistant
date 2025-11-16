# Model-Based Software Engineering Assistant 🤖

A production-ready AI-powered system that transforms UML/PlantUML diagrams into fully refactored, tested, and documented code using multi-agent orchestration with LangGraph.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Overview

This system implements a complete **Model-Based Software Engineering (MBSE)** workflow that:

1. **Parses** UML/PlantUML models into intermediate representation
2. **Analyzes** designs using SOLID principles, design patterns, and RAG-enhanced knowledge base
3. **Generates** refactored, production-ready code with proper abstractions
4. **Creates** comprehensive test suites with analysis-aware test generation
5. **Executes** tests in sandboxed environments
6. **Critiques** and suggests improvements
7. **Saves** all artifacts persistently with full project memory

## 🏗️ Architecture

```
👤 User
    │
    ▼
🤖 Model-Based SE Orchestrator Agent (LangGraph)
    ├── 🗂️ Model Parser Agent → 🧰 uml_to_json_tool
    ├── 🔍 Model Analysis Agent → 📚 design_rule_RAG + 📐 pattern_checker + 🧠 project_memory
    ├── 🏗️ Code Generation Agent → 🧰 code_writer_tool + 🧰 test_generator_tool
    ├── 🧪 Test & Verification Agent → 🧰 code_executor_tool + 🧰 unit_test_runner_tool
    └── 🧑‍🏫 Critic & Refactoring Agent → 📄 improvement_suggestions_tool
    │
    ▼
📦 Final Output: Code + Tests + Analysis + Reports
```

## ✨ Key Features

### 🔍 **Advanced Design Analysis**
- **SOLID Principles**: Detects violations of SRP, OCP, LSP, ISP, DIP
- **Design Patterns**: Identifies missing abstractions and suggests patterns
- **Code Metrics**: LCOM (cohesion), fan-in/fan-out (coupling), cyclomatic complexity
- **RAG-Enhanced**: Retrieves relevant design knowledge from vector database (FAISS/Chroma)

### 🏗️ **Intelligent Code Generation**
- **Refactoring-Aware**: Automatically splits god classes and extracts interfaces
- **Dependency Injection**: Implements proper DI patterns
- **Analysis-Driven**: Uses analysis findings to guide refactoring decisions
- **Multi-File Output**: Generates organized project structure

### 🧪 **Comprehensive Testing**
- **Analysis-Aware Tests**: Generates tests for cohesion, coupling, DI patterns
- **Multiple Test Types**: Unit, integration, dependency injection, cohesion tests
- **Sandbox Execution**: Safe isolated test execution with timeout protection
- **Coverage Analysis**: Tracks test coverage and quality

### 🛡️ **Production Ready**
- **LLM Fallback**: Automatic fallback from Gemini to GPT-4o-mini
- **Retry Logic**: Exponential backoff with configurable retries
- **Error Handling**: Comprehensive error tracking and recovery
- **Persistent Storage**: All artifacts saved with project memory

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- OpenAI API key (for fallback)
- Google Gemini API key (optional, primary LLM)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/model-based-se-assistant.git
cd model-based-se-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your API keys:
# OPENAI_API_KEY=your_key_here
# GOOGLE_API_KEY=your_key_here (optional)
```

### Usage

```python
from backend.graph import get_compiled_graph

# Define your UML model
plantuml_model = """
@startuml
class OrderManager {
  +database: Database
  +emailService: EmailService
  +createOrder(userId, items): Order
  +calculateTotal(order): float
  +sendConfirmationEmail(order)
}
class Database {
  +save(data)
}
class EmailService {
  +send(to, subject, body)
}
OrderManager --> Database
OrderManager --> EmailService
@enduml
"""

# Create workflow graph
graph = get_compiled_graph()

# Execute workflow
result = graph.invoke({
    "project_id": "my-project",
    "model_text": plantuml_model,
    "model_format": "plantuml",
    "description": "Order management system"
})

# Access generated artifacts
print(f"Generated {len(result['generated_code']['files'])} files")
print(f"Found {len(result['analysis_report']['findings'])} issues")
print(f"Created {result['generated_tests']['total_tests']} tests")
```

### Output Structure

```
projects/my-project/
├── analysis_report.json    # Design analysis
├── memory.json              # Project memory
├── interfaces.py            # Extracted interfaces
├── models.py                # Data models
├── repositories/            # Repository pattern
├── services/                # Service layer
└── tests/                   # All test files
    ├── __init__.py
    ├── test_interfaces.py
    ├── test_models.py
    └── test_services.py
```

## 📊 Workflow Steps

1. **Parse** (`node_parse_model`) - Converts PlantUML to JSON IR
2. **Analyze** (`node_analyze_model`) - Detects design issues with RAG
3. **Generate Code** (`node_generate_code`) - Creates refactored code
4. **Generate Tests** (`node_generate_tests`) - Creates analysis-aware tests
5. **Save Artifacts** (`node_save_artifacts`) - Persists files to disk
6. **Execute Tests** (`node_run_tests`) - Runs pytest in sandbox
7. **Critique** (`node_critique`) - Reviews and suggests improvements
8. **Final Report** (`node_final_report`) - Assembles comprehensive report

## 🧪 Running Tests

```bash
# Run all unit tests (excludes integration tests that require LLM)
pytest -m "not integration"

# Run integration tests (requires API keys)
pytest -m integration

# Run with coverage
pytest --cov=backend --cov-report=html

# Run full workflow test
python scripts/test_full_workflow.py
```

## 🔧 Configuration

Edit `backend/config.py` or use environment variables:

```python
# LLM Configuration
LLM_PROVIDER = "gemini"              # Primary LLM: "gemini" or "openai"
LLM_FALLBACK_MODEL = "gpt-4o-mini"   # Fallback model
LLM_MAX_TOKENS = 8192                # Max output tokens
LLM_MAX_RETRIES = 3                  # Retry attempts

# RAG Configuration
RAG_BACKEND = "faiss"                # Vector DB: "faiss" or "chroma"
EMBEDDING_PROVIDER = "google"        # Embeddings: "google" or "ollama"

# Project Settings
PROJECTS_DIR = "./projects"          # Output directory
SANDBOX_TIMEOUT = 60                 # Test timeout (seconds)
```

## 📚 Documentation

- [Architecture Overview](backend/README.md)
- [Agent Specifications](backend/agents.py)
- [Graph Workflow](backend/graph.py)
- [Knowledge Base](data/knowledge_base/)
- [Test Suite](tests/)

## 🎯 Use Cases

- **Model-Driven Development**: Transform UML to code automatically
- **Legacy Code Refactoring**: Analyze and refactor existing designs
- **Design Review**: Automated SOLID/pattern analysis
- **Test Generation**: Create comprehensive test suites from models
- **Documentation**: Generate analysis reports and improvement plans

## 🔍 Example Results

From a God Class with 6 methods and multiple responsibilities:

**Before:**
```python
class OrderManager:
    def createOrder(...)
    def calculateTotal(...)
    def sendEmail(...)
    def saveToDatabase(...)
    def generateInvoice(...)
    def logActivity(...)
```

**After (7 refactored files):**
- `interfaces.py` - IRepository, IEmailService, ILogger
- `models.py` - Order, OrderItem data classes
- `repositories/` - Database abstractions
- `services/` - OrderService, EmailService, LogService
- `tests/` - 12+ comprehensive tests

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) file

## 🙏 Acknowledgments

- Built with [LangChain](https://github.com/langchain-ai/langchain) and [LangGraph](https://github.com/langchain-ai/langgraph)
- Uses [FAISS](https://github.com/facebookresearch/faiss) for vector similarity
- Inspired by SOLID principles and design patterns literature

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**⭐ Star this repo if you find it useful!**
