# Model-Based Software Engineering Assistant

A comprehensive multi-agent LLM system for analyzing software models, generating code artifacts, and proposing refactorings.

## Project Structure

```
backend/
├── __init__.py              # Package initialization
├── config.py                # Configuration and environment variables
├── memory.py                # Project and session memory management
├── tools.py                 # Tool definitions for agent use
├── agents.py                # Agent class definitions
└── graph.py                 # LangGraph workflow orchestration

notebooks/
└── demo.ipynb               # Interactive demonstration notebook

tests/                        # Unit and integration tests
logs/                         # Execution logs and traces
data/
└── knowledge_base/          # RAG documents and embeddings

requirements.txt             # pip dependencies
pyproject.toml              # Modern Python project configuration
```

## Backend Modules

### `config.py`
Centralized configuration management for:
- LLM settings (model, temperature, token limits)
- API keys and authentication
- RAG backend selection (FAISS/Chroma)
- Project directories and paths
- Logging configuration

### `memory.py`
Memory management for both short and long-term context:
- **ProjectMemory**: Persistent storage of project state, model IR, generated artifacts, and analysis history
- **SessionMemory**: Transient session context, messages, and intermediate results
- Methods for saving/loading state and tracking project history

### `tools.py`
LangChain-compatible tools that agents can invoke:
- **File I/O**: `read_project_file`, `write_project_file`, `list_project_files`
- **Parsing**: `parse_plantuml_classes` (converts PlantUML to IR)
- **Execution**: `run_pytest`, `run_python_script`
- **Validation**: `validate_python_syntax`, `check_code_quality`
- **RAG**: `retrieve_design_knowledge` (retrieves patterns and principles)

### `agents.py`
Specialized agent implementations:
- **ParserAgent**: Converts model descriptions to intermediate representation
- **AnalysisAgent**: Analyzes models for design issues using RAG
- **CodeGenerationAgent**: Generates source code from model IR
- **TestGenerationAgent**: Creates pytest test cases
- **CriticAgent**: Reviews artifacts and proposes improvements
- **OrchestratorAgent**: Coordinates the workflow

### `graph.py`
LangGraph state machine definition:
- **WorkflowState**: Dataclass representing workflow state
- **Nodes**: `parse`, `analyze`, `codegen`, `testgen`, `execute`, `critique`, `final_report`
- **Conditional edges**: Routing logic based on errors and analysis results
- **Graph compilation**: Full workflow orchestration

## Key Features

### Multi-Agent Orchestration
- Dedicated agents for specific tasks (parsing, analysis, code generation, testing)
- Clear separation of concerns and context engineering
- Conditional routing based on workflow results

### Model Analysis
- Parse PlantUML class diagrams and UML descriptions
- Check consistency rules and design heuristics
- Detect issues: missing methods, god objects, bad dependencies

### Code Generation
- Generate Python class skeletons from model IR
- Create method signatures and basic implementations
- Maintain consistency with model specifications

### Test Generation
- Auto-generate pytest test cases
- Create test patterns based on model design
- Validate code through test execution

### RAG Integration
- Curated knowledge base of design principles (SOLID, patterns)
- Semantic retrieval of relevant guidelines
- Enriched analysis with references to best practices

### Memory Management
- Long-term project memory with history tracking
- Session memory for conversational context
- Automatic state persistence

### Observability
- Structured JSON logging of all operations
- Execution traces with timestamps and details
- Metrics collection (classes, tests, issues, performance)
- Optional LangSmith integration for hosted tracing

## Getting Started

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or with all optional features
pip install -e ".[all]"
```

### Running the Demo

```bash
cd notebooks
jupyter notebook demo.ipynb
```

The demo notebook walks through:
1. Setting up agents and tools
2. Defining a sample PlantUML model
3. Running the multi-agent workflow
4. Inspecting results and metrics
5. Reviewing execution traces

### Environment Variables

```bash
# LLM Configuration
export LLM_MODEL="gpt-4o-mini"           # OpenAI model
export LLM_TEMPERATURE="0.7"
export OPENAI_API_KEY="sk-..."

# RAG Configuration
export RAG_BACKEND="faiss"                # or "chroma"
export TOP_K_RETRIEVAL="3"

# Observability
export ENABLE_LANGSMITH="false"
export LOG_LEVEL="INFO"
```

## Architecture Overview

### Storing API Keys Securely

We recommend storing API keys in a local `.env` file in the project root (it's already ignored by `.gitignore`). You can use the included `.env.example` as a template:

1. Copy the example into a local file:

```bash
cp .env.example .env
```

2. Open `.env` and fill in the keys (e.g. `GOOGLE_API_KEY`, `OPENAI_API_KEY`).

3. The project automatically loads `.env` during startup (using `python-dotenv` from `backend/config.py`) so you don't need to export the environment variables manually.

4. If you want to export an API key for a single session (e.g., CI), run:

```bash
export GOOGLE_API_KEY="your-key-here"
export OPENAI_API_KEY="your-openai-key"
```

5. To verify the keys are available in the environment run (from the project root):

```bash
python -c "from backend.config import GOOGLE_API_KEY, OPENAI_API_KEY; print('Google:', bool(GOOGLE_API_KEY)); print('OpenAI:', bool(OPENAI_API_KEY))"
You can also run the small helper script to validate keys and LLM factory initialization:

```bash
python scripts/check_keys.py
```

Note: Never commit your `.env` file — keep it local and private. The included `.env.example` is safe to commit and serves as a template.

```
┌─────────────────────────────────────────────────┐
│         User Interface (Jupyter/FastAPI)        │
└──────────────┬──────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────┐
│      LangGraph Orchestration Layer              │
│  (State Machine with Multi-Agent Routing)       │
└──────────────┬──────────────────────────────────┘
               │
     ┌─────────┼─────────┬──────────────┬─────────┐
     │         │         │              │         │
┌────▼──┐ ┌───▼────┐ ┌──▼─────┐ ┌────▼───┐ ┌──▼──┐
│Parser │ │Analysis│ │CodeGen │ │TestGen │ │Test │
│Agent  │ │Agent   │ │Agent   │ │Agent   │ │Exec │
└────┬──┘ └───┬────┘ └──┬─────┘ └────┬───┘ └─┬───┘
     │        │         │            │       │
┌────▼────────▼─────────▼────────────▼───────▼────┐
│               Tools & Infrastructure             │
│  ┌─────────┐ ┌──────┐ ┌────────┐ ┌──────────┐   │
│  │File I/O │ │Parse │ │Execute │ │  RAG    │   │
│  │         │ │Model │ │Code    │ │Retriever│   │
│  └─────────┘ └──────┘ └────────┘ └──────────┘   │
└─────────────────────────────────────────────────┘
```

## Workflow Pipeline

```
Input Model & Description
          │
          ▼
    ┌─────────────┐
    │Parse Model  │ → Intermediate Representation
    └──────┬──────┘
           │
           ▼
    ┌──────────────┐
    │Analyze Model │ → Findings & Recommendations
    └──────┬───────┘
           │
           ▼
    ┌──────────────────┐
    │Generate Code     │ → Python Classes & Methods
    └──────┬───────────┘
           │
           ▼
    ┌──────────────────┐
    │Generate Tests    │ → pytest Test Suite
    └──────┬───────────┘
           │
           ▼
    ┌──────────────────┐
    │Execute Tests     │ → Test Results & Coverage
    └──────┬───────────┘
           │
      ┌────▼─────────────────┐
      │ Issues Found?         │
      └────┬──────────┬───────┘
           │ Yes      │ No
           │          │
      ┌────▼──────┐   │
      │Critique   │   │
      │& Suggest  │   │
      │Refactors  │   │
      └────┬──────┘   │
           │          │
           └────┬─────┘
                │
                ▼
         ┌──────────────────┐
         │Final Report      │
         │& Metrics         │
         └──────────────────┘
```

## Future Enhancements

- [ ] Full LLM integration (ChatOpenAI, Claude, Gemini)
- [ ] FAISS/Chroma vector store with embeddings
- [ ] Real PlantUML parser
- [ ] Actual code and test generation
- [ ] Test execution with coverage
- [ ] LLM-as-a-judge evaluation
- [ ] FastAPI web service
- [ ] LangSmith observability
- [ ] Multi-language code generation (Java, TypeScript, etc.)
- [ ] State machine and sequence diagram support
- [ ] Refactoring automation
- [ ] Interactive web dashboard

## References

- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [SOLID Principles](https://en.wikipedia.org/wiki/SOLID)
- [Design Patterns](https://refactoring.guru/design-patterns)
- [UML Best Practices](https://www.uml-diagrams.org/)

## License

MIT License - See LICENSE file for details
