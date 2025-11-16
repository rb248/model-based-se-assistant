"""
LangGraph workflow definition for the Model-Based Software Engineering Assistant.

Implements the multi-agent orchestration graph with proper state management,
routing, and error handling.
"""

import logging
from typing import Any, Dict
from pathlib import Path

from langgraph.graph import StateGraph, END
from langchain_core.pydantic_v1 import BaseModel, Field

from backend.agents import (
    ParserAgent,
    AnalysisAgent,
    CodeGenerationAgent,
    TestGenerationAgent,
    CriticAgent
)
from backend.config import PROJECTS_DIR
from backend.memory import ProjectMemory

logger = logging.getLogger(__name__)


# ============================================================================
# State Definition
# ============================================================================

class WorkflowState(BaseModel):
    """
    Represents the state of the workflow as it progresses through agents.
    """

    # Input
    project_id: str = Field(description="Unique project identifier")
    model_text: str = Field(description="Raw model text")
    model_format: str = Field(default="plantuml", description="Model format type")
    description: str = Field(default="", description="Natural language description")

    # Intermediate results
    model_ir: Dict[str, Any] = Field(default_factory=dict, description="Parsed model IR")
    analysis_report: Dict[str, Any] = Field(default_factory=dict, description="Analysis findings")
    generated_code: Dict[str, Any] = Field(default_factory=dict, description="Generated source code")
    generated_tests: Dict[str, Any] = Field(default_factory=dict, description="Generated test code")
    test_results: Dict[str, Any] = Field(default_factory=dict, description="Test execution results")

    # Final output
    critique: Dict[str, Any] = Field(default_factory=dict, description="Critique and suggestions")
    final_report: Dict[str, Any] = Field(default_factory=dict, description="Final summary report")

    # Execution tracking
    errors: list = Field(default_factory=list, description="Errors encountered")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")


# ============================================================================
# Node Functions
# ============================================================================

def node_parse_model(state: WorkflowState) -> WorkflowState:
    """Parse the input model into an IR."""
    logger.info(f"[PARSE] Processing {state.model_format} model for project {state.project_id}")
    
    try:
        agent = ParserAgent()
        model_ir = agent.parse_model(state.model_text, state.model_format)
        state.model_ir = model_ir
        logger.info("[PARSE] Model parsing completed successfully")
    except Exception as e:
        logger.error(f"[PARSE] Error parsing model: {e}")
        state.errors.append(f"Parser error: {str(e)}")

    return state


def node_analyze_model(state: WorkflowState) -> WorkflowState:
    """Analyze the model for design issues."""
    if not state.model_ir or state.errors:
        logger.warning("[ANALYZE] Skipping analysis due to parse errors")
        return state

    logger.info("[ANALYZE] Analyzing model for design issues")

    try:
        agent = AnalysisAgent()
        analysis = agent.analyze_model(state.model_ir, state.description)
        state.analysis_report = analysis
        logger.info(f"[ANALYZE] Found {len(analysis.get('findings', []))} issues")
    except Exception as e:
        logger.error(f"[ANALYZE] Error analyzing model: {e}")
        state.errors.append(f"Analysis error: {str(e)}")

    return state


def node_generate_code(state: WorkflowState) -> WorkflowState:
    """Generate code from the model."""
    if not state.model_ir or state.errors:
        logger.warning("[CODEGEN] Skipping code generation due to earlier errors")
        return state

    logger.info("[CODEGEN] Generating source code from model")

    try:
        agent = CodeGenerationAgent()
        code = agent.generate_code(
            state.model_ir,
            language="python",
            analysis_report=state.analysis_report,
            apply_refactorings=True,
        )
        state.generated_code = code
        logger.info(f"[CODEGEN] Generated {len(code.get('files', []))} code files")
    except Exception as e:
        logger.error(f"[CODEGEN] Error generating code: {e}")
        state.errors.append(f"Code generation error: {str(e)}")

    return state


def node_generate_tests(state: WorkflowState) -> WorkflowState:
    """Generate test cases based on generated code and analysis."""
    logger.info("[TESTGEN] Generating test cases")
    
    try:
        agent = TestGenerationAgent()
        
        # Generate tests with analysis awareness
        result = agent.generate_tests(
            model_ir=state.model_ir,
            generated_code=state.generated_code,
            analysis_report=state.analysis_report,
            framework="pytest",
            include_integration_tests=True
        )
        
        state.generated_tests = result
        logger.info(
            f"[TESTGEN] Generated {len(result.get('test_files', []))} test files "
            f"with ~{result.get('total_tests', 0)} test cases"
        )
    except Exception as e:
        logger.error(f"[TESTGEN] Error generating tests: {e}", exc_info=True)
        state.errors.append(f"Test generation error: {str(e)}")
    
    return state


def node_save_artifacts(state: WorkflowState) -> WorkflowState:
    """Save generated code and tests to persistent storage."""
    if state.errors:
        logger.warning("[SAVE] Skipping artifact saving due to errors")
        return state
    
    logger.info(f"[SAVE] Saving artifacts for project {state.project_id}")
    
    try:
        # Create project directory
        project_path = PROJECTS_DIR / state.project_id
        project_path.mkdir(parents=True, exist_ok=True)
        
        # Save source files to root
        source_files = state.generated_code.get("files", [])
        for file_info in source_files:
            file_path = project_path / file_info.get("path", "unknown.py")
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(file_info.get("content", ""))
            logger.debug(f"[SAVE] Saved source file: {file_path}")
        
        # Save test files to tests/ subfolder
        test_files = state.generated_tests.get("test_files", [])
        tests_dir = project_path / "tests"
        tests_dir.mkdir(parents=True, exist_ok=True)
        
        for test_file in test_files:
            # Remove 'tests/' prefix if it exists to avoid duplication
            test_path = test_file.get("path", "test_unknown.py")
            if test_path.startswith("tests/"):
                test_path = test_path[6:]  # Remove 'tests/' prefix
            
            file_path = tests_dir / test_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(test_file.get("content", ""))
            logger.debug(f"[SAVE] Saved test file: {file_path}")
        
        # Create __init__.py in tests folder
        init_file = tests_dir / "__init__.py"
        if not init_file.exists():
            init_file.write_text("# Test package init\n")
        
        # Save analysis report
        if state.analysis_report:
            analysis_file = project_path / "analysis_report.json"
            import json
            analysis_file.write_text(json.dumps(state.analysis_report, indent=2))
            logger.debug(f"[SAVE] Saved analysis report: {analysis_file}")
        
        # Save to project memory
        memory = ProjectMemory(state.project_id)
        memory_state = {
            "model_ir": state.model_ir,
            "generated_files": [f.get("path") for f in source_files],
            "test_files": [f.get("path") for f in test_files],
            "analysis_findings": len(state.analysis_report.get("findings", [])),
            "test_results": state.test_results,
            "status": "completed" if not state.errors else "partial"
        }
        memory.save(memory_state)
        
        # Add analysis to history
        if state.analysis_report:
            memory.add_analysis(state.analysis_report)
        
        logger.info(
            f"[SAVE] Saved {len(source_files)} source files and "
            f"{len(test_files)} test files to {project_path}"
        )
        
    except Exception as e:
        logger.error(f"[SAVE] Error saving artifacts: {e}", exc_info=True)
        state.errors.append(f"Artifact saving error: {str(e)}")
    
    return state


def node_run_tests(state: WorkflowState) -> WorkflowState:
    """Execute tests and collect results."""
    if not state.generated_tests or state.errors:
        logger.warning("[EXECUTE] Skipping test execution due to earlier errors or no tests")
        return state

    logger.info("[EXECUTE] Running tests")

    try:
        test_files = state.generated_tests.get("test_files", [])
        if not test_files:
            logger.warning("[EXECUTE] No test files to execute")
            return state
        
        # Write test files to temp directory
        import tempfile
        import subprocess
        from pathlib import Path
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            
            # Write test files
            for test_file in test_files:
                file_path = tmppath / test_file.get("path", "test_unknown.py")
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(test_file.get("content", ""))
            
            # Also write source files if available
            source_files = state.generated_code.get("files", [])
            for source_file in source_files:
                file_path = tmppath / source_file.get("path", "unknown.py")
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(source_file.get("content", ""))

            # Ensure __init__.py exists for package imports (e.g., src/)
            for p in tmppath.rglob('*'):
                if p.is_dir():
                    init_file = p / "__init__.py"
                    if not init_file.exists():
                        try:
                            init_file.write_text("# Package init\n")
                        except Exception:
                            # Some directories (like tmp root) might be non-writable, ignore
                            pass
            
            # Run pytest
            try:
                import os as _os
                env = _os.environ.copy()
                env["PYTHONPATH"] = str(tmppath)
                result = subprocess.run(
                    ["pytest", str(tmppath), "-v", "--tb=short"],
                    cwd=tmpdir,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                # Parse results
                passed = result.stdout.count(" PASSED")
                failed = result.stdout.count(" FAILED")
                errors = result.stdout.count(" ERROR")
                
                state.test_results = {
                    "status": "completed",
                    "passed": passed,
                    "failed": failed,
                    "errors": errors,
                    "exit_code": result.returncode,
                    "stdout": result.stdout[:1000],  # Limit output
                    "stderr": result.stderr[:1000] if result.stderr else ""
                }
                
                logger.info(
                    f"[EXECUTE] Tests completed: "
                    f"{passed} passed, {failed} failed, {errors} errors"
                )
                
            except subprocess.TimeoutExpired:
                logger.error("[EXECUTE] Test execution timeout")
                state.test_results = {
                    "status": "timeout",
                    "message": "Test execution exceeded 60 seconds"
                }
            except FileNotFoundError:
                logger.warning("[EXECUTE] pytest not found, skipping test execution")
                state.test_results = {
                    "status": "skipped",
                    "message": "pytest not installed"
                }
                
    except Exception as e:
        logger.error(f"[EXECUTE] Error executing tests: {e}", exc_info=True)
        state.errors.append(f"Test execution error: {str(e)}")
        state.test_results = {
            "status": "error",
            "message": str(e)
        }

    return state


def node_critique(state: WorkflowState) -> WorkflowState:
    """Review artifacts and propose improvements."""
    logger.info("[CRITIQUE] Reviewing artifacts and generating critique")

    try:
        agent = CriticAgent()
        critique = agent.critique(
            state.analysis_report,
            state.generated_code,
            state.test_results
        )
        state.critique = critique
        logger.info("[CRITIQUE] Critique completed")
    except Exception as e:
        logger.error(f"[CRITIQUE] Error during critique: {e}")
        state.errors.append(f"Critique error: {str(e)}")

    return state


def node_final_report(state: WorkflowState) -> WorkflowState:
    """Assemble the final report."""
    logger.info("[REPORT] Assembling final report")

    state.final_report = {
        "project_id": state.project_id,
        "status": "success" if not state.errors else "partial",
        "errors": state.errors,
        "model_ir_classes": len(state.model_ir.get("classes", [])),
        "generated_files": len(state.generated_code.get("files", [])),
        "test_cases": state.generated_tests.get("total_tests", 0),
        "test_results": state.test_results,
        "analysis_findings": len(state.analysis_report.get("findings", [])),
        "critique_suggestions": len(state.critique.get("refactoring_suggestions", []))
    }

    logger.info("[REPORT] Final report assembled")
    return state


# ============================================================================
# Conditional Routing
# ============================================================================

def should_proceed_to_analysis(state: WorkflowState) -> str:
    """Determine if we should proceed to analysis."""
    if state.errors:
        return "final_report"
    return "analyze"


def should_critique(state: WorkflowState) -> str:
    """Determine if we should run the critic agent."""
    # Run critic if there are findings or test failures
    has_findings = len(state.analysis_report.get("findings", [])) > 0
    has_failures = 0
    if state.test_results:
        try:
            has_failures = int(state.test_results.get("failed", 0)) > 0
        except Exception:
            has_failures = False
    
    if has_findings or has_failures:
        return "critique"
    return "final_report"


# ============================================================================
# Graph Construction
# ============================================================================

def build_workflow_graph() -> StateGraph:
    """
    Build the LangGraph workflow.

    Returns:
        Compiled StateGraph ready for execution.
    """
    graph = StateGraph(WorkflowState)

    # Add nodes
    graph.add_node("parse", node_parse_model)
    graph.add_node("analyze", node_analyze_model)
    graph.add_node("codegen", node_generate_code)
    graph.add_node("testgen", node_generate_tests)
    graph.add_node("save", node_save_artifacts)
    graph.add_node("execute", node_run_tests)
    graph.add_node("critique", node_critique)
    graph.add_node("final_report", node_final_report)

    # Set entry point
    graph.set_entry_point("parse")

    # Add edges
    graph.add_conditional_edges(
        "parse",
        should_proceed_to_analysis,
        {
            "analyze": "analyze",
            "final_report": "final_report"
        }
    )
    graph.add_edge("analyze", "codegen")
    graph.add_edge("codegen", "testgen")
    graph.add_edge("testgen", "save")
    graph.add_edge("save", "execute")
    graph.add_conditional_edges(
        "execute",
        should_critique,
        {
            "critique": "critique",
            "final_report": "final_report"
        }
    )
    graph.add_edge("critique", "final_report")
    graph.add_edge("final_report", END)

    return graph


def get_compiled_graph():
    """Get the compiled workflow graph."""
    graph = build_workflow_graph()
    return graph.compile()
