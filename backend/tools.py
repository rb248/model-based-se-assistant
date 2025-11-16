"""
Tools for the Model-Based Software Engineering Assistant.

Implements parsing, file operations, code execution, and RAG retrieval
that agents can call during the workflow.
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

from backend.config import PROJECTS_DIR, SANDBOX_TIMEOUT, DEFAULT_CODE_LANGUAGE

logger = logging.getLogger(__name__)


# ============================================================================
# File and Project Tools
# ============================================================================

@tool
def read_project_file(project_id: str, file_path: str) -> str:
    """
    Read a file from a project directory.

    Args:
        project_id: The project identifier.
        file_path: Relative path to the file within the project.

    Returns:
        Content of the file.
    """
    try:
        full_path = PROJECTS_DIR / project_id / file_path
        # Security: ensure path is within project directory
        if not str(full_path.resolve()).startswith(str((PROJECTS_DIR / project_id).resolve())):
            raise ValueError("Path traversal not allowed")
        
        with open(full_path, "r") as f:
            return f.read()
    except FileNotFoundError:
        return f"File not found: {file_path}"
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return f"Error reading file: {str(e)}"


@tool
def write_project_file(project_id: str, file_path: str, content: str) -> str:
    """
    Write content to a file in a project directory.

    Args:
        project_id: The project identifier.
        file_path: Relative path to the file within the project.
        content: Content to write.

    Returns:
        Status message.
    """
    try:
        full_path = PROJECTS_DIR / project_id / file_path
        # Security: ensure path is within project directory
        if not str(full_path.resolve()).startswith(str((PROJECTS_DIR / project_id).resolve())):
            raise ValueError("Path traversal not allowed")
        
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_path, "w") as f:
            f.write(content)
        return f"File written successfully: {file_path}"
    except Exception as e:
        logger.error(f"Error writing file {file_path}: {e}")
        return f"Error writing file: {str(e)}"


@tool
def list_project_files(project_id: str, directory: str = ".") -> List[str]:
    """
    List files in a project directory.

    Args:
        project_id: The project identifier.
        directory: Subdirectory to list (default is root).

    Returns:
        List of relative file paths.
    """
    try:
        base_path = PROJECTS_DIR / project_id / directory
        if not base_path.exists():
            return []
        
        files = []
        for item in base_path.rglob("*"):
            if item.is_file():
                rel_path = item.relative_to(PROJECTS_DIR / project_id)
                files.append(str(rel_path))
        return sorted(files)
    except Exception as e:
        logger.error(f"Error listing files: {e}")
        return []


# ============================================================================
# Model Parsing Tools
# ============================================================================

@tool
def parse_plantuml_classes(project_id: str, plantuml_text: str) -> Dict[str, Any]:
    """
    Parse a PlantUML class diagram into an intermediate representation.

    Args:
        project_id: The project identifier.
        plantuml_text: PlantUML text describing classes.

    Returns:
        Dictionary containing parsed classes, attributes, methods, and relationships.
    """
    # TODO: Implement actual PlantUML parsing
    # For MVP, this is a placeholder that would be enhanced with proper parsing
    logger.info(f"Parsing PlantUML for project {project_id}")
    
    return {
        "classes": [],
        "relationships": [],
        "attributes": [],
        "methods": [],
        "raw_input": plantuml_text[:100] + "..." if len(plantuml_text) > 100 else plantuml_text
    }


# ============================================================================
# Code Execution Tools
# ============================================================================

@tool
def run_pytest(project_id: str, test_dir: str = "tests") -> Dict[str, Any]:
    """
    Run pytest on project test files.

    Args:
        project_id: The project identifier.
        test_dir: Directory containing tests (relative to project root).

    Returns:
        Dictionary with test results, including passed/failed counts and output.
    """
    try:
        project_path = PROJECTS_DIR / project_id
        test_path = project_path / test_dir
        
        if not test_path.exists():
            return {
                "status": "no_tests",
                "message": f"Test directory not found: {test_dir}",
                "passed": 0,
                "failed": 0,
                "output": ""
            }
        
        result = subprocess.run(
            ["python", "-m", "pytest", str(test_path), "-v", "--tb=short"],
            cwd=str(project_path),
            capture_output=True,
            text=True,
            timeout=SANDBOX_TIMEOUT
        )
        
        return {
            "status": "completed",
            "returncode": result.returncode,
            "passed": result.stdout.count(" PASSED"),
            "failed": result.stdout.count(" FAILED"),
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    except subprocess.TimeoutExpired:
        logger.error(f"Test execution timed out for project {project_id}")
        return {
            "status": "timeout",
            "message": f"Test execution exceeded {SANDBOX_TIMEOUT} seconds"
        }
    except Exception as e:
        logger.error(f"Error running tests: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


@tool
def run_python_script(project_id: str, script_path: str) -> Dict[str, Any]:
    """
    Run a Python script in a project directory.

    Args:
        project_id: The project identifier.
        script_path: Path to the script (relative to project root).

    Returns:
        Dictionary with execution results.
    """
    try:
        project_path = PROJECTS_DIR / project_id
        full_script_path = project_path / script_path
        
        if not full_script_path.exists():
            return {
                "status": "not_found",
                "message": f"Script not found: {script_path}"
            }
        
        result = subprocess.run(
            ["python", str(full_script_path)],
            cwd=str(project_path),
            capture_output=True,
            text=True,
            timeout=SANDBOX_TIMEOUT
        )
        
        return {
            "status": "completed",
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    except subprocess.TimeoutExpired:
        logger.error(f"Script execution timed out for project {project_id}")
        return {
            "status": "timeout",
            "message": f"Script execution exceeded {SANDBOX_TIMEOUT} seconds"
        }
    except Exception as e:
        logger.error(f"Error running script: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


# ============================================================================
# RAG and Knowledge Retrieval Tools
# ============================================================================

@tool
def retrieve_design_knowledge(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Retrieve relevant design knowledge and principles.

    Args:
        query: Query string about design principles or patterns.
        top_k: Number of results to return.

    Returns:
        List of relevant knowledge snippets with sources.
    """
    # TODO: Implement actual RAG with FAISS or Chroma
    # For MVP, return empty list
    logger.info(f"Retrieving design knowledge for query: {query}")
    return []


# ============================================================================
# Analysis and Validation Tools
# ============================================================================

@tool
def validate_python_syntax(code: str) -> Dict[str, Any]:
    """
    Validate Python code syntax.

    Args:
        code: Python code to validate.

    Returns:
        Dictionary with validation results and any errors.
    """
    try:
        compile(code, "<string>", "exec")
        return {
            "valid": True,
            "errors": []
        }
    except SyntaxError as e:
        return {
            "valid": False,
            "errors": [
                {
                    "line": e.lineno,
                    "offset": e.offset,
                    "message": e.msg,
                    "text": e.text
                }
            ]
        }
    except Exception as e:
        return {
            "valid": False,
            "errors": [{"message": str(e)}]
        }


@tool
def check_code_quality(project_id: str, file_path: str) -> Dict[str, Any]:
    """
    Check code quality using static analysis.

    Args:
        project_id: The project identifier.
        file_path: Path to the file to check.

    Returns:
        Dictionary with quality metrics and issues.
    """
    # TODO: Integrate flake8, mypy, or similar tools
    logger.info(f"Checking code quality for {file_path}")
    return {
        "metrics": {},
        "issues": []
    }
