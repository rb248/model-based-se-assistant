"""
Agent definitions for the Model-Based Software Engineering Assistant.

Each agent is a specialized LLM with specific tools and system prompts
for a particular task in the workflow.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from backend.llms import create_base_llm as _create_base_llm_with_fallback

logger = logging.getLogger(__name__)


def create_base_llm():
    """
    Create a base LLM instance with retry logic and GPT-4o-mini fallback.

    Returns:
        LLM instance wrapped with fallback support (Gemini -> GPT-4o-mini).
    """
    return _create_base_llm_with_fallback(enable_fallback=True)


# ============================================================================
# Parser Agent
# ============================================================================

class ParserAgent:
    """
    Agent responsible for parsing model descriptions (PlantUML, UML, etc.)
    into a normalized intermediate representation (IR).
    """

    def __init__(self, llm=None, name: str = None):
        """Initialize the parser agent.

        Args:
            llm: Optional LLM instance or callable for testing.
            name: Optional name for the agent (useful in tests).
        """
        # Allow injection of a test or mock LLM (callable returns text)
        self.llm = llm or create_base_llm()
        self.name = name or "parser_agent"
        self.logger = logger

    def parse_model(self, model_text: str, model_format: str = "plantuml") -> Dict[str, Any]:
        """
        Parse a model description into IR using Gemini.

        Args:
            model_text: Raw model text.
            model_format: Format of the model (e.g., "plantuml").

        Returns:
            Dictionary containing the parsed model IR.
        """
        self.logger.info(f"[{self.name}] Parsing {model_format} model with Gemini")

        # New implementation: few-shot conversation and Pydantic validation
        system_instr = (
            "You are a strict JSON parser that converts PlantUML class diagrams to JSON.\n\n"
            "CRITICAL RULES:\n"
            "1. Output ONLY valid JSON - no text before or after\n"
            "2. Use EXACTLY these field names (not alternatives):\n"
            "   - 'params' (not 'parameters')\n"
            "   - 'returns' (not 'returnType' or 'return_type')\n"
            "   - 'from' (not 'source')\n"
            "   - 'to' (not 'target')\n"
            "3. Always include all required fields even if empty\n"
            "4. Match the exact JSON structure shown in examples\n\n"
            "Required JSON schema:\n"
            "{\n"
            '  "classes": [{"name": str, "attributes": [{"name": str, "type": str}], '
            '"methods": [{"name": str, "params": [str], "returns": str}], "description": str}],\n'
            '  "relationships": [{"from": str, "to": str, "type": str, "multiplicity": str}],\n'
            '  "notes": [str]\n'
            "}"
        )

        example_user_1 = (
            "PlantUML:\n"
            "class User {\n  +id: int\n  +name: string\n}\n"
        )

        example_assistant_1 = json.dumps(
            {
                "classes": [
                    {
                        "name": "User",
                        "attributes": [{"name": "id", "type": "int"}, {"name": "name", "type": "string"}],
                        "methods": [],
                        "description": ""
                    }
                ],
                "relationships": [],
                "notes": []
            },
            indent=2,
        )

        example_user_2 = (
            "PlantUML:\n"
            "class Calculator {\n"
            "  + add(x, y): int\n"
            "  + divide(a, b): float\n"
            "}\n"
            "class Order {\n  +order_id: int\n}\n"
            "Calculator --> Order\n"
        )

        example_assistant_2 = json.dumps(
            {
                "classes": [
                    {
                        "name": "Calculator",
                        "attributes": [],
                        "methods": [
                            {"name": "add", "params": ["x", "y"], "returns": "int"},
                            {"name": "divide", "params": ["a", "b"], "returns": "float"}
                        ],
                        "description": ""
                    },
                    {"name": "Order", "attributes": [{"name": "order_id", "type": "int"}], "methods": [], "description": ""}
                ],
                "relationships": [{"from": "Calculator", "to": "Order", "type": "association", "multiplicity": "1"}],
                "notes": []
            },
            indent=2,
        )

        example_user_3 = (
            "PlantUML:\n"
            "class Animal {\n"
            "  + makeSound(): void\n"
            "}\n"
            "class Dog {\n"
            "  - breed: string\n"
            "}\n"
            "Dog --|> Animal\n"
        )

        example_assistant_3 = json.dumps(
            {
                "classes": [
                    {
                        "name": "Animal",
                        "attributes": [],
                        "methods": [{"name": "makeSound", "params": [], "returns": "void"}],
                        "description": ""
                    },
                    {
                        "name": "Dog",
                        "attributes": [{"name": "breed", "type": "string"}],
                        "methods": [],
                        "description": ""
                    }
                ],
                "relationships": [{"from": "Dog", "to": "Animal", "type": "inheritance", "multiplicity": "1"}],
                "notes": []
            },
            indent=2,
        )

        try:
            self.logger.info(f"[{self.name}] LLM check: callable={callable(self.llm)}, has_generate={hasattr(self.llm, 'generate')}")
            # Mock/callable LLM path (e.g., DummyLLM)
            if callable(self.llm) and not hasattr(self.llm, "generate"):
                self.logger.info(f"[{self.name}] Taking callable LLM path")
                prompt = (
                    f"SYSTEM:\n{system_instr}\n\n"
                    f"EXAMPLE 1 USER:\n{example_user_1}\nEXAMPLE 1 ASSISTANT:\n{example_assistant_1}\n\n"
                    f"EXAMPLE 2 USER:\n{example_user_2}\nEXAMPLE 2 ASSISTANT:\n{example_assistant_2}\n\n"
                    f"EXAMPLE 3 USER:\n{example_user_3}\nEXAMPLE 3 ASSISTANT:\n{example_assistant_3}\n\n"
                    f"USER INPUT:\n{model_text}\n\nRespond with JSON only."
                )
                try:
                    raw = self.llm(prompt)
                    self.logger.info(f"[{self.name}] Callable path: raw response length={len(raw) if raw else 0}, first 200: {raw[:200] if raw else 'EMPTY'}")
                except Exception as call_error:
                    self.logger.error(f"[{self.name}] Error calling LLM: {call_error}")
                    raise
                
                # Clean up response - remove markdown code blocks if present
                raw = raw.strip()
                if raw.startswith("```json"):
                    raw = raw[7:]
                if raw.startswith("```"):
                    raw = raw[3:]
                if raw.endswith("```"):
                    raw = raw[:-3]
                raw = raw.strip()
                
                self.logger.info(f"[{self.name}] Cleaned response length: {len(raw)}")
                parsed = json.loads(raw)
            else:
                # Real LLM path: create chat message list
                messages = [
                    SystemMessage(content=system_instr),
                    HumanMessage(content=f"EXAMPLE 1:\n{example_user_1}"),
                    AIMessage(content=example_assistant_1),
                    HumanMessage(content=f"EXAMPLE 2:\n{example_user_2}"),
                    AIMessage(content=example_assistant_2),
                    HumanMessage(content=f"EXAMPLE 3:\n{example_user_3}"),
                    AIMessage(content=example_assistant_3),
                    HumanMessage(content=f"Parse the following PlantUML and return the JSON only:\n{model_text}"),
                ]

                chat = self.llm
                text = None
                
                # Try direct invocation first (most LangChain chat models support this)
                try:
                    response = chat.invoke(messages)
                    if hasattr(response, "content"):
                        text = response.content
                    elif isinstance(response, str):
                        text = response
                    else:
                        text = str(response)
                    self.logger.debug(f"[{self.name}] Used invoke() method successfully")
                except (AttributeError, TypeError) as e:
                    self.logger.debug(f"[{self.name}] invoke() failed: {e}, trying alternatives")
                    # Fallback: try calling directly
                    try:
                        maybe = chat(messages)
                        if isinstance(maybe, str):
                            text = maybe
                        elif hasattr(maybe, "content"):
                            text = maybe.content
                        else:
                            text = getattr(maybe, "text", None) or str(maybe)
                        self.logger.debug(f"[{self.name}] Used direct call successfully")
                    except Exception as e2:
                        self.logger.debug(f"[{self.name}] Direct call failed: {e2}, trying generate()")
                        # Last resort: try generate
                        if hasattr(chat, "generate"):
                            resp = chat.generate([messages])
                            gens = getattr(resp, "generations", None)
                            if gens and isinstance(gens, list) and len(gens) > 0:
                                if isinstance(gens[0], list) and len(gens[0]) > 0:
                                    text = gens[0][0].text
                                else:
                                    text = gens[0].text if hasattr(gens[0], 'text') else None
                            self.logger.debug(f"[{self.name}] Used generate() method")

                if not text:
                    self.logger.error(f"[{self.name}] All extraction methods failed. Response type: {type(chat)}")
                    raise RuntimeError("Could not extract text from LLM response")

                self.logger.info(f"[{self.name}] Extracted text length: {len(text) if text else 0}")

                # Clean up response - remove markdown code blocks if present
                text = text.strip()
                if text.startswith("```json"):
                    text = text[7:]
                if text.startswith("```"):
                    text = text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()

                # Log the raw response for debugging
                self.logger.info(f"[{self.name}] Cleaned text length: {len(text)}, first 200 chars: {text[:200]}")

                parsed = json.loads(text)

            # Validate using Pydantic schema
            from backend.schema import ModelIR

            validated = ModelIR.model_validate(parsed)
            return validated.model_dump(by_alias=True)

        except Exception as e:
            self.logger.error(f"[{self.name}] Error parsing model: {e}")
            return {
                "classes": [],
                "relationships": [],
                "error": str(e)
            }


# ============================================================================
# Analysis Agent
# ============================================================================

class AnalysisAgent:
    """
    Agent responsible for analyzing models against design principles,
    detecting issues, and providing recommendations using RAG.
    """

    def __init__(self, llm=None, retriever=None, name: str = None):
        """
        Initialize the analysis agent.
        
        Args:
            llm: Optional LLM instance or callable for testing.
            retriever: Optional RAG retriever for design knowledge.
            name: Optional name for the agent (useful in tests).
        """
        self.llm = llm or create_base_llm()
        self.name = name or "analysis_agent"
        self.logger = logger
        self.retriever = retriever

    def _calculate_lcom(self, cls: Dict[str, Any]) -> float:
        """
        Calculate Lack of Cohesion of Methods (LCOM) metric.
        
        LCOM measures the cohesiveness of a class by counting method pairs
        that don't share attributes. Lower is better.
        
        Args:
            cls: Class dictionary with methods and attributes.
            
        Returns:
            LCOM score (0.0 to 1.0, where 1.0 is least cohesive).
        """
        methods = cls.get("methods", [])
        attributes = cls.get("attributes", [])
        
        if not methods or not attributes:
            return 0.0
        
        # For simplicity, we use a heuristic based on parameter names
        # In real implementation, would need dataflow analysis
        method_attrs = []
        for method in methods:
            params = method.get("params", [])
            # Assume methods using similar parameter names work on related data
            method_attrs.append(set(params))
        
        # Count pairs of methods that share no attributes
        non_shared_pairs = 0
        total_pairs = 0
        
        for i in range(len(method_attrs)):
            for j in range(i + 1, len(method_attrs)):
                total_pairs += 1
                if not method_attrs[i].intersection(method_attrs[j]):
                    non_shared_pairs += 1
        
        if total_pairs == 0:
            return 0.0
        
        return non_shared_pairs / total_pairs

    def _detect_god_classes(self, model_ir: Dict[str, Any], thresholds: Dict[str, int] = None) -> List[Dict[str, Any]]:
        """
        Detect god classes (classes with too many responsibilities).
        
        Criteria:
        - More than method_threshold methods (default: 10)
        - More than attribute_threshold attributes (default: 5)
        - More than relationship_threshold outgoing relationships (default: 7)
        - High LCOM score (> 0.7) indicating low cohesion
        
        Args:
            model_ir: Model intermediate representation.
            thresholds: Optional dict with 'methods', 'attributes', 'relationships', 'lcom'.
            
        Returns:
            List of findings for god classes.
        """
        if thresholds is None:
            thresholds = {
                "methods": 10,
                "attributes": 5,
                "relationships": 7,
                "lcom": 0.7
            }
        
        findings = []
        classes = model_ir.get("classes", [])
        relationships = model_ir.get("relationships", [])
        
        for cls in classes:
            class_name = cls.get("name", "Unknown")
            methods = cls.get("methods", [])
            attributes = cls.get("attributes", [])
            
            # Count outgoing relationships
            outgoing_rels = sum(1 for r in relationships if r.get("from") == class_name)
            
            # Calculate cohesion
            lcom = self._calculate_lcom(cls)
            
            issues = []
            if len(methods) > thresholds["methods"]:
                issues.append(f"{len(methods)} methods")
            if len(attributes) > thresholds["attributes"]:
                issues.append(f"{len(attributes)} attributes")
            if outgoing_rels > thresholds["relationships"]:
                issues.append(f"{outgoing_rels} dependencies")
            if lcom > thresholds["lcom"]:
                issues.append(f"low cohesion (LCOM={lcom:.2f})")
            
            if issues:
                findings.append({
                    "severity": "critical" if len(issues) >= 2 else "warning",
                    "issue": f"Class '{class_name}' appears to be a God Class with {', '.join(issues)}. This violates Single Responsibility Principle.",
                    "affected_entities": [class_name],
                    "violated_principle": "SRP",
                    "category": "solid"
                })
        
        return findings

    def _cluster_methods_by_responsibility(self, methods: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Cluster methods by semantic responsibility using keyword analysis.
        
        Args:
            methods: List of method dictionaries.
            
        Returns:
            Dict mapping responsibility domain to list of method names.
        """
        # Define responsibility domains with associated keywords
        domains = {
            "data_access": ["get", "set", "fetch", "load", "save", "store", "persist", "query", "find", "retrieve"],
            "business_logic": ["calculate", "process", "compute", "validate", "verify", "check", "apply", "execute"],
            "presentation": ["display", "render", "format", "print", "show", "draw", "present"],
            "communication": ["send", "receive", "notify", "email", "message", "broadcast", "publish"],
            "lifecycle": ["create", "init", "destroy", "cleanup", "dispose", "close", "open", "start", "stop"],
            "transformation": ["convert", "transform", "map", "parse", "serialize", "deserialize", "encode", "decode"],
            "logging": ["log", "trace", "debug", "error", "warn", "audit"],
            "security": ["auth", "authorize", "authenticate", "encrypt", "decrypt", "hash", "verify"],
        }
        
        clusters = {domain: [] for domain in domains}
        uncategorized = []
        
        for method in methods:
            method_name = method.get("name", "").lower()
            categorized = False
            
            for domain, keywords in domains.items():
                if any(keyword in method_name for keyword in keywords):
                    clusters[domain].append(method.get("name", ""))
                    categorized = True
                    break
            
            if not categorized:
                uncategorized.append(method.get("name", ""))
        
        # Remove empty clusters
        clusters = {k: v for k, v in clusters.items() if v}
        if uncategorized:
            clusters["uncategorized"] = uncategorized
        
        return clusters

    def _detect_solid_violations(self, model_ir: Dict[str, Any], thresholds: Dict[str, int] = None) -> List[Dict[str, Any]]:
        """
        Detect SOLID principle violations with semantic analysis.
        
        Checks:
        - SRP: Classes with multiple responsibility domains, names suggesting multiple responsibilities
        - ISP: Abstract classes/interfaces with too many methods or unrelated method groups
        - DIP: Direct dependencies on concrete implementations
        - OCP: Classes with many conditional branches (heuristic)
        
        Args:
            model_ir: Model intermediate representation.
            thresholds: Optional dict with 'srp_methods', 'srp_domains', 'isp_methods'.
            
        Returns:
            List of findings for SOLID violations.
        """
        if thresholds is None:
            thresholds = {
                "srp_methods": 5,
                "srp_domains": 2,
                "isp_methods": 7,
                "isp_unrelated_ratio": 0.6
            }
        
        findings = []
        classes = model_ir.get("classes", [])
        relationships = model_ir.get("relationships", [])
        
        # SRP: Semantic clustering analysis
        for cls in classes:
            class_name = cls.get("name", "")
            methods = cls.get("methods", [])
            
            if len(methods) > thresholds["srp_methods"]:
                # Cluster methods by responsibility
                clusters = self._cluster_methods_by_responsibility(methods)
                
                # If methods span multiple domains, likely SRP violation
                if len(clusters) > thresholds["srp_domains"]:
                    domain_summary = ", ".join([f"{domain} ({len(methods_list)} methods)" 
                                                for domain, methods_list in clusters.items()])
                    findings.append({
                        "severity": "warning",
                        "issue": f"Class '{class_name}' violates SRP - methods span {len(clusters)} responsibility domains: {domain_summary}. Consider splitting into focused classes.",
                        "affected_entities": [class_name],
                        "violated_principle": "SRP",
                        "category": "solid"
                    })
        
        # SRP: Check for classes with names indicating multiple responsibilities (legacy check)
        srp_indicators = ["Manager", "Handler", "Util", "Helper", "Service", "Controller"]
        for cls in classes:
            class_name = cls.get("name", "")
            methods = cls.get("methods", [])
            
            if any(indicator in class_name for indicator in srp_indicators) and len(methods) > thresholds["srp_methods"]:
                # Check if already reported by semantic analysis
                if not any(f["affected_entities"] == [class_name] and "responsibility domains" in f["issue"] 
                          for f in findings):
                    findings.append({
                        "severity": "warning",
                        "issue": f"Class '{class_name}' may violate SRP - name suggests utility/coordinator role with {len(methods)} methods. Consider splitting responsibilities.",
                        "affected_entities": [class_name],
                        "violated_principle": "SRP",
                        "category": "solid"
                    })
        
        # ISP: Check for abstract classes/interfaces with too many or unrelated methods
        abstract_keywords = ["Abstract", "Interface", "Base", "IFace"]
        for cls in classes:
            class_name = cls.get("name", "")
            description = cls.get("description", "").lower()
            methods = cls.get("methods", [])
            
            is_abstract = ("abstract" in description or "interface" in description or
                          any(keyword in class_name for keyword in abstract_keywords) or
                          class_name.startswith("I") and class_name[1].isupper())
            
            if is_abstract and len(methods) > thresholds["isp_methods"]:
                # Check if methods are semantically related
                clusters = self._cluster_methods_by_responsibility(methods)
                if len(clusters) > 1:
                    findings.append({
                        "severity": "warning",
                        "issue": f"Interface '{class_name}' has {len(methods)} methods spanning {len(clusters)} domains, violating ISP. Consider splitting into focused interfaces.",
                        "affected_entities": [class_name],
                        "violated_principle": "ISP",
                        "category": "solid"
                    })
        
        # DIP: Check for dependencies on concrete implementations
        concrete_indicators = ["MySQL", "Postgres", "SQLite", "File", "HTTP", "REST", "JSON", "XML", "CSV"]
        abstract_names = set()
        for cls in classes:
            name = cls.get("name", "")
            desc = cls.get("description", "").lower()
            if "abstract" in desc or "interface" in desc or name.startswith("I") and name[1].isupper():
                abstract_names.add(name)
        
        for rel in relationships:
            from_class = rel.get("from", "")
            to_class = rel.get("to", "")
            rel_type = rel.get("type", "").lower()
            
            # Skip inheritance relationships
            if rel_type in ["inheritance", "extends", "inherits"]:
                continue
            
            # Check if depending on concrete implementation
            if any(indicator in to_class for indicator in concrete_indicators):
                findings.append({
                    "severity": "info",
                    "issue": f"Class '{from_class}' depends on concrete implementation '{to_class}'. Consider depending on an abstraction instead (DIP).",
                    "affected_entities": [from_class, to_class],
                    "violated_principle": "DIP",
                    "category": "solid"
                })
            # Check if high-level depends directly on low-level without abstraction
            elif to_class not in abstract_names and from_class not in abstract_names:
                # Heuristic: if from_class looks high-level and to_class looks low-level
                high_level_indicators = ["Service", "Controller", "Manager", "Facade"]
                low_level_indicators = ["Repository", "DAO", "Client", "Adapter"]
                if (any(ind in from_class for ind in high_level_indicators) and
                    any(ind in to_class for ind in low_level_indicators)):
                    findings.append({
                        "severity": "info",
                        "issue": f"High-level class '{from_class}' directly depends on low-level '{to_class}' without abstraction (DIP).",
                        "affected_entities": [from_class, to_class],
                        "violated_principle": "DIP",
                        "category": "solid"
                    })
        
        return findings

    def _detect_fan_in_fan_out(self, model_ir: Dict[str, Any], thresholds: Dict[str, int] = None) -> List[Dict[str, Any]]:
        """
        Detect classes with excessive incoming (fan-in) or outgoing (fan-out) dependencies.
        
        High fan-out indicates tight coupling. High fan-in may indicate god classes or central hubs.
        
        Args:
            model_ir: Model intermediate representation.
            thresholds: Optional dict with 'fan_in', 'fan_out'.
            
        Returns:
            List of findings for coupling issues.
        """
        if thresholds is None:
            thresholds = {"fan_in": 7, "fan_out": 7}
        
        findings = []
        relationships = model_ir.get("relationships", [])
        classes = model_ir.get("classes", [])
        class_names = {cls.get("name") for cls in classes}
        
        # Calculate fan-in and fan-out for each class
        fan_in = {name: 0 for name in class_names}
        fan_out = {name: 0 for name in class_names}
        
        for rel in relationships:
            from_class = rel.get("from", "")
            to_class = rel.get("to", "")
            rel_type = rel.get("type", "").lower()
            
            # Skip inheritance (less problematic than other dependencies)
            if rel_type in ["inheritance", "extends", "inherits"]:
                continue
            
            if from_class in class_names:
                fan_out[from_class] += 1
            if to_class in class_names:
                fan_in[to_class] += 1
        
        # Report high fan-out (tight coupling)
        for class_name, count in fan_out.items():
            if count > thresholds["fan_out"]:
                findings.append({
                    "severity": "warning",
                    "issue": f"Class '{class_name}' has high fan-out ({count} dependencies), indicating tight coupling. Consider reducing dependencies or introducing facades/mediators.",
                    "affected_entities": [class_name],
                    "violated_principle": None,
                    "category": "coupling"
                })
        
        # Report high fan-in (potential bottleneck or god class)
        for class_name, count in fan_in.items():
            if count > thresholds["fan_in"]:
                findings.append({
                    "severity": "info",
                    "issue": f"Class '{class_name}' has high fan-in ({count} dependents). This may indicate a central hub or god class. Verify responsibilities are appropriate.",
                    "affected_entities": [class_name],
                    "violated_principle": None,
                    "category": "coupling"
                })
        
        return findings

    def _detect_long_parameter_lists(self, model_ir: Dict[str, Any], threshold: int = 5) -> List[Dict[str, Any]]:
        """
        Detect methods with too many parameters, suggesting missing abstractions.
        
        Args:
            model_ir: Model intermediate representation.
            threshold: Max acceptable parameter count (default: 5).
            
        Returns:
            List of findings for long parameter lists.
        """
        findings = []
        classes = model_ir.get("classes", [])
        
        for cls in classes:
            class_name = cls.get("name", "")
            methods = cls.get("methods", [])
            
            for method in methods:
                method_name = method.get("name", "")
                params = method.get("params", [])
                
                if len(params) > threshold:
                    findings.append({
                        "severity": "warning",
                        "issue": f"Method '{class_name}.{method_name}' has {len(params)} parameters (>{threshold}). Consider introducing a parameter object or builder pattern.",
                        "affected_entities": [class_name],
                        "violated_principle": None,
                        "category": "structure"
                    })
        
        return findings

    def _detect_data_clumps(self, model_ir: Dict[str, Any], min_occurrences: int = 3) -> List[Dict[str, Any]]:
        """
        Detect groups of parameters/attributes that appear together frequently,
        suggesting missing value objects.
        
        Args:
            model_ir: Model intermediate representation.
            min_occurrences: Minimum times a group must appear (default: 3).
            
        Returns:
            List of findings for data clumps.
        """
        findings = []
        classes = model_ir.get("classes", [])
        
        # Collect parameter groups from methods
        param_groups = []
        method_locations = []
        
        for cls in classes:
            class_name = cls.get("name", "")
            methods = cls.get("methods", [])
            
            for method in methods:
                method_name = method.get("name", "")
                params = method.get("params", [])
                
                if len(params) >= 3:
                    # Generate all 3-param combinations
                    for i in range(len(params)):
                        for j in range(i + 1, len(params)):
                            for k in range(j + 1, len(params)):
                                group = tuple(sorted([params[i], params[j], params[k]]))
                                param_groups.append(group)
                                method_locations.append((class_name, method_name))
        
        # Count occurrences of each group
        from collections import Counter
        group_counts = Counter(param_groups)
        
        reported_groups = set()
        for group, count in group_counts.items():
            if count >= min_occurrences and group not in reported_groups:
                # Find all locations
                locations = [method_locations[i] for i, g in enumerate(param_groups) if g == group]
                unique_locations = list(set(locations))[:3]  # Limit to 3 examples
                
                location_str = ", ".join([f"{cls}.{meth}" for cls, meth in unique_locations])
                affected_classes = list(set([cls for cls, _ in unique_locations]))
                
                findings.append({
                    "severity": "info",
                    "issue": f"Data clump detected: parameters {group} appear together {count} times in methods like {location_str}. Consider introducing a value object or data class.",
                    "affected_entities": affected_classes,
                    "violated_principle": None,
                    "category": "structure"
                })
                
                reported_groups.add(group)
        
        return findings

    def _detect_missing_abstractions(self, model_ir: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect missing abstractions (similar classes that should share an interface).
        
        Looks for:
        - Multiple classes with similar method names
        - Classes with common prefixes/suffixes suggesting a pattern
        
        Args:
            model_ir: Model intermediate representation.
            
        Returns:
            List of findings for missing abstractions.
        """
        findings = []
        classes = model_ir.get("classes", [])
        
        # Group classes by common method names
        method_groups = {}
        for cls in classes:
            class_name = cls.get("name", "")
            methods = cls.get("methods", [])
            
            for method in methods:
                method_name = method.get("name", "")
                if method_name:
                    if method_name not in method_groups:
                        method_groups[method_name] = []
                    method_groups[method_name].append(class_name)
        
        # Find common method patterns across multiple classes
        for method_name, class_list in method_groups.items():
            if len(class_list) >= 3:
                findings.append({
                    "severity": "info",
                    "issue": f"Multiple classes ({', '.join(class_list)}) implement method '{method_name}'. Consider introducing a common interface or abstract class.",
                    "affected_entities": class_list,
                    "violated_principle": None,
                    "category": "pattern"
                })
        
        # Check for classes with common suffixes suggesting a pattern
        suffixes = {}
        for cls in classes:
            class_name = cls.get("name", "")
            # Extract potential suffix patterns
            for suffix in ["Sender", "Handler", "Provider", "Strategy", "Factory"]:
                if class_name.endswith(suffix):
                    if suffix not in suffixes:
                        suffixes[suffix] = []
                    suffixes[suffix].append(class_name)
        
        for suffix, class_list in suffixes.items():
            if len(class_list) >= 3:
                findings.append({
                    "severity": "info",
                    "issue": f"Multiple {suffix} classes found ({', '.join(class_list)}). Consider using a common interface with Strategy or Factory pattern.",
                    "affected_entities": class_list,
                    "violated_principle": None,
                    "category": "pattern"
                })
        
        return findings

    def _detect_circular_dependencies(self, model_ir: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect circular dependencies in the model.
        
        Uses depth-first search to find cycles in the dependency graph.
        
        Args:
            model_ir: Model intermediate representation.
            
        Returns:
            List of findings for circular dependencies.
        """
        findings = []
        relationships = model_ir.get("relationships", [])
        
        # Build adjacency list
        graph = {}
        for rel in relationships:
            from_class = rel.get("from", "")
            to_class = rel.get("to", "")
            if from_class and to_class:
                if from_class not in graph:
                    graph[from_class] = []
                graph[from_class].append(to_class)
        
        # DFS to detect cycles
        visited = set()
        rec_stack = set()
        cycles = []
        
        def dfs(node, path):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor, path.copy()):
                        return True
                elif neighbor in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    cycles.append(cycle)
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in graph:
            if node not in visited:
                dfs(node, [])
        
        # Report unique cycles
        seen_cycles = set()
        for cycle in cycles:
            cycle_key = tuple(sorted(cycle))
            if cycle_key not in seen_cycles:
                seen_cycles.add(cycle_key)
                cycle_str = " → ".join(cycle)
                findings.append({
                    "severity": "critical",
                    "issue": f"Circular dependency detected: {cycle_str}. This creates tight coupling and makes the system brittle.",
                    "affected_entities": list(set(cycle)),
                    "violated_principle": "DIP",
                    "category": "coupling"
                })
        
        return findings

    def _calculate_metrics(self, model_ir: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate quality metrics for the model.
        
        Metrics:
        - avg_methods_per_class
        - max_methods_per_class
        - avg_attributes_per_class
        - max_attributes_per_class
        - relationship_density (relationships per class)
        - max_inheritance_depth
        
        Args:
            model_ir: Model intermediate representation.
            
        Returns:
            Dictionary of computed metrics.
        """
        classes = model_ir.get("classes", [])
        relationships = model_ir.get("relationships", [])
        
        if not classes:
            return {}
        
        method_counts = [len(cls.get("methods", [])) for cls in classes]
        attribute_counts = [len(cls.get("attributes", [])) for cls in classes]
        
        metrics = {
            "avg_methods_per_class": sum(method_counts) / len(classes) if classes else 0.0,
            "max_methods_per_class": max(method_counts) if method_counts else 0,
            "avg_attributes_per_class": sum(attribute_counts) / len(classes) if classes else 0.0,
            "max_attributes_per_class": max(attribute_counts) if attribute_counts else 0,
            "relationship_density": len(relationships) / len(classes) if classes else 0.0,
            "total_classes": len(classes),
            "total_relationships": len(relationships)
        }
        
        # Calculate inheritance depth (simplified: count inheritance relationships)
        inheritance_rels = [r for r in relationships if r.get("type", "").lower() in ["inheritance", "extends", "inherits"]]
        metrics["inheritance_count"] = len(inheritance_rels)
        
        return metrics

    def _build_rag_query(
        self,
        description: str,
        findings: List[Dict[str, Any]],
        metrics: Dict[str, Any],
        max_findings: int = 4
    ) -> str:
        """Create a rich RAG query that includes semantic tags and key metrics."""

        base_query = (description or "software architecture review").strip()

        # Collect tags from findings
        tags = set()
        focus_points = []
        for finding in findings[:max_findings]:
            principle = finding.get("violated_principle")
            category = finding.get("category")
            severity = finding.get("severity")
            entities = finding.get("affected_entities", [])

            if principle:
                tags.add(f"principle:{principle}")
            if category:
                tags.add(f"category:{category}")
            if severity:
                tags.add(f"severity:{severity}")

            issue = finding.get("issue", "")[:60]
            entity_snippet = ", ".join(entities[:2]) if entities else "model"
            label = principle or category or "issue"
            focus_points.append(f"{label}: {entity_snippet} ({issue})")

            # Heuristic tags for known smells
            issue_lower = issue.lower()
            if "god class" in issue_lower:
                tags.add("smell:god_class")
            if "circular" in issue_lower:
                tags.add("smell:circular_dependency")

        if not focus_points:
            focus_points.append("general design best practices")

        # Summarize metrics into lightweight tags
        metric_keys = [
            "max_methods_per_class",
            "avg_methods_per_class",
            "relationship_density",
            "total_classes",
            "inheritance_count"
        ]
        metric_bits = []
        for key in metric_keys:
            if key in metrics:
                value = metrics[key]
                metric_bits.append(f"{key}={round(value, 2) if isinstance(value, float) else value}")

        metric_section = f"metrics[{'; '.join(metric_bits)}]" if metric_bits else "metrics[none]"
        tag_section = f"tags[{', '.join(sorted(tags))}]" if tags else "tags[general]"
        focus_section = "; ".join(focus_points)

        return f"Design guidance for {base_query}. Focus: {focus_section}. {tag_section} {metric_section}"

    def analyze_model(self, model_ir: Dict[str, Any], description: str = "") -> Dict[str, Any]:
        """
        Analyze a model IR for design issues and improvements using RAG and LLM.

        Args:
            model_ir: Intermediate representation of the model.
            description: Natural language description of the model.

        Returns:
            Dictionary containing analysis report with findings and recommendations.
        """
        self.logger.info(f"[{self.name}] Analyzing model with {len(model_ir.get('classes', []))} classes")

        try:
            # Step 1: Run deterministic detectors with configurable thresholds
            self.logger.debug(f"[{self.name}] Running deterministic detectors")
            
            # Thresholds can be customized per project
            god_class_thresholds = {"methods": 10, "attributes": 5, "relationships": 7, "lcom": 0.7}
            solid_thresholds = {"srp_methods": 5, "srp_domains": 2, "isp_methods": 7, "isp_unrelated_ratio": 0.6}
            coupling_thresholds = {"fan_in": 7, "fan_out": 7}
            
            deterministic_findings = []
            deterministic_findings.extend(self._detect_god_classes(model_ir, god_class_thresholds))
            deterministic_findings.extend(self._detect_solid_violations(model_ir, solid_thresholds))
            deterministic_findings.extend(self._detect_missing_abstractions(model_ir))
            deterministic_findings.extend(self._detect_circular_dependencies(model_ir))
            deterministic_findings.extend(self._detect_fan_in_fan_out(model_ir, coupling_thresholds))
            deterministic_findings.extend(self._detect_long_parameter_lists(model_ir, threshold=5))
            deterministic_findings.extend(self._detect_data_clumps(model_ir, min_occurrences=3))

            print(f"\n[DEBUG] Deterministic findings:")
            for f in deterministic_findings:
                print(json.dumps(f, indent=2))
            
            # Step 2: Calculate metrics
            metrics = self._calculate_metrics(model_ir)
            self.logger.debug(f"[{self.name}] Computed metrics: {metrics}")
            
            # Step 3: Retrieve design knowledge if retriever available
            retrieved_knowledge = []
            if self.retriever:
                try:
                    query = self._build_rag_query(description, deterministic_findings, metrics)
                    self.logger.debug(f"[{self.name}] RAG query: {query}")
                    print(f"\n[DEBUG] RAG query sent: {query}\n")
                    docs = self.retriever.invoke(query)
                    retrieved_knowledge = [
                        {
                            "title": doc.metadata.get("title", "Unknown"),
                            "content": doc.page_content[:300],
                            "category": doc.metadata.get("category", "Unknown")
                        }
                        for doc in docs[:3]
                    ]
                    self.logger.debug(f"[{self.name}] Retrieved {len(retrieved_knowledge)} knowledge documents")
                except Exception as e:
                    self.logger.warning(f"[{self.name}] RAG retrieval failed: {e}")
            
            # Step 4: Build prompt with few-shot examples and context
            system_prompt = """You are an expert software architect reviewing UML models.
Analyze the model using:
1. Pre-computed metrics and detections (provided below)
2. Retrieved design principles (SOLID, patterns, best practices)
3. Your expertise in software architecture

Focus on: SOLID violations, missing abstractions, god objects, tight coupling, unclear responsibilities.

CRITICAL RULES:
1. Output ONLY valid JSON - no text before or after
2. Use EXACTLY these field names:
   - "severity" (not "level" or "priority") with values: "critical", "warning", "info"
   - "affected_entities" (not "affected_classes" or "entities")
   - "violated_principle" (not "principle" or "violation")
3. Always include all required fields even if empty
4. Match the exact JSON structure shown in examples

Required JSON schema:
{
  "findings": [{"severity": str, "issue": str, "affected_entities": [str], "violated_principle": str|null, "category": str}],
  "recommendations": [{"title": str, "description": str, "priority": str, "affected_entities": [str], "design_pattern": str|null, "rationale": str}],
  "patterns_detected": [str],
  "quality_score": float,
  "quality_metrics": {...},
  "summary": str
}"""

            # Few-shot Example 1: God class detection
            example_user_1 = """Analyze this model:

Model Structure:
{
  "classes": [{
    "name": "UserManager",
    "methods": [{"name": "createUser"}, {"name": "deleteUser"}, {"name": "sendEmail"}, 
                {"name": "validateEmail"}, {"name": "hashPassword"}, {"name": "logActivity"},
                {"name": "generateReport"}, {"name": "exportData"}, {"name": "importData"},
                {"name": "sendNotification"}, {"name": "trackAnalytics"}, {"name": "managePermissions"}],
    "attributes": [{"name": "users"}, {"name": "emailService"}, {"name": "logger"}, 
                   {"name": "database"}, {"name": "reportGenerator"}, {"name": "analytics"}]
  }]
}

Pre-analysis Findings:
- God class detected: UserManager with 12 methods, 6 attributes

Metrics: {"max_methods_per_class": 12, "max_attributes_per_class": 6}"""

            example_assistant_1 = json.dumps({
                "findings": [
                    {
                        "severity": "critical",
                        "issue": "UserManager is a God Class with 12 methods and 6 attributes, violating Single Responsibility Principle. It handles user management, email, validation, logging, reporting, and analytics.",
                        "affected_entities": ["UserManager"],
                        "violated_principle": "SRP",
                        "category": "solid"
                    }
                ],
                "recommendations": [
                    {
                        "title": "Split UserManager into focused classes",
                        "description": "Extract: UserRepository (user CRUD), EmailService (email operations), UserValidator (validation), ReportGenerator (reporting), AnalyticsTracker (analytics)",
                        "priority": "high",
                        "affected_entities": ["UserManager"],
                        "design_pattern": None,
                        "rationale": "Each class will have a single, well-defined responsibility, improving maintainability and testability"
                    }
                ],
                "patterns_detected": [],
                "quality_score": 0.3,
                "quality_metrics": {"max_methods_per_class": 12, "max_attributes_per_class": 6},
                "summary": "Critical SRP violation detected. UserManager should be split into 5 focused classes."
            })

            # Few-shot Example 2: Missing abstraction
            example_user_2 = """Analyze this model:

Model Structure:
{
  "classes": [
    {"name": "Order", "methods": [{"name": "calculateTotal"}]},
    {"name": "PayPalPayment", "methods": [{"name": "processPayment"}]},
    {"name": "StripePayment", "methods": [{"name": "processPayment"}]},
    {"name": "CreditCardPayment", "methods": [{"name": "processPayment"}]}
  ],
  "relationships": [
    {"from": "Order", "to": "PayPalPayment", "type": "dependency"},
    {"from": "Order", "to": "StripePayment", "type": "dependency"},
    {"from": "Order", "to": "CreditCardPayment", "type": "dependency"}
  ]
}

Pre-analysis Findings:
- Multiple classes (PayPalPayment, StripePayment, CreditCardPayment) implement method 'processPayment'

Retrieved Knowledge:
- Strategy Pattern: Define family of algorithms, encapsulate each, make them interchangeable
- Dependency Inversion Principle: Depend on abstractions, not concretions"""

            example_assistant_2 = json.dumps({
                "findings": [
                    {
                        "severity": "warning",
                        "issue": "Order depends on multiple concrete payment implementations (PayPalPayment, StripePayment, CreditCardPayment). This violates Dependency Inversion Principle and creates tight coupling.",
                        "affected_entities": ["Order", "PayPalPayment", "StripePayment", "CreditCardPayment"],
                        "violated_principle": "DIP",
                        "category": "solid"
                    },
                    {
                        "severity": "info",
                        "issue": "Missing abstraction: Three payment classes implement similar processPayment method without a common interface.",
                        "affected_entities": ["PayPalPayment", "StripePayment", "CreditCardPayment"],
                        "violated_principle": None,
                        "category": "pattern"
                    }
                ],
                "recommendations": [
                    {
                        "title": "Introduce PaymentGateway interface with Strategy pattern",
                        "description": "Create PaymentGateway interface with processPayment method. Have all payment classes implement this interface. Order should depend only on PaymentGateway, not concrete implementations.",
                        "priority": "high",
                        "affected_entities": ["Order", "PayPalPayment", "StripePayment", "CreditCardPayment"],
                        "design_pattern": "Strategy",
                        "rationale": "Enables adding new payment methods without modifying Order. Reduces coupling and follows DIP."
                    }
                ],
                "patterns_detected": [],
                "quality_score": 0.6,
                "quality_metrics": {},
                "summary": "Missing PaymentGateway abstraction creates tight coupling. Recommend Strategy pattern."
            })

            # Few-shot Example 3: SOLID violation
            example_user_3 = """Analyze this model:

Model Structure:
{
  "classes": [
    {"name": "Logger", "methods": [{"name": "log"}, {"name": "writeToFile"}], 
     "attributes": [{"name": "filePath"}]},
    {"name": "FileWriter", "methods": [{"name": "write"}]}
  ],
  "relationships": [
    {"from": "Logger", "to": "FileWriter", "type": "composition"}
  ]
}

Pre-analysis Findings:
- Logger depends on concrete implementation 'FileWriter'

Retrieved Knowledge:
- Dependency Inversion Principle: High-level modules should not depend on low-level modules. Both should depend on abstractions."""

            example_assistant_3 = json.dumps({
                "findings": [
                    {
                        "severity": "warning",
                        "issue": "Logger directly depends on concrete FileWriter class, violating Dependency Inversion Principle. This makes it impossible to log to other destinations (database, cloud) without modifying Logger.",
                        "affected_entities": ["Logger", "FileWriter"],
                        "violated_principle": "DIP",
                        "category": "solid"
                    }
                ],
                "recommendations": [
                    {
                        "title": "Introduce ILogWriter interface",
                        "description": "Create ILogWriter interface with write() method. Logger should depend on ILogWriter. FileWriter implements ILogWriter. Future writers (DatabaseWriter, CloudWriter) can implement same interface.",
                        "priority": "medium",
                        "affected_entities": ["Logger", "FileWriter"],
                        "design_pattern": "Dependency Injection",
                        "rationale": "Follows DIP, enables testing with mock writers, allows adding new log destinations without changing Logger"
                    }
                ],
                "patterns_detected": [],
                "quality_score": 0.7,
                "quality_metrics": {},
                "summary": "Logger violates DIP by depending on concrete FileWriter. Recommend introducing ILogWriter interface."
            })

            # Build user message with all context
            user_message = f"""Analyze this model:

Model Structure:
{json.dumps(model_ir, indent=2)[:1500]}

Description: {description or "No description provided"}

Pre-analysis Findings:
{json.dumps(deterministic_findings, indent=2)[:800] if deterministic_findings else "None detected"}

Metrics:
{json.dumps(metrics, indent=2)}

Retrieved Design Knowledge:
{json.dumps(retrieved_knowledge, indent=2)[:600] if retrieved_knowledge else "No additional knowledge retrieved"}"""

            # Handle mock LLM (callable that returns string)
            if callable(self.llm) and not hasattr(self.llm, 'invoke'):
                self.logger.debug(f"[{self.name}] Using mock/test LLM")
                # For testing: concatenate all prompts
                full_prompt = f"{system_prompt}\n\nExample 1:\n{example_user_1}\n{example_assistant_1}\n\nExample 2:\n{example_user_2}\n{example_assistant_2}\n\nExample 3:\n{example_user_3}\n{example_assistant_3}\n\nActual:\n{user_message}"
                text = self.llm(full_prompt)
            else:
                # Real LLM: use conversation format
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=example_user_1),
                    AIMessage(content=example_assistant_1),
                    HumanMessage(content=example_user_2),
                    AIMessage(content=example_assistant_2),
                    HumanMessage(content=example_user_3),
                    AIMessage(content=example_assistant_3),
                    HumanMessage(content=user_message)
                ]
                
                # Try multiple extraction methods
                text = None
                try:
                    response = self.llm.invoke(messages)
                    if hasattr(response, "content"):
                        text = response.content
                    elif isinstance(response, str):
                        text = response
                    else:
                        text = str(response)
                    self.logger.info(f"[{self.name}] Raw LLM response (first 500 chars): {text[:500] if text else 'None'}")
                    self.logger.debug(f"[{self.name}] Used invoke() method successfully")
                except Exception as e:
                    self.logger.warning(f"[{self.name}] LLM invoke failed: {e}, trying alternatives")
                    # Fallback to direct call
                    try:
                        maybe = self.llm(messages)
                        text = maybe.content if hasattr(maybe, "content") else str(maybe)
                        self.logger.debug(f"[{self.name}] Used direct call successfully")
                    except Exception as e2:
                        self.logger.warning(f"[{self.name}] Direct call failed: {e2}, trying generate()")
                        # Last resort: try generate
                        if hasattr(self.llm, "generate"):
                            try:
                                resp = self.llm.generate([messages])
                                gens = getattr(resp, "generations", None)
                                if gens and isinstance(gens, list) and len(gens) > 0:
                                    if isinstance(gens[0], list) and len(gens[0]) > 0:
                                        text = gens[0][0].text
                                    else:
                                        text = gens[0].text if hasattr(gens[0], 'text') else None
                                self.logger.debug(f"[{self.name}] Used generate() method")
                            except Exception as e3:
                                self.logger.error(f"[{self.name}] All extraction methods failed: {e3}")

                if not text:
                    self.logger.error(f"[{self.name}] All LLM methods failed. Response type: {type(self.llm)}")
                    raise RuntimeError("Could not extract text from LLM response")

                # Clean up response
                text = text.strip()
                if text.startswith("```json"):
                    text = text[7:]
                if text.startswith("```"):
                    text = text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()

            self.logger.info(f"[{self.name}] Cleaned LLM response length: {len(text)} chars")

            # Parse JSON
            llm_result = json.loads(text)
            
            # Step 5: Merge deterministic findings with LLM findings
            all_findings = list(deterministic_findings)
            
            # Deduplicate: hash by affected entities + issue substring
            seen = set()
            for finding in deterministic_findings:
                entities = tuple(sorted(finding.get("affected_entities", [])))
                issue_key = finding.get("issue", "")[:50]
                seen.add((entities, issue_key))
            
            # Add LLM findings that aren't duplicates
            for llm_finding in llm_result.get("findings", []):
                entities = tuple(sorted(llm_finding.get("affected_entities", [])))
                issue_key = llm_finding.get("issue", "")[:50]
                if (entities, issue_key) not in seen:
                    all_findings.append(llm_finding)
                    seen.add((entities, issue_key))

            print(f"\n[DEBUG] Merged findings (deterministic + LLM):")
            for f in all_findings:
                print(json.dumps(f, indent=2))
            
            # Step 6: Build final report
            from backend.schema import AnalysisReport
            
            # Normalize findings - map invalid category values
            for finding in all_findings:
                # Map invalid categories to valid ones
                category = finding.get("category", "other")
                valid_categories = ["solid", "pattern", "structure", "coupling", "cohesion", "other"]
                if category not in valid_categories:
                    # Try to infer the correct category
                    if "solid" in category.lower() or finding.get("violated_principle") in ["SRP", "OCP", "LSP", "ISP", "DIP"]:
                        finding["category"] = "solid"
                    elif "pattern" in category.lower() or "design" in category.lower():
                        finding["category"] = "pattern"
                    elif "coupling" in category.lower() or "dependency" in category.lower():
                        finding["category"] = "coupling"
                    elif "cohesion" in category.lower():
                        finding["category"] = "cohesion"
                    elif "structure" in category.lower():
                        finding["category"] = "structure"
                    else:
                        finding["category"] = "other"
            
            # Normalize recommendations - map invalid priority values
            recommendations = llm_result.get("recommendations", [])
            for rec in recommendations:
                # Map 'critical' to 'high', ensure valid priority
                if rec.get("priority") == "critical":
                    rec["priority"] = "high"
                elif rec.get("priority") not in ["high", "medium", "low"]:
                    rec["priority"] = "medium"  # Default to medium if invalid
            
            final_report = {
                "findings": all_findings,
                "recommendations": recommendations,
                "patterns_detected": llm_result.get("patterns_detected", []),
                "quality_score": llm_result.get("quality_score", 0.5),
                "quality_metrics": metrics,
                "summary": llm_result.get("summary", f"Analyzed {len(model_ir.get('classes', []))} classes, found {len(all_findings)} issues")
            }
            
            # Validate with Pydantic
            validated = AnalysisReport.model_validate(final_report)
            result = validated.model_dump()
            
            self.logger.info(f"[{self.name}] Analysis complete: {len(result['findings'])} findings, {len(result['recommendations'])} recommendations")
            return result
            
        except Exception as e:
            self.logger.error(f"[{self.name}] Error analyzing model: {e}")
            import traceback
            traceback.print_exc()
            return {
                "findings": [],
                "recommendations": [],
                "patterns_detected": [],
                "quality_score": 0.0,
                "quality_metrics": {},
                "summary": "",
                "error": str(e)
            }


# ============================================================================
# Code Generation Agent
# ============================================================================

class CodeGenerationAgent:
    """
    Agent responsible for generating source code from model IR with analysis-aware refactoring.
    Creates class stubs, method signatures, applies SOLID principles, and generates interfaces.
    """

    def __init__(self, llm=None, name: str = None):
        """Initialize the code generation agent.
        
        Args:
            llm: Optional LLM instance or callable for testing.
            name: Optional name for the agent (useful in tests).
        """
        self.llm = llm or create_base_llm()
        self.name = name or "code_generation_agent"
        self.logger = logger

    def _extract_refactoring_opportunities(self, analysis_report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract actionable refactoring opportunities from analysis report.
        
        Returns dict with:
        - god_classes: Classes to split
        - missing_abstractions: Interfaces to create
        - dip_violations: Dependencies to abstract
        - srp_violations: Classes with mixed responsibilities
        """
        opportunities = {
            "god_classes": [],
            "missing_abstractions": [],
            "dip_violations": [],
            "srp_violations": []
        }
        
        if not analysis_report:
            return opportunities
        
        findings = analysis_report.get("findings", [])
        
        for finding in findings:
            issue = finding.get("issue", "").lower()
            entities = finding.get("affected_entities", [])
            principle = finding.get("violated_principle")
            
            if "god class" in issue or "god object" in issue:
                opportunities["god_classes"].extend(entities)
            
            if "interface" in issue or "abstraction" in issue:
                opportunities["missing_abstractions"].append({
                    "entities": entities,
                    "issue": finding.get("issue")
                })
            
            if principle == "DIP" or "depends on concrete" in issue:
                opportunities["dip_violations"].append({
                    "entities": entities,
                    "issue": finding.get("issue")
                })
            
            if principle == "SRP" and "responsibility" in issue:
                opportunities["srp_violations"].extend(entities)
        
        # Deduplicate
        opportunities["god_classes"] = list(set(opportunities["god_classes"]))
        opportunities["srp_violations"] = list(set(opportunities["srp_violations"]))
        
        return opportunities

    def generate_code(
        self,
        model_ir: Dict[str, Any],
        language: str = "python",
        analysis_report: Dict[str, Any] = None,
        apply_refactorings: bool = True
    ) -> Dict[str, Any]:
        """
        Generate code from model IR with analysis-aware refactoring.

        Args:
            model_ir: Intermediate representation of the model.
            language: Target programming language (default: python).
            analysis_report: Full analysis report with findings and recommendations.
            apply_refactorings: Whether to automatically apply refactorings (default: True).

        Returns:
            Dictionary with generated file paths and content.
        """
        self.logger.info(f"[{self.name}] Generating {language} code (refactorings: {apply_refactorings})")

        # Extract refactoring opportunities
        refactorings = self._extract_refactoring_opportunities(analysis_report) if analysis_report else {}
        self.logger.info(f"[{self.name}] Extracted refactorings: {refactorings}")
        
        # Build system prompt  
        refactoring_instructions = ""
        if apply_refactorings and refactorings:
            refactoring_instructions = f"\nREFACTORING REQUIRED:\n{self._build_refactoring_instructions(refactorings, model_ir)}"
        
        system_prompt = f"""You are an expert Python code generator. Generate production-ready code from class models.

REQUIREMENTS:
- Follow SOLID principles (SRP, DIP, ISP)
- Use type hints and docstrings
- Apply proper separation of concerns
{refactoring_instructions}

OUTPUT: Return JSON with "files" array. Each file has "path" and "content" fields."""

        # Build user message with model and analysis context
        analysis_summary = ""
        if analysis_report:
            findings = analysis_report.get("findings", [])[:5]
            recommendations = analysis_report.get("recommendations", [])[:3]
            
            if findings:
                analysis_summary += "\nDESIGN ISSUES DETECTED:\n"
                for i, f in enumerate(findings, 1):
                    analysis_summary += f"{i}. [{f.get('severity', 'info').upper()}] {f.get('issue', '')[:100]}\n"
                    analysis_summary += f"   Affected: {', '.join(f.get('affected_entities', [])[:3])}\n"
            
            if recommendations:
                analysis_summary += "\nRECOMMENDATIONS:\n"
                for i, r in enumerate(recommendations, 1):
                    analysis_summary += f"{i}. {r.get('title', '')} (Priority: {r.get('priority', 'medium')})\n"
                    analysis_summary += f"   {r.get('description', '')[:150]}\n"

        # Escape curly braces in JSON for template
        model_json = json.dumps(model_ir, indent=2)[:1500].replace("{", "{{").replace("}", "}}")
        
        task_description = ""
        if apply_refactorings and refactorings:
            task_description = """Your task: Take the problematic model below and REFACTOR it according to the instructions.
DO NOT generate the model as-is. TRANSFORM it by splitting classes, extracting interfaces, and applying proper design patterns."""
        else:
            task_description = f"Your task: Generate {language} code implementing this model structure."
        
        user_message = f"""{task_description}

ORIGINAL MODEL STRUCTURE:
{model_json}

{analysis_summary}

{"MODE: REFACTORING ENABLED - Transform the model to fix issues" if apply_refactorings else "MODE: Direct generation - implement model as-is"}

Generate well-structured, production-ready code with proper separation of concerns."""

        self.logger.info(f"[{self.name}] System prompt length: {len(system_prompt)}, User message length: {len(user_message)}")
        self.logger.debug(f"[{self.name}] Refactoring instructions:\n{self._build_refactoring_instructions(refactorings, model_ir)}")

        try:
            # Handle mock/callable LLM
            if callable(self.llm) and not hasattr(self.llm, 'invoke'):
                full_prompt = f"{system_prompt}\n\n{user_message}"
                text = self.llm(full_prompt)
                result = json.loads(text)
            else:
                # Real LLM path
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("user", user_message)
                ])
                
                # First get raw LLM output without parser
                chain_without_parser = prompt | self.llm
                raw_response = chain_without_parser.invoke({})
                
                # Log the raw output for debugging
                raw_text = raw_response.content if hasattr(raw_response, 'content') else str(raw_response)
                self.logger.info(f"[{self.name}] Raw LLM output length: {len(raw_text)} chars")
                self.logger.debug(f"[{self.name}] Full raw output:\n{raw_text}")
                
                # Now parse it
                parser = JsonOutputParser()
                try:
                    result = parser.parse(raw_text)
                except Exception as parse_error:
                    self.logger.error(f"[{self.name}] JSON parsing failed. Raw output (first 1000 chars):\n{raw_text[:1000]}")
                    self.logger.error(f"[{self.name}] Raw output (last 500 chars):\n{raw_text[-500:]}")
                    raise parse_error
            
            self.logger.info(f"[{self.name}] Generated {len(result.get('files', []))} files")
            return result
            
        except Exception as e:
            self.logger.error(f"[{self.name}] Error generating code: {e}")
            
            # Check for rate limit errors
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in ['rate limit', 'quota', 'resource_exhausted', '429']):
                self.logger.error(f"[{self.name}] RATE LIMIT ERROR DETECTED")
            
            import traceback
            traceback.print_exc()
            return {
                "files": [],
                "language": language,
                "error": str(e)
            }

    def _build_refactoring_instructions(self, refactorings: Dict[str, Any], model_ir: Dict[str, Any]) -> str:
        """Build specific refactoring instructions based on detected issues."""
        instructions = []
        
        if refactorings.get("god_classes"):
            god_classes = refactorings["god_classes"]
            
            # Build detailed splitting instructions for each god class
            for god_class in god_classes:
                # Find the class in model_ir to analyze its methods
                class_info = None
                for cls in model_ir.get("classes", []):
                    if cls.get("name") == god_class:
                        class_info = cls
                        break
                
                if class_info:
                    methods = class_info.get("methods", [])
                    method_names = [m.get('name') for m in methods]
                    
                    # Categorize methods by responsibility
                    data_methods = [m for m in method_names if any(word in m.lower() for word in ["save", "load", "database", "query", "fetch", "store"])]
                    comm_methods = [m for m in method_names if any(word in m.lower() for word in ["send", "email", "notify", "message"])]
                    log_methods = [m for m in method_names if any(word in m.lower() for word in ["log", "audit", "track"])]
                    business_methods = [m for m in method_names if m not in data_methods and m not in comm_methods and m not in log_methods]
                    
                    service_name = god_class.replace('Manager', 'Service')
                    repo_name = god_class.replace('Manager', 'Repository')
                    
                    parts = []
                    parts.append(f"1. {service_name} - Business logic: {', '.join(business_methods) if business_methods else 'orchestration'}")
                    parts.append(f"2. {repo_name} - Data access: {', '.join(data_methods) if data_methods else 'database operations'}")
                    
                    if comm_methods:
                        parts.append(f"3. EmailService - Communication: {', '.join(comm_methods)}")
                    if log_methods:
                        parts.append(f"4. LogService - Logging: {', '.join(log_methods)}")
                    
                    instructions.append(f"""
- GOD CLASS: {god_class} with {len(methods)} methods
  
  REQUIRED: Split into {len(parts)} separate files:
  {chr(10).join(parts)}
  
  Generate each as a SEPARATE file with dependency injection.""")
                else:
                    instructions.append(f"""
- GOD CLASS: {god_class}
  Split into Repository, Service, and supporting classes in separate files.""")
        
        if refactorings.get("missing_abstractions"):
            for abstraction in refactorings["missing_abstractions"][:3]:
                entities = abstraction.get("entities", [])
                if len(entities) >= 2:
                    instructions.append(f"""
- MISSING ABSTRACTION: {', '.join(entities)} implement similar behavior
  Action: Create a common interface/abstract base class.
  Example: Create I{entities[0].replace('Processor', '').replace('Handler', '')}Interface
  Make all implementations inherit from this interface.
  Use dependency injection to depend on the interface, not concrete classes.""")
        
        if refactorings.get("dip_violations"):
            for violation in refactorings["dip_violations"][:3]:
                entities = violation.get("entities", [])
                issue = violation.get("issue", "")
                if "mysql" in issue.lower() or "database" in issue.lower():
                    instructions.append(f"""
- DIP VIOLATION: Direct database coupling in {', '.join(entities)}
  Action: Create an IRepository interface for data access.
  Implement MySQLRepository, PostgresRepository, InMemoryRepository.
  Use dependency injection - classes depend on IRepository, not concrete implementations.""")
                elif entities:
                    instructions.append(f"""
- DIP VIOLATION: {', '.join(entities)} depends on concrete implementations
  Action: Introduce abstraction layer.
  Create interface/protocol for the dependency.
  Use dependency injection to inject the concrete implementation.""")
        
        if not instructions:
            instructions.append("- No critical refactorings needed. Generate clean, SOLID-compliant code.")
        
        return "\n".join(instructions)


# ============================================================================
# Test Generation Agent
# ============================================================================

class TestGenerationAgent:
    """
    Agent responsible for generating test cases and test suites
    based on the generated code and model specification.
    
    This agent generates analysis-aware tests that:
    - Verify refactoring correctness (god class splitting, interface extraction)
    - Test cohesion (classes have focused responsibilities)
    - Test coupling (dependency injection working correctly)
    - Validate parameter handling (especially for long parameter lists)
    - Test interface implementations (missing abstractions)
    """

    def __init__(self, llm: Optional[Any] = None):
        """Initialize the test generation agent."""
        self.llm = llm if llm is not None else create_base_llm()
        self.name = "test_generation_agent"
        self.logger = logger

    def generate_tests(
        self,
        model_ir: Dict[str, Any],
        generated_code: Dict[str, Any],
        analysis_report: Optional[Dict[str, Any]] = None,
        framework: str = "pytest",
        include_integration_tests: bool = True
    ) -> Dict[str, Any]:
        """
        Generate comprehensive test cases with analysis-aware test generation.

        Args:
            model_ir: Intermediate representation of the model.
            generated_code: Dictionary with 'files' list containing generated code.
            analysis_report: Optional analysis report with findings/recommendations.
            framework: Testing framework to use (default: pytest).
            include_integration_tests: Whether to generate integration tests.

        Returns:
            Dictionary with:
            - test_files: List of test file dicts with 'path' and 'content'
            - framework: Testing framework used
            - total_tests: Estimated number of tests
            - test_categories: List of test categories generated
        """
        self.logger.info(f"[{self.name}] Generating {framework} test cases")

        # Extract test generation strategy from analysis
        test_strategy = self._extract_test_strategy(analysis_report) if analysis_report else {}
        
        # Build the prompt with analysis-aware instructions
        system_prompt = self._build_system_prompt(framework, test_strategy)
        user_message = self._build_user_message(
            model_ir, 
            generated_code, 
            analysis_report,
            test_strategy,
            include_integration_tests
        )

        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("user", user_message)
            ])
            
            # Log prompt details for debugging
            self.logger.info(f"[{self.name}] System prompt length: {len(system_prompt)} chars")
            self.logger.info(f"[{self.name}] User message length: {len(user_message)} chars")
            self.logger.debug(f"[{self.name}] System prompt preview (first 300 chars): {system_prompt[:300]}")
            self.logger.debug(f"[{self.name}] User message preview (first 500 chars): {user_message[:500]}")
            
            parser = JsonOutputParser()
            
            # Handle both Runnable LLMs and simple callables (for testing)
            if hasattr(self.llm, '__or__'):  # Runnable with pipe support
                # For real LLMs, invoke without parser first to see raw output
                self.logger.info(f"[{self.name}] Invoking LLM with prompt...")
                chain_without_parser = prompt | self.llm
                llm_output = chain_without_parser.invoke({})
                self.logger.info(f"[{self.name}] LLM invocation completed")
                
                # Log raw output for debugging
                if hasattr(llm_output, 'content'):
                    raw_text = llm_output.content
                else:
                    raw_text = str(llm_output)
                
                self.logger.info(f"[{self.name}] Raw LLM output (first 500 chars): {raw_text[:500]}")
                self.logger.info(f"[{self.name}] Raw LLM output length: {len(raw_text)} chars")
                
                # Strip markdown code blocks if present
                cleaned_text = raw_text.strip()
                if cleaned_text.startswith("```json"):
                    cleaned_text = cleaned_text[7:]
                elif cleaned_text.startswith("```"):
                    cleaned_text = cleaned_text[3:]
                if cleaned_text.endswith("```"):
                    cleaned_text = cleaned_text[:-3]
                cleaned_text = cleaned_text.strip()
                
                if cleaned_text != raw_text:
                    self.logger.info(f"[{self.name}] Stripped markdown blocks, new length: {len(cleaned_text)} chars")
                
                # Now parse the output
                try:
                    result = parser.parse(cleaned_text)
                except Exception as parse_error:
                    self.logger.error(f"[{self.name}] JSON parsing failed. Raw output: {raw_text[:1000]}")
                    raise
            else:  # Simple callable (for testing with mocks)
                # Avoid using ChatPromptTemplate.format_messages() for simple
                # callables to prevent issues with literal braces in JSON.
                concatenated = system_prompt + "\n\n" + user_message
                if hasattr(self.llm, 'invoke'):
                    llm_output = self.llm.invoke(concatenated)
                elif callable(self.llm):
                    llm_output = self.llm(concatenated)
                else:
                    raise AttributeError("LLM object must provide 'invoke' or be callable")
                # Parse JSON from string if needed
                if isinstance(llm_output, str):
                    result = json.loads(llm_output)
                else:
                    result = llm_output
            
            # Validate and enrich result
            test_files = result.get('test_files', [])
            if not test_files:
                self.logger.warning(f"[{self.name}] No test files generated")
            
            # Count estimated tests (rough heuristic: count 'def test_' in content)
            total_tests = sum(
                file.get('content', '').count('def test_') 
                for file in test_files
            )
            
            enriched_result = {
                "test_files": test_files,
                "framework": framework,
                "total_tests": total_tests,
                "test_categories": test_strategy.get('categories', []),
                "analysis_aware": bool(analysis_report)
            }
            
            self.logger.info(
                f"[{self.name}] Generated {len(test_files)} test files "
                f"with ~{total_tests} test cases"
            )
            return enriched_result
            
        except Exception as e:
            self.logger.error(f"[{self.name}] Error generating tests: {e}", exc_info=True)
            return {
                "test_files": [],
                "framework": framework,
                "total_tests": 0,
                "test_categories": [],
                "error": str(e)
            }

    def _extract_test_strategy(self, analysis_report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract test generation strategy from analysis report.
        
        Returns dict with:
        - categories: List of test categories to generate
        - focus_areas: Specific areas needing test coverage
        - god_classes: Classes that were split (test separation)
        - missing_abstractions: Interfaces to test
        - dip_violations: Dependencies to verify
        """
        strategy = {
            "categories": ["unit", "integration"],
            "focus_areas": [],
            "god_classes": [],
            "missing_abstractions": [],
            "dip_violations": []
        }
        
        findings = analysis_report.get("findings", [])
        recommendations = analysis_report.get("recommendations", [])
        
        for finding in findings:
            issue = finding.get("issue", "").lower()
            category = finding.get("category", "")
            violated_principle = finding.get("violated_principle", "")
            entities = finding.get("affected_entities", [])
            
            # God class splitting -> test cohesion
            if "god" in issue or violated_principle == "SRP":
                strategy["god_classes"].extend(entities)
                strategy["focus_areas"].append("cohesion")
                strategy["categories"].append("cohesion")
            
            # Missing abstractions -> test interface implementations
            if category == "pattern" and "abstraction" in issue.lower():
                strategy["missing_abstractions"].extend(entities)
                strategy["focus_areas"].append("abstraction")
                strategy["categories"].append("interface")
            
            # DIP violations -> test dependency injection
            if violated_principle == "DIP":
                strategy["dip_violations"].extend(entities)
                strategy["focus_areas"].append("coupling")
                strategy["categories"].append("dependency_injection")
        
        # Deduplicate and remove empty
        strategy["categories"] = list(set(strategy["categories"]))
        strategy["focus_areas"] = list(set(strategy["focus_areas"]))
        
        return strategy

    def _build_system_prompt(self, framework: str, test_strategy: Dict[str, Any]) -> str:
        """Build system prompt with test generation instructions."""
        focus_areas = test_strategy.get("focus_areas", [])
        
        prompt = f"""You are an expert {framework} test developer specializing in design quality testing.

Generate comprehensive, executable test cases following these principles:

CRITICAL REQUIREMENTS:
1. All tests must be valid, executable {framework} code
2. Use proper imports (pytest, unittest.mock, etc.)
3. Include clear docstrings explaining what each test validates
4. Mock external dependencies appropriately
5. Test both happy paths and edge cases
6. IMPORTANT: Return ONLY valid JSON with this EXACT structure (no extra text):
{{{{
  "test_files": [
    {{{{"path": "tests/test_example.py", "content": "# Complete test file content"}}}}
  ]
}}}}

DO NOT include markdown code blocks (```json) or any text outside the JSON structure.

TEST CATEGORIES TO GENERATE:"""
        categories = test_strategy.get("categories", ["unit", "integration"])
        
        if "unit" in categories:
            prompt += """
- UNIT TESTS: Test individual methods and classes in isolation
  * Test all public methods
  * Test error conditions and exceptions
  * Use mocks for dependencies
  * Aim for 80%+ coverage"""
        
        if "cohesion" in categories:
            prompt += """
- COHESION TESTS: Verify classes have focused, single responsibilities
  * Test that refactored classes handle only related operations
  * Verify separation of concerns (e.g., business logic vs data access)
  * Check that classes don't call unrelated methods"""
        
        if "dependency_injection" in categories:
            prompt += """
- DEPENDENCY INJECTION TESTS: Verify proper decoupling
  * Test that classes accept dependencies via constructor/parameters
  * Mock injected dependencies
  * Verify no hardcoded concrete implementations
  * Test with different dependency implementations"""
        
        if "interface" in categories:
            prompt += """
- INTERFACE TESTS: Validate interface implementations
  * Test all interface method implementations
  * Verify multiple implementations work with same interface
  * Test polymorphic behavior"""
        
        if "integration" in categories:
            prompt += """
- INTEGRATION TESTS: Test component interactions
  * Test workflows across multiple classes
  * Verify data flow between components
  * Test error propagation"""

        if focus_areas:
            prompt += f"""

FOCUS AREAS (prioritize these in tests):
{', '.join(focus_areas).upper()}"""

        prompt += """

CODE QUALITY REQUIREMENTS:
- Use descriptive test names: test_<what>_<condition>_<expected>
- Group related tests in classes
- Use pytest fixtures for common setup
- Add parametrize for multiple test cases
- Include assertions with clear failure messages"""

        return prompt

    def _build_user_message(
        self,
        model_ir: Dict[str, Any],
        generated_code: Dict[str, Any],
        analysis_report: Optional[Dict[str, Any]],
        test_strategy: Dict[str, Any],
        include_integration_tests: bool
    ) -> str:
        """Build user message with context for test generation."""
        
        message = "Generate comprehensive test cases for the following code:\n\n"
        
        # Add model context
        classes = model_ir.get("classes", [])
        message += f"MODEL CONTEXT:\n"
        message += f"- {len(classes)} classes: {', '.join(c.get('name', 'Unknown') for c in classes[:5])}\n"
        if len(classes) > 5:
            message += f"  ... and {len(classes) - 5} more\n"
        
        # Add generated code
        files = generated_code.get("files", [])
        message += f"\nGENERATED CODE FILES ({len(files)} files):\n"
        
        for file_info in files[:5]:  # Limit to first 5 files to avoid token limits
            path = file_info.get("path", "unknown")
            content = file_info.get("content", "")
            # Include first 1500 chars of each file
            truncated_content = content[:1500]
            if len(content) > 1500:
                truncated_content += "\n... (content truncated)"
            # Escape curly braces for ChatPromptTemplate
            truncated_content = truncated_content.replace("{", "{{").replace("}", "}}")
            message += f"\n--- {path} ---\n{truncated_content}\n"
        
        if len(files) > 5:
            message += f"\n... and {len(files) - 5} more files (not shown to save space)\n"
        
        # Add analysis-aware instructions
        if analysis_report:
            findings = analysis_report.get("findings", [])
            recommendations = analysis_report.get("recommendations", [])
            
            message += f"\nANALYSIS REPORT:\n"
            message += f"- {len(findings)} design issues detected\n"
            message += f"- {len(recommendations)} recommendations provided\n"
            
            # Highlight key findings for test generation
            if test_strategy.get("god_classes"):
                message += f"\nGOD CLASSES THAT WERE SPLIT (test cohesion):\n"
                for cls in test_strategy["god_classes"][:3]:
                    message += f"- {cls}\n"
            
            if test_strategy.get("missing_abstractions"):
                message += f"\nINTERFACES CREATED (test implementations):\n"
                for abstraction in test_strategy["missing_abstractions"][:3]:
                    message += f"- {abstraction}\n"
            
            if test_strategy.get("dip_violations"):
                message += f"\nDEPENDENCIES REFACTORED (test injection):\n"
                for dep in test_strategy["dip_violations"][:3]:
                    message += f"- {dep}\n"
        
        # Add test requirements
        message += f"\nTEST REQUIREMENTS:\n"
        message += f"- Framework: pytest\n"
        message += f"- Include integration tests: {include_integration_tests}\n"
        message += f"- Test categories: {', '.join(test_strategy.get('categories', ['unit']))}\n"
        
        message += """

Generate tests that:
1. Validate the refactored code works correctly
2. Verify design improvements (cohesion, decoupling, abstractions)
3. Cover edge cases and error conditions
4. Are executable and follow pytest best practices

Return JSON with "test_files" list containing dicts with "path" and "content" keys."""
        
        return message


# ============================================================================
# Critic / Refactoring Agent
# ============================================================================

class CriticAgent:
    """
    Agent responsible for reviewing generated code and analyses,
    identifying improvement opportunities, and proposing refactorings.
    """

    def __init__(self):
        """Initialize the critic agent."""
        self.llm = create_base_llm()
        self.name = "critic_agent"
        self.logger = logger

    def critique(
        self,
        analysis_report: Dict[str, Any],
        code_files: Dict[str, Any],
        test_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Provide critique and refactoring suggestions using Gemini.

        Args:
            analysis_report: Output from the analysis agent.
            code_files: Generated code files.
            test_results: Results from test execution.

        Returns:
            Dictionary with proposed improvements and refactorings.
        """
        self.logger.info(f"[{self.name}] Critiquing generated artifacts")

        system_prompt = """You are an expert code reviewer and architect.
Review the provided model analysis, generated code, and test results.

Provide a detailed critique including:
- Code quality issues (maintainability, readability, efficiency)
- Design improvements based on SOLID principles
- Refactoring suggestions with specific examples
- Missing features or error handling
- Security considerations
- Performance optimizations

Return a JSON object with: issues, refactoring_suggestions, quality_score (0-100), and reasoning."""

        user_message = f"""Review these artifacts:

Analysis Report:
{json.dumps(analysis_report, indent=2)[:500]}

Code Files:
{json.dumps(code_files, indent=2)[:500]}

Test Results:
{json.dumps(test_results, indent=2)[:500]}

Provide comprehensive critique and suggestions."""

        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("user", user_message)
            ])

            parser = JsonOutputParser()
            # Support both runnable LLMs and simple callables for testing
            if hasattr(self.llm, '__or__'):
                chain_without_parser = prompt | self.llm
                llm_output = chain_without_parser.invoke({})
                if hasattr(llm_output, 'content'):
                    raw_text = llm_output.content
                else:
                    raw_text = str(llm_output)
                self.logger.info(f"[{self.name}] Raw LLM output: {raw_text[:500]}")
                try:
                    result = parser.parse(raw_text)
                except Exception as parse_err:
                    self.logger.error(f"[{self.name}] JSON parsing failed for critique. Raw output: {raw_text[:1000]}")
                    raise
            else:
                # Avoid prompt.format_messages() for simple callables because
                # they use Python format-style braces which may conflict with
                # literal JSON in the message. Instead, concatenate the system
                # and user messages and pass a single string to the callable.
                concatenated = system_prompt + "\n\n" + user_message
                llm_output = self.llm.invoke(concatenated)
                if isinstance(llm_output, str):
                    result = json.loads(llm_output)
                else:
                    result = llm_output
            self.logger.info(f"[{self.name}] Critique complete - quality score: {result.get('quality_score', 0)}")
            return result
            
        except Exception as e:
            self.logger.error(f"[{self.name}] Error during critique: {e}")
            return {
                "issues": [],
                "refactoring_suggestions": [],
                "quality_score": 0.0,
                "error": str(e)
            }


# ============================================================================
# Orchestrator / Coordinator
# ============================================================================

class OrchestratorAgent:
    """
    Coordinates the workflow, routing between agents and managing state.
    """

    def __init__(self):
        """Initialize the orchestrator agent."""
        self.llm = create_base_llm()
        self.logger = logger
        self.name = "orchestrator_agent"
        
        # Initialize agents
        from backend.knowledge_base import get_knowledge_base
        kb = get_knowledge_base()
        
        self.parser = ParserAgent()
        self.analyzer = AnalysisAgent(retriever=kb.get_simple_retriever())
        self.codegen = CodeGenerationAgent()
        self.testgen = TestGenerationAgent()
        self.critic = CriticAgent()

    def orchestrate(
        self,
        model_text: str,
        model_format: str = "plantuml",
        description: str = ""
    ) -> Dict[str, Any]:
        """
        Orchestrate the full workflow for analyzing and generating code.

        Args:
            model_text: Raw model text.
            model_format: Format of the model.
            description: Natural language description.

        Returns:
            Dictionary containing final results from all agents.
        """
        self.logger.info("Starting orchestration workflow")
        
        # Parse model
        model_ir = self.parser.parse_model(model_text, model_format)
        if not model_ir.get("classes"):
            return {
                "status": "error",
                "message": "Failed to parse model",
                "error": model_ir.get("error", "Unknown error")
            }
        
        # Analyze model
        analysis = self.analyzer.analyze_model(model_ir, description)
        
        # Generate code
        code = self.codegen.generate_code(model_ir, "python", analysis)
        
        # Generate tests
        tests = self.testgen.generate_tests(model_ir, code, "pytest")
        
        # Critique
        critique = self.critic.critique(analysis, code, {})
        
        self.logger.info("Orchestration workflow completed")
        
        return {
            "status": "success",
            "model_ir": model_ir,
            "analysis": analysis,
            "code": code,
            "tests": tests,
            "critique": critique
        }
