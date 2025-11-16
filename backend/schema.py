from pydantic import BaseModel, Field, model_validator
from typing import List, Literal, Dict, Optional, Any


class Attribute(BaseModel):
    name: str
    type: str


class Method(BaseModel):
    name: str
    params: List[str] = Field(default_factory=list)
    returns: str = ""


class ClassIR(BaseModel):
    name: str
    attributes: List[Attribute] = Field(default_factory=list)
    methods: List[Method] = Field(default_factory=list)
    description: str = ""


class Relationship(BaseModel):
    from_: str = Field(..., alias="from")
    to: str
    type: str
    multiplicity: str


class ModelIR(BaseModel):
    classes: List[ClassIR] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


# ============================================================================
# Analysis Agent Output Schemas
# ============================================================================

class AnalysisFinding(BaseModel):
    """Represents a design issue or concern found in the model."""
    
    severity: Literal["critical", "warning", "info"] = Field(
        description="Severity level of the finding"
    )
    issue: str = Field(
        description="Description of the design issue"
    )
    affected_entities: List[str] = Field(
        default_factory=list,
        description="Classes, methods, or relationships affected by this issue"
    )
    violated_principle: Optional[str] = Field(
        None,
        description="Design principle violated (e.g., 'SRP', 'DIP', 'DRY')"
    )
    category: Literal["solid", "pattern", "structure", "coupling", "cohesion", "other"] = Field(
        default="other",
        description="Category of the issue"
    )


class AnalysisRecommendation(BaseModel):
    """Represents a recommended improvement or refactoring."""
    
    title: str = Field(
        description="Short title for the recommendation"
    )
    description: str = Field(
        description="Detailed description of what to do"
    )
    priority: Literal["high", "medium", "low"] = Field(
        description="Priority level for implementing this recommendation"
    )
    affected_entities: List[str] = Field(
        default_factory=list,
        description="Classes or components that would be changed"
    )
    design_pattern: Optional[str] = Field(
        None,
        description="Design pattern to apply (e.g., 'Factory', 'Strategy', 'Observer')"
    )
    rationale: str = Field(
        default="",
        description="Why this recommendation improves the design"
    )


class AnalysisReport(BaseModel):
    """Complete analysis report for a model."""
    
    findings: List[AnalysisFinding] = Field(
        default_factory=list,
        description="List of design issues found"
    )
    recommendations: List[AnalysisRecommendation] = Field(
        default_factory=list,
        description="List of recommended improvements"
    )
    patterns_detected: List[str] = Field(
        default_factory=list,
        description="Design patterns detected in the model"
    )
    quality_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overall quality score from 0.0 (poor) to 1.0 (excellent)"
    )
    quality_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Computed metrics (e.g., avg_methods_per_class, max_methods_per_class)"
    )
    summary: str = Field(
        default="",
        description="High-level summary of the analysis"
    )
    
    @model_validator(mode='after')
    def validate_critical_findings_have_recommendations(self) -> 'AnalysisReport':
        """Ensure critical findings have corresponding recommendations."""
        critical_entities = set()
        for finding in self.findings:
            if finding.severity == "critical":
                critical_entities.update(finding.affected_entities)
        
        # Check if recommendations cover critical entities
        if critical_entities:
            recommended_entities = set()
            for rec in self.recommendations:
                recommended_entities.update(rec.affected_entities)
            
            # If critical findings exist but no high-priority recommendations, add a warning
            high_priority_recs = [r for r in self.recommendations if r.priority == "high"]
            if not high_priority_recs and len(self.findings) > 0:
                # Add a generic recommendation if none exist
                if not self.recommendations:
                    self.recommendations.append(
                        AnalysisRecommendation(
                            title="Address critical issues",
                            description="Review and address the critical findings identified in the analysis",
                            priority="high",
                            affected_entities=list(critical_entities),
                            rationale="Critical issues require immediate attention to improve design quality"
                        )
                    )
        
        return self
