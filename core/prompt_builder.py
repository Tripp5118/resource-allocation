# core/prompt_builders.py
"""Prompt builders for BOAgent with different instruction styles."""

from abc import ABC, abstractmethod
from typing import Optional


class PromptBuilder(ABC):
    """Base class for building agent prompts."""
    
    def __init__(
        self,
        include_uncertainty: bool = False,
        include_hypervolume: bool = False,
    ):
        """
        Args:
            include_uncertainty: Whether to include uncertainty metrics
            include_hypervolume: Whether to include hypervolume metrics
        """
        self.include_uncertainty = include_uncertainty
        self.include_hypervolume = include_hypervolume
    
    @abstractmethod
    def build_decision_framework(self) -> str:
        """Build the decision framework instructions for the agent."""
        pass
    
    def build_system_message(
        self,
        problem_description: str,
        obj1_name: str,
        obj2_name: str,
        iteration: int,
        budget_remaining: float,
        time_remaining: float,
        time_per_point: float,
        global_context: str,
        event_warning: str,
    ) -> str:
        """Build the system message for the agent."""
        return f"""You are an expert AI managing a Bayesian Optimization campaign.

Problem:
{problem_description or '(no additional description provided)'}

Objectives:
- {obj1_name}
- {obj2_name}

Current Status:
- Iteration: {iteration}
- Budget: ${budget_remaining:.2f}
- Time: {time_remaining:.1f} weeks

Important: Each iteration takes {time_per_point:.1f} week(s) regardless of batch size (parallel execution).

{global_context}{event_warning}

When ready to decide, respond:
SELECTED_OPTION: <0-5>
REASONING: <brief justification>"""
    
    def build_initial_message(
        self,
        analysis: str,
        options_desc: str,
        max_reasoning_steps: int,
    ) -> str:
        """Build the initial message prompting the agent to make a decision."""
        decision_framework = self.build_decision_framework()
        
        base_message = f"""Make a resource allocation decision for this iteration.

{analysis}

{options_desc}"""
        
        if decision_framework:
            base_message += f"\n\n{decision_framework}"
        
        base_message += f"""

You have up to {max_reasoning_steps} reasoning steps.

When ready:
SELECTED_OPTION: <0-5>
REASONING: <justification>

Begin analysis."""
        
        return base_message
    
    def enhance_analysis(
        self,
        base_analysis: str,
        uncertainty_obj1: Optional[float] = None,
        uncertainty_obj2: Optional[float] = None,
        current_hypervolume: Optional[float] = None,
        obj1_name: str = "Objective 1",
        obj2_name: str = "Objective 2",
    ) -> str:
        """Add optional uncertainty and hypervolume information to analysis."""
        enhanced = base_analysis
        
        if self.include_uncertainty and uncertainty_obj1 is not None:
            enhanced += f"""

**4. Knowledge Uncertainty**:
   - {obj1_name} uncertainty: {uncertainty_obj1:.2f}
   - {obj2_name} uncertainty: {uncertainty_obj2:.2f}"""
        
        if self.include_hypervolume and current_hypervolume is not None:
            enhanced += f"""

**5. Multi-Objective Progress**:
   - Current hypervolume: {current_hypervolume:.2f}"""
        
        return enhanced
    
    def enhance_options_description(self, options_desc: str) -> str:
        """Add clarifications about acquisition metrics."""
        # Add note about EHVI and MI not being directly comparable
        clarification = "\n**Note**: EHVI (Expected Hypervolume Improvement) and Entropy (Mutual Information) measure different aspects and are not directly comparable. EHVI focuses on optimization potential, while Entropy measures information gain.\n"
        
        # Insert after the header
        lines = options_desc.split('\n')
        if lines and lines[0].startswith('## Allocation Options'):
            lines.insert(1, clarification)
        else:
            lines.insert(0, clarification)
        
        return '\n'.join(lines)


class MinimalPromptBuilder(PromptBuilder):
    """Minimal prompt with no decision guidelines."""
    
    def build_decision_framework(self) -> str:
        """No decision framework - agent must decide on its own."""
        return ""


class DefaultPromptBuilder(PromptBuilder):
    """Default prompt with basic decision guidelines."""
    
    def build_decision_framework(self) -> str:
        """Provide basic guidance on decision-making."""
        return """Decision framework:
- The allocation options show EHVI (Expected Hypervolume Improvement) and Entropy (Mutual Information)
- High EHVI options focus on optimization (improving known good regions)
- High Entropy options focus on exploration (learning about uncertain regions)
- Consider your runway: if running low on resources, prioritize optimization
- Consider recent performance: if plateauing, exploration may help; if improving, continue optimizing"""


class SimpleGuidelinePromptBuilder(PromptBuilder):
    """Simple guideline prompt with minimal direction."""
    
    def build_decision_framework(self) -> str:
        """Provide simple guideline on decision-making."""
        return """Decision framework (guideline):
- If recent progress is improving with enough runway → prefer more exploitation
- If progress is plateauing or runway is short → prefer more exploration or balanced"""


# Factory function for easy creation
def create_prompt_builder(
    style: str = "default",
    include_uncertainty: bool = False,
    include_hypervolume: bool = False,
) -> PromptBuilder:
    """
    Factory function to create prompt builders.
    
    Args:
        style: One of "minimal", "default", "simple_guideline"
        include_uncertainty: Whether to include uncertainty metrics
        include_hypervolume: Whether to include hypervolume metrics
    
    Returns:
        PromptBuilder instance
    """
    builders = {
        "minimal": MinimalPromptBuilder,
        "default": DefaultPromptBuilder,
        "simple_guideline": SimpleGuidelinePromptBuilder,
    }
    
    if style not in builders:
        raise ValueError(f"Unknown prompt style: {style}. Choose from {list(builders.keys())}")
    
    return builders[style](
        include_uncertainty=include_uncertainty,
        include_hypervolume=include_hypervolume,
    )