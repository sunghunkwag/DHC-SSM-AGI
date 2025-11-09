"""
Dynamic Goal System

Implements mechanisms for the system to question, redefine, and evolve
its own goals based on experience and environmental changes.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class GoalType(Enum):
    """Types of goals the system can have."""
    PERFORMANCE = "performance"
    EXPLORATION = "exploration"
    SAFETY = "safety"
    EFFICIENCY = "efficiency"
    GENERALIZATION = "generalization"
    NOVELTY = "novelty"
    META_LEARNING = "meta_learning"


@dataclass
class Goal:
    """
    Represents a system goal with priority and satisfaction level.
    """
    goal_type: GoalType
    description: str
    priority: float  # 0.0 to 1.0
    satisfaction: float  # 0.0 to 1.0
    is_fundamental: bool = False
    parent_goal: Optional['Goal'] = None
    sub_goals: List['Goal'] = None
    
    def __post_init__(self):
        if self.sub_goals is None:
            self.sub_goals = []
    
    def __repr__(self):
        return (f"Goal({self.goal_type.value}, "
                f"priority={self.priority:.2f}, "
                f"satisfaction={self.satisfaction:.2f})")


class GoalQuestioner(nn.Module):
    """
    Questions whether current goals are appropriate and fundamental enough.
    
    This component enables the system to critically evaluate its own objectives.
    """
    
    def __init__(self, goal_dim: int = 64, context_dim: int = 128):
        super().__init__()
        
        # Goal evaluation network
        self.evaluator = nn.Sequential(
            nn.Linear(goal_dim + context_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 4),  # [appropriateness, fundamentality, alignment, obsolescence]
        )
        
        # Criticality analyzer
        self.criticality_analyzer = nn.Sequential(
            nn.Linear(goal_dim + context_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        goal_embedding: torch.Tensor,
        context_embedding: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Question and evaluate a goal.
        
        Args:
            goal_embedding: Embedding of the goal
            context_embedding: Current context (environment, performance, etc.)
            
        Returns:
            Dictionary containing evaluation scores
        """
        combined = torch.cat([goal_embedding, context_embedding], dim=-1)
        
        # Evaluate goal
        eval_scores = torch.sigmoid(self.evaluator(combined))
        
        # Analyze criticality
        criticality = self.criticality_analyzer(combined)
        
        return {
            'appropriateness': eval_scores[0],
            'fundamentality': eval_scores[1],
            'alignment': eval_scores[2],
            'obsolescence': eval_scores[3],
            'criticality': criticality.squeeze(),
            'should_reconsider': eval_scores[3] > 0.7 or eval_scores[1] < 0.3,
        }


class GoalGenerator(nn.Module):
    """
    Generates new goals based on experience and higher-level objectives.
    """
    
    def __init__(
        self,
        context_dim: int = 128,
        goal_dim: int = 64,
        num_goal_types: int = 7,
    ):
        super().__init__()
        self.num_goal_types = num_goal_types
        self.goal_dim = goal_dim
        
        # Goal generation network
        self.generator = nn.Sequential(
            nn.Linear(context_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, num_goal_types * (goal_dim + 2)),  # embedding + priority + fundamentality
        )
    
    def forward(
        self,
        context: torch.Tensor,
        top_k: int = 3,
    ) -> List[Tuple[torch.Tensor, float, float, GoalType]]:
        """
        Generate new goals.
        
        Args:
            context: Current context embedding
            top_k: Number of goals to generate
            
        Returns:
            List of (goal_embedding, priority, fundamentality, goal_type)
        """
        output = self.generator(context)
        output = output.view(self.num_goal_types, self.goal_dim + 2)
        
        goal_embeddings = output[:, :self.goal_dim]
        priorities = torch.sigmoid(output[:, self.goal_dim])
        fundamentalities = torch.sigmoid(output[:, self.goal_dim + 1])
        
        # Select top-k by priority
        top_indices = torch.topk(priorities, k=min(top_k, self.num_goal_types)).indices
        
        goal_types = list(GoalType)
        generated_goals = []
        
        for idx in top_indices:
            idx = idx.item()
            generated_goals.append((
                goal_embeddings[idx],
                priorities[idx].item(),
                fundamentalities[idx].item(),
                goal_types[idx],
            ))
        
        return generated_goals


class GoalHierarchyManager(nn.Module):
    """
    Manages hierarchical relationships between goals.
    
    Higher-level goals can spawn sub-goals, and goal satisfaction
    propagates through the hierarchy.
    """
    
    def __init__(self, goal_dim: int = 64):
        super().__init__()
        
        # Hierarchy analyzer
        self.hierarchy_analyzer = nn.Sequential(
            nn.Linear(goal_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
    
    def compute_parent_child_score(
        self,
        parent_embedding: torch.Tensor,
        child_embedding: torch.Tensor,
    ) -> float:
        """
        Compute how well a child goal serves a parent goal.
        
        Args:
            parent_embedding: Parent goal embedding
            child_embedding: Child goal embedding
            
        Returns:
            Alignment score (0.0 to 1.0)
        """
        combined = torch.cat([parent_embedding, child_embedding], dim=-1)
        score = self.hierarchy_analyzer(combined)
        return score.item()
    
    def propagate_satisfaction(
        self,
        goal_hierarchy: Dict[str, Goal],
    ) -> Dict[str, float]:
        """
        Propagate satisfaction scores through goal hierarchy.
        
        Parent goal satisfaction is influenced by child goal satisfaction.
        
        Args:
            goal_hierarchy: Dictionary of goals
            
        Returns:
            Updated satisfaction scores
        """
        updated_scores = {}
        
        for goal_id, goal in goal_hierarchy.items():
            if not goal.sub_goals:
                # Leaf goal, use its own satisfaction
                updated_scores[goal_id] = goal.satisfaction
            else:
                # Parent goal, aggregate child satisfactions
                child_satisfactions = [
                    child.satisfaction for child in goal.sub_goals
                ]
                # Weighted average (could be more sophisticated)
                aggregated = sum(child_satisfactions) / len(child_satisfactions)
                updated_scores[goal_id] = 0.7 * goal.satisfaction + 0.3 * aggregated
        
        return updated_scores


class DynamicGoalSystem(nn.Module):
    """
    Dynamic Goal System for AGI
    
    Enables the system to:
    1. Question whether current goals are appropriate
    2. Generate new goals based on experience
    3. Redefine goals dynamically
    4. Manage goal hierarchies
    5. Ask "Are my goals fundamental enough?"
    """
    
    def __init__(
        self,
        context_dim: int = 128,
        goal_dim: int = 64,
        num_goal_types: int = 7,
    ):
        super().__init__()
        
        self.goal_questioner = GoalQuestioner(goal_dim, context_dim)
        self.goal_generator = GoalGenerator(context_dim, goal_dim, num_goal_types)
        self.hierarchy_manager = GoalHierarchyManager(goal_dim)
        
        # Current active goals
        self.active_goals: Dict[str, Goal] = {}
        self.goal_embeddings: Dict[str, torch.Tensor] = {}
        
        # Goal evolution history
        self.goal_history: List[Dict[str, Goal]] = []
    
    def question_goal(
        self,
        goal_id: str,
        context: torch.Tensor,
    ) -> Dict[str, any]:
        """
        Question whether a goal is still appropriate.
        
        Args:
            goal_id: ID of the goal to question
            context: Current context
            
        Returns:
            Evaluation results
        """
        if goal_id not in self.active_goals:
            raise ValueError(f"Goal {goal_id} not found")
        
        goal_embedding = self.goal_embeddings[goal_id]
        evaluation = self.goal_questioner(goal_embedding, context)
        
        return {
            'goal': self.active_goals[goal_id],
            'evaluation': evaluation,
            'recommendation': (
                'reconsider' if evaluation['should_reconsider']
                else 'maintain'
            ),
        }
    
    def generate_new_goals(
        self,
        context: torch.Tensor,
        top_k: int = 3,
    ) -> List[Goal]:
        """
        Generate new goals based on current context.
        
        Args:
            context: Current context
            top_k: Number of goals to generate
            
        Returns:
            List of newly generated goals
        """
        generated = self.goal_generator(context, top_k)
        
        new_goals = []
        for embedding, priority, fundamentality, goal_type in generated:
            goal = Goal(
                goal_type=goal_type,
                description=f"Generated {goal_type.value} goal",
                priority=priority,
                satisfaction=0.0,
                is_fundamental=fundamentality > 0.7,
            )
            new_goals.append(goal)
            
            # Store embedding
            goal_id = f"{goal_type.value}_{len(self.active_goals)}"
            self.goal_embeddings[goal_id] = embedding
        
        return new_goals
    
    def redefine_goals(
        self,
        context: torch.Tensor,
        force_regeneration: bool = False,
    ) -> Dict[str, any]:
        """
        Redefine goals based on current state and questioning.
        
        Args:
            context: Current context
            force_regeneration: Force generation of new goals
            
        Returns:
            Dictionary with redefinition results
        """
        # Question all active goals
        evaluations = {}
        goals_to_reconsider = []
        
        for goal_id in list(self.active_goals.keys()):
            eval_result = self.question_goal(goal_id, context)
            evaluations[goal_id] = eval_result
            
            if eval_result['recommendation'] == 'reconsider' or force_regeneration:
                goals_to_reconsider.append(goal_id)
        
        # Generate new goals if needed
        new_goals = []
        if goals_to_reconsider or force_regeneration:
            new_goals = self.generate_new_goals(context, top_k=len(goals_to_reconsider) + 1)
        
        # Save current state to history
        self.goal_history.append(self.active_goals.copy())
        
        return {
            'evaluations': evaluations,
            'goals_reconsidered': goals_to_reconsider,
            'new_goals': new_goals,
            'total_active_goals': len(self.active_goals),
        }
    
    def update_goal_satisfaction(
        self,
        goal_id: str,
        satisfaction: float,
    ):
        """
        Update satisfaction level for a goal.
        
        Args:
            goal_id: ID of the goal
            satisfaction: New satisfaction level (0.0 to 1.0)
        """
        if goal_id in self.active_goals:
            self.active_goals[goal_id].satisfaction = satisfaction
            
            # Propagate through hierarchy
            updated_scores = self.hierarchy_manager.propagate_satisfaction(
                self.active_goals
            )
            
            for gid, score in updated_scores.items():
                self.active_goals[gid].satisfaction = score
    
    def get_highest_priority_goal(self) -> Optional[Goal]:
        """Get the currently highest priority goal."""
        if not self.active_goals:
            return None
        
        return max(
            self.active_goals.values(),
            key=lambda g: g.priority * (1.0 - g.satisfaction)
        )
    
    def get_diagnostics(self) -> Dict[str, any]:
        """Get diagnostic information about goal system."""
        return {
            'num_active_goals': len(self.active_goals),
            'num_fundamental_goals': sum(
                1 for g in self.active_goals.values() if g.is_fundamental
            ),
            'average_satisfaction': (
                sum(g.satisfaction for g in self.active_goals.values()) / len(self.active_goals)
                if self.active_goals else 0.0
            ),
            'goal_evolution_steps': len(self.goal_history),
            'highest_priority_goal': (
                str(self.get_highest_priority_goal())
                if self.get_highest_priority_goal() else None
            ),
        }
