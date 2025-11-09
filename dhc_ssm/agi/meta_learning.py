"""
Meta-Learning Engine

Implements MAML-inspired meta-learning for rapid adaptation to new tasks.
Enables the system to learn how to learn efficiently.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Callable
from copy import deepcopy
import logging

logger = logging.getLogger(__name__)


class MAMLOptimizer:
    """
    Model-Agnostic Meta-Learning optimizer.
    
    Learns an initialization that can be quickly fine-tuned to new tasks
    with just a few gradient steps.
    """
    
    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        num_inner_steps: int = 5,
    ):
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.num_inner_steps = num_inner_steps
        
        # Outer optimizer for meta-parameters
        self.meta_optimizer = torch.optim.Adam(
            model.parameters(),
            lr=outer_lr
        )
    
    def inner_loop(
        self,
        task_data: Tuple[torch.Tensor, torch.Tensor],
        loss_fn: Callable,
    ) -> Tuple[nn.Module, float]:
        """
        Perform inner loop adaptation for a single task.
        
        Args:
            task_data: Tuple of (inputs, targets) for the task
            loss_fn: Loss function
            
        Returns:
            Tuple of (adapted_model, final_loss)
        """
        inputs, targets = task_data
        
        # Create a copy of the model for adaptation
        adapted_model = deepcopy(self.model)
        
        # Inner loop optimization
        for step in range(self.num_inner_steps):
            # Forward pass
            predictions = adapted_model(inputs)
            loss = loss_fn(predictions, targets)
            
            # Compute gradients
            grads = torch.autograd.grad(
                loss,
                adapted_model.parameters(),
                create_graph=True,
            )
            
            # Manual SGD update
            for param, grad in zip(adapted_model.parameters(), grads):
                param.data = param.data - self.inner_lr * grad
        
        # Final loss after adaptation
        final_predictions = adapted_model(inputs)
        final_loss = loss_fn(final_predictions, targets)
        
        return adapted_model, final_loss.item()
    
    def outer_loop(
        self,
        task_batch: List[Tuple[torch.Tensor, torch.Tensor]],
        loss_fn: Callable,
    ) -> Dict[str, float]:
        """
        Perform outer loop meta-update across multiple tasks.
        
        Args:
            task_batch: List of task data tuples
            loss_fn: Loss function
            
        Returns:
            Dictionary with training metrics
        """
        self.meta_optimizer.zero_grad()
        
        total_loss = 0.0
        task_losses = []
        
        # Accumulate gradients across tasks
        for task_data in task_batch:
            adapted_model, task_loss = self.inner_loop(task_data, loss_fn)
            task_losses.append(task_loss)
            
            # Compute meta-gradient
            inputs, targets = task_data
            predictions = adapted_model(inputs)
            loss = loss_fn(predictions, targets)
            
            loss.backward()
            total_loss += loss.item()
        
        # Meta-update
        self.meta_optimizer.step()
        
        return {
            'meta_loss': total_loss / len(task_batch),
            'mean_task_loss': sum(task_losses) / len(task_losses),
            'min_task_loss': min(task_losses),
            'max_task_loss': max(task_losses),
        }


class TaskEncoder(nn.Module):
    """
    Encodes task characteristics into a latent representation.
    
    This enables the system to understand task structure and
    generalize across similar tasks.
    """
    
    def __init__(self, input_dim: int = 256, task_dim: int = 64):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, task_dim),
        )
        
        # Task similarity network
        self.similarity_net = nn.Sequential(
            nn.Linear(task_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, task_data: torch.Tensor) -> torch.Tensor:
        """
        Encode task into latent representation.
        
        Args:
            task_data: Representative data from the task
            
        Returns:
            Task embedding
        """
        return self.encoder(task_data)
    
    def compute_similarity(
        self,
        task_embedding_1: torch.Tensor,
        task_embedding_2: torch.Tensor,
    ) -> float:
        """
        Compute similarity between two tasks.
        
        Args:
            task_embedding_1: First task embedding
            task_embedding_2: Second task embedding
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        combined = torch.cat([task_embedding_1, task_embedding_2], dim=-1)
        similarity = self.similarity_net(combined)
        return similarity.item()


class AdaptationController(nn.Module):
    """
    Controls the adaptation process based on task characteristics.
    
    Decides how many adaptation steps are needed and what learning
    rate to use for different tasks.
    """
    
    def __init__(self, task_dim: int = 64):
        super().__init__()
        
        # Predict optimal adaptation parameters
        self.controller = nn.Sequential(
            nn.Linear(task_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # [num_steps, learning_rate]
        )
    
    def forward(self, task_embedding: torch.Tensor) -> Dict[str, float]:
        """
        Determine optimal adaptation parameters for a task.
        
        Args:
            task_embedding: Task embedding
            
        Returns:
            Dictionary with adaptation parameters
        """
        output = self.controller(task_embedding)
        
        # Num steps: 1 to 20
        num_steps = 1 + torch.sigmoid(output[0]) * 19
        
        # Learning rate: 0.0001 to 0.1
        lr = 0.0001 + torch.sigmoid(output[1]) * 0.0999
        
        return {
            'num_inner_steps': int(num_steps.item()),
            'inner_lr': lr.item(),
        }


class MetaLearningEngine(nn.Module):
    """
    Meta-Learning Engine for AGI
    
    Implements meta-learning capabilities that enable the system to:
    1. Learn how to learn efficiently
    2. Rapidly adapt to new tasks with few examples
    3. Transfer knowledge across tasks
    4. Understand task structure and relationships
    5. Optimize its own learning process
    """
    
    def __init__(
        self,
        model: nn.Module,
        input_dim: int = 256,
        task_dim: int = 64,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        num_inner_steps: int = 5,
    ):
        super().__init__()
        
        self.model = model
        self.task_encoder = TaskEncoder(input_dim, task_dim)
        self.adaptation_controller = AdaptationController(task_dim)
        
        # MAML optimizer
        self.maml = MAMLOptimizer(
            model, inner_lr, outer_lr, num_inner_steps
        )
        
        # Task memory
        self.task_memory: Dict[str, torch.Tensor] = {}
        self.task_performance: Dict[str, float] = {}
    
    def encode_task(
        self,
        task_data: torch.Tensor,
        task_id: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Encode a task into latent representation.
        
        Args:
            task_data: Representative data from the task
            task_id: Optional task identifier for memory
            
        Returns:
            Task embedding
        """
        task_embedding = self.task_encoder(task_data)
        
        if task_id is not None:
            self.task_memory[task_id] = task_embedding.detach()
        
        return task_embedding
    
    def find_similar_tasks(
        self,
        task_embedding: torch.Tensor,
        top_k: int = 3,
    ) -> List[Tuple[str, float]]:
        """
        Find similar tasks from memory.
        
        Args:
            task_embedding: Query task embedding
            top_k: Number of similar tasks to return
            
        Returns:
            List of (task_id, similarity_score) tuples
        """
        if not self.task_memory:
            return []
        
        similarities = []
        for task_id, stored_embedding in self.task_memory.items():
            similarity = self.task_encoder.compute_similarity(
                task_embedding, stored_embedding
            )
            similarities.append((task_id, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def adapt_to_task(
        self,
        task_data: Tuple[torch.Tensor, torch.Tensor],
        task_id: Optional[str] = None,
        auto_tune: bool = True,
    ) -> Dict[str, any]:
        """
        Adapt the model to a new task.
        
        Args:
            task_data: Tuple of (inputs, targets)
            task_id: Optional task identifier
            auto_tune: Whether to auto-tune adaptation parameters
            
        Returns:
            Dictionary with adaptation results
        """
        inputs, targets = task_data
        
        # Encode task
        task_embedding = self.encode_task(inputs.mean(dim=0), task_id)
        
        # Find similar tasks
        similar_tasks = self.find_similar_tasks(task_embedding)
        
        # Get optimal adaptation parameters
        if auto_tune:
            adapt_params = self.adaptation_controller(task_embedding)
            self.maml.num_inner_steps = adapt_params['num_inner_steps']
            self.maml.inner_lr = adapt_params['inner_lr']
        
        # Perform adaptation
        loss_fn = nn.CrossEntropyLoss()
        adapted_model, final_loss = self.maml.inner_loop(task_data, loss_fn)
        
        # Store performance
        if task_id is not None:
            self.task_performance[task_id] = final_loss
        
        return {
            'adapted_model': adapted_model,
            'final_loss': final_loss,
            'task_embedding': task_embedding,
            'similar_tasks': similar_tasks,
            'adaptation_params': {
                'num_steps': self.maml.num_inner_steps,
                'learning_rate': self.maml.inner_lr,
            },
        }
    
    def meta_train(
        self,
        task_batch: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Dict[str, float]:
        """
        Perform meta-training across multiple tasks.
        
        Args:
            task_batch: List of task data tuples
            
        Returns:
            Training metrics
        """
        loss_fn = nn.CrossEntropyLoss()
        metrics = self.maml.outer_loop(task_batch, loss_fn)
        
        return metrics
    
    def get_learning_efficiency(self) -> float:
        """
        Measure how efficiently the system learns new tasks.
        
        Returns:
            Learning efficiency score (lower is better)
        """
        if not self.task_performance:
            return float('inf')
        
        # Average final loss across tasks
        avg_loss = sum(self.task_performance.values()) / len(self.task_performance)
        
        return avg_loss
    
    def get_diagnostics(self) -> Dict[str, any]:
        """Get diagnostic information about meta-learning system."""
        return {
            'num_tasks_seen': len(self.task_memory),
            'learning_efficiency': self.get_learning_efficiency(),
            'inner_lr': self.maml.inner_lr,
            'outer_lr': self.maml.outer_lr,
            'num_inner_steps': self.maml.num_inner_steps,
            'recent_tasks': list(self.task_performance.keys())[-5:],
        }
