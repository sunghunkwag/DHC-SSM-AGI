"""
AGI-Enhanced DHC-SSM Demonstration

This example demonstrates the key AGI capabilities of the DHC-SSM-AGI system:
1. Meta-cognitive self-awareness
2. Uncertainty quantification
3. Meta-learning adaptation
4. Self-improvement
5. Dynamic goal management
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.append('..')

from dhc_ssm.agi import AGIEnhancedDHCSSM
from dhc_ssm.utils.config import get_default_config


def demonstrate_basic_inference():
    """Demonstrate basic inference with AGI insights."""
    print("=" * 80)
    print("DEMO 1: Basic Inference with AGI Insights")
    print("=" * 80)
    
    # Create model
    config = get_default_config()
    model = AGIEnhancedDHCSSM(config)
    model.eval()
    
    # Create sample input
    x = torch.randn(8, 3, 32, 32)
    targets = torch.randint(0, 10, (8,))
    
    # Forward pass with AGI enabled
    with torch.no_grad():
        results = model(x, targets=targets, enable_agi=True)
    
    # Display results
    print(f"\nPredictions shape: {results['predictions'].shape}")
    print(f"Uncertainty level: {results['uncertainty']['level']}")
    print(f"Total uncertainty: {results['uncertainty']['total'].mean():.4f}")
    print(f"Epistemic uncertainty: {results['uncertainty']['epistemic'].mean():.4f}")
    print(f"Aleatoric uncertainty: {results['uncertainty']['aleatoric'].mean():.4f}")
    print(f"Confidence: {results['uncertainty']['confidence'].mean():.4f}")
    print(f"Should explore: {results['should_explore']}")
    print(f"Should adapt: {results['should_adapt']}")
    print(f"Performance trend: {results['metacognition']['performance_trend']:.4f}")
    print(f"Weakness score: {results['metacognition']['weakness_score'].mean():.4f}")
    
    print("\nDemo 1 completed successfully!\n")


def demonstrate_meta_learning():
    """Demonstrate meta-learning adaptation to new tasks."""
    print("=" * 80)
    print("DEMO 2: Meta-Learning Adaptation")
    print("=" * 80)
    
    # Create model
    config = get_default_config()
    model = AGIEnhancedDHCSSM(config)
    
    # Create multiple tasks
    tasks = []
    for i in range(3):
        task_inputs = torch.randn(16, 3, 32, 32)
        task_targets = torch.randint(0, 10, (16,))
        tasks.append((task_inputs, task_targets, f"task_{i}"))
    
    print(f"\nAdapting to {len(tasks)} different tasks...\n")
    
    # Adapt to each task
    for task_inputs, task_targets, task_id in tasks:
        task_data = (task_inputs, task_targets)
        
        adaptation_results = model.adapt_to_task(
            task_data,
            task_id=task_id,
            auto_tune=True
        )
        
        print(f"Task: {task_id}")
        print(f"  Final loss: {adaptation_results['final_loss']:.4f}")
        print(f"  Adaptation steps: {adaptation_results['adaptation_params']['num_steps']}")
        print(f"  Learning rate: {adaptation_results['adaptation_params']['learning_rate']:.6f}")
        print(f"  Similar tasks: {[t[0] for t in adaptation_results['similar_tasks'][:2]]}")
        print()
    
    # Check meta-learning diagnostics
    ml_diagnostics = model.meta_learning.get_diagnostics()
    print(f"Meta-Learning Summary:")
    print(f"  Tasks seen: {ml_diagnostics['num_tasks_seen']}")
    print(f"  Learning efficiency: {ml_diagnostics['learning_efficiency']:.4f}")
    
    print("\nDemo 2 completed successfully!\n")


def demonstrate_self_improvement():
    """Demonstrate recursive self-improvement."""
    print("=" * 80)
    print("DEMO 3: Recursive Self-Improvement")
    print("=" * 80)
    
    # Create model
    config = get_default_config()
    model = AGIEnhancedDHCSSM(config)
    
    # Simulate some training history
    print("\nSimulating training history...")
    for epoch in range(5):
        x = torch.randn(8, 3, 32, 32)
        targets = torch.randint(0, 10, (8,))
        
        with torch.no_grad():
            results = model(x, targets=targets, enable_agi=True)
    
    # Get current state and metrics
    current_state = model.vectorize_config()
    x = torch.randn(8, 3, 32, 32)
    targets = torch.randint(0, 10, (8,))
    
    with torch.no_grad():
        results = model(x, targets=targets, enable_agi=True)
    current_metrics = model.compute_metrics(results['predictions'], targets)
    
    # Trigger self-improvement
    print("\nGenerating improvement hypotheses...")
    improvement_results = model.improve_self(current_state, current_metrics)
    
    print(f"\nGenerated {len(improvement_results['hypotheses'])} hypotheses:")
    for i, hypothesis in enumerate(improvement_results['hypotheses'], 1):
        print(f"  {i}. {hypothesis}")
    
    print(f"\nSelf-Improvement Statistics:")
    print(f"  Success rate: {improvement_results['success_rate']:.2%}")
    print(f"  Best strategies: {improvement_results['best_strategies']}")
    
    print("\nDemo 3 completed successfully!\n")


def demonstrate_uncertainty_tracking():
    """Demonstrate uncertainty quantification over time."""
    print("=" * 80)
    print("DEMO 4: Uncertainty Tracking")
    print("=" * 80)
    
    # Create model
    config = get_default_config()
    model = AGIEnhancedDHCSSM(config)
    model.eval()
    
    print("\nTracking uncertainty over multiple inferences...")
    
    # Run multiple inferences
    for i in range(15):
        x = torch.randn(4, 3, 32, 32)
        
        with torch.no_grad():
            results = model(x, enable_agi=True)
    
    # Get uncertainty diagnostics
    unc_diagnostics = model.uncertainty.get_diagnostics()
    
    print(f"\nUncertainty Diagnostics:")
    print(f"  History length: {unc_diagnostics['history_length']}")
    print(f"  Average total uncertainty: {unc_diagnostics['avg_total_uncertainty']:.4f}")
    print(f"  Average epistemic uncertainty: {unc_diagnostics['avg_epistemic_uncertainty']:.4f}")
    print(f"  Average aleatoric uncertainty: {unc_diagnostics['avg_aleatoric_uncertainty']:.4f}")
    print(f"  Is learning: {unc_diagnostics['is_learning']}")
    print(f"  Trends:")
    for key, value in unc_diagnostics['trends'].items():
        print(f"    {key}: {value:.4f}")
    
    print("\nDemo 4 completed successfully!\n")


def demonstrate_comprehensive_diagnostics():
    """Demonstrate comprehensive system diagnostics."""
    print("=" * 80)
    print("DEMO 5: Comprehensive System Diagnostics")
    print("=" * 80)
    
    # Create model
    config = get_default_config()
    model = AGIEnhancedDHCSSM(config)
    
    # Run some operations to populate history
    print("\nRunning operations to populate system history...")
    
    for i in range(10):
        x = torch.randn(4, 3, 32, 32)
        targets = torch.randint(0, 10, (4,))
        
        with torch.no_grad():
            results = model(x, targets=targets, enable_agi=True)
    
    # Get comprehensive diagnostics
    diagnostics = model.get_comprehensive_diagnostics()
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE SYSTEM DIAGNOSTICS")
    print("=" * 80)
    
    print("\nBase Model:")
    for key, value in diagnostics['base_model'].items():
        print(f"  {key}: {value}")
    
    print("\nMeta-Cognition:")
    for key, value in diagnostics['metacognition'].items():
        print(f"  {key}: {value}")
    
    print("\nSelf-Improvement:")
    for key, value in diagnostics['self_improvement'].items():
        if key != 'recent_results':
            print(f"  {key}: {value}")
    
    print("\nGoal System:")
    for key, value in diagnostics['goal_system'].items():
        print(f"  {key}: {value}")
    
    print("\nUncertainty Quantification:")
    for key, value in diagnostics['uncertainty'].items():
        if key != 'trends':
            print(f"  {key}: {value}")
    
    print("\nMeta-Learning:")
    for key, value in diagnostics['meta_learning'].items():
        if key != 'recent_tasks':
            print(f"  {key}: {value}")
    
    print("\nSystem Modes:")
    for key, value in diagnostics['modes'].items():
        print(f"  {key}: {value}")
    
    print("\nDemo 5 completed successfully!\n")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 80)
    print("DHC-SSM-AGI DEMONSTRATION")
    print("=" * 80)
    print("\nThis demo showcases the AGI capabilities of DHC-SSM-AGI:")
    print("1. Basic inference with uncertainty and meta-cognition")
    print("2. Meta-learning adaptation to new tasks")
    print("3. Recursive self-improvement")
    print("4. Uncertainty tracking over time")
    print("5. Comprehensive system diagnostics")
    print("\n" + "=" * 80 + "\n")
    
    try:
        demonstrate_basic_inference()
        demonstrate_meta_learning()
        demonstrate_self_improvement()
        demonstrate_uncertainty_tracking()
        demonstrate_comprehensive_diagnostics()
        
        print("=" * 80)
        print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nThe DHC-SSM-AGI system has demonstrated:")
        print("- Self-awareness through meta-cognitive monitoring")
        print("- Uncertainty quantification for informed decision-making")
        print("- Rapid adaptation through meta-learning")
        print("- Structured self-improvement capabilities")
        print("- Comprehensive diagnostic and introspection tools")
        print("\nThese capabilities form the foundation for artificial general intelligence.")
        print("=" * 80 + "\n")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
