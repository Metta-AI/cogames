"""
Visualization utilities for CoGames tutorials.

This module provides comprehensive visualization functions for the tutorial notebooks,
including training progress plots, value function heatmaps, subtask tracking, and
multi-agent coordination analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Optional, Tuple


# =============================================================================
# Core Visualizations (All Stages)
# =============================================================================

def smooth_curve(values: List[float], window: int = 50) -> np.ndarray:
    """Apply moving average smoothing to a curve.
    
    Args:
        values: List of values to smooth
        window: Size of smoothing window
        
    Returns:
        Smoothed values as numpy array
    """
    if len(values) < window:
        return np.array(values)
    return np.convolve(values, np.ones(window) / window, mode='valid')


def plot_training_progress(
    episode_returns: List[float],
    episode_lengths: List[float],
    window: int = 50,
    success_threshold: float = 2.0
) -> plt.Figure:
    """Plot training progress showing returns and episode lengths.
    
    Args:
        episode_returns: List of episode returns
        episode_lengths: List of episode lengths
        window: Smoothing window size
        success_threshold: Return threshold for success line
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Episode returns
    ax1.plot(episode_returns, alpha=0.2, color='blue', label='Raw')
    if len(episode_returns) >= window:
        smoothed_returns = smooth_curve(episode_returns, window)
        x_smooth = range(window - 1, len(episode_returns))
        ax1.plot(x_smooth, smoothed_returns, color='blue', linewidth=2,
                 label=f'Smoothed ({window} episodes)')
    ax1.axhline(success_threshold, color='red', linestyle='--', 
                label=f'Success Threshold ({success_threshold})')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Return')
    ax1.set_title('Episode Returns Over Training')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Episode lengths (efficiency)
    ax2.plot(episode_lengths, alpha=0.2, color='green', label='Raw')
    if len(episode_lengths) >= window:
        smoothed_lengths = smooth_curve(episode_lengths, window)
        x_smooth = range(window - 1, len(episode_lengths))
        ax2.plot(x_smooth, smoothed_lengths, color='green', linewidth=2,
                 label=f'Smoothed ({window} episodes)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.set_title('Episode Length Over Training (Lower = More Efficient)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_success_rate(
    episode_returns: List[float],
    window: int = 50,
    threshold: float = 2.0
) -> plt.Figure:
    """Plot rolling success rate.
    
    Args:
        episode_returns: List of episode returns
        window: Rolling window size
        threshold: Success threshold
        
    Returns:
        Matplotlib figure
    """
    successes = [1 if r >= threshold else 0 for r in episode_returns]
    
    fig = plt.figure(figsize=(10, 4))
    
    if len(successes) >= window:
        success_rate = smooth_curve(successes, window)
        x_smooth = range(window - 1, len(successes))
        plt.plot(x_smooth, success_rate, color='green', linewidth=2)
    else:
        plt.plot(successes, color='green', alpha=0.5)
    
    plt.axhline(0.6, color='orange', linestyle='--', label='60% Target')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.title(f'Success Rate (Rolling {window} episodes)')
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return fig


def plot_value_heatmap(
    policy,
    env,
    map_height: int = 10,
    map_width: int = 10
) -> plt.Figure:
    """Visualize learned value function across map positions.
    
    Shows which positions the agent thinks are valuable.
    High values near chest/assembler indicate good learning.
    
    Args:
        policy: Trained policy with network() method
        env: Environment instance
        map_height: Map height
        map_width: Map width
        
    Returns:
        Matplotlib figure
    """
    import torch
    
    values = np.zeros((map_height, map_width))
    
    # TODO: Implement observation creation for each position
    # This requires understanding MettagGrid's observation construction
    # For now, create placeholder
    
    fig = plt.figure(figsize=(10, 8))
    im = plt.imshow(values, cmap='viridis', interpolation='nearest', origin='lower')
    plt.colorbar(im, label='Estimated Value')
    plt.title('Learned Value Function Heatmap\n(Brighter = Higher Expected Return)')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    
    # TODO: Overlay object positions (chest, assembler, etc.)
    
    return fig


# =============================================================================
# Stage 1: Navigation Visualizations
# =============================================================================

def plot_navigation_progress(episode_data: Dict) -> plt.Figure:
    """Track agent's distance to chest over training.
    
    Args:
        episode_data: Dictionary with 'distances' key containing list of distance arrays
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Distance over sample episode
    if 'distances' in episode_data and len(episode_data['distances']) > 0:
        sample_episode = episode_data['distances'][-1]
        ax1.plot(sample_episode, color='blue', linewidth=2)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Distance to Chest')
        ax1.set_title('Distance to Goal During Episode\n(Final Episode)')
        ax1.grid(True, alpha=0.3)
        
        # Average distances over training
        initial_distances = [ep[0] for ep in episode_data['distances'] if len(ep) > 0]
        final_distances = [ep[-1] for ep in episode_data['distances'] if len(ep) > 0]
        
        ax2.plot(initial_distances, alpha=0.3, label='Initial Distance', color='red')
        if len(initial_distances) >= 20:
            ax2.plot(smooth_curve(initial_distances, 20), linewidth=2, 
                    label='Initial (Smoothed)', color='red')
        ax2.plot(final_distances, alpha=0.3, label='Final Distance', color='green')
        if len(final_distances) >= 20:
            ax2.plot(smooth_curve(final_distances, 20), linewidth=2,
                    label='Final (Smoothed)', color='green')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Distance')
        ax2.set_title('Distance to Chest Over Training')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_position_heatmap(
    visited_positions: List[Tuple[int, int]],
    map_height: int = 10,
    map_width: int = 10
) -> plt.Figure:
    """Show where agent spent time during training.
    
    Args:
        visited_positions: List of (x, y) position tuples
        map_height: Map height
        map_width: Map width
        
    Returns:
        Matplotlib figure
    """
    heatmap = np.zeros((map_height, map_width))
    
    for pos in visited_positions:
        x, y = pos
        if 0 <= x < map_width and 0 <= y < map_height:
            heatmap[y, x] += 1
    
    fig = plt.figure(figsize=(10, 8))
    im = plt.imshow(heatmap, cmap='hot', interpolation='nearest', origin='lower')
    plt.colorbar(im, label='Visit Count')
    plt.title('Agent Position Heatmap\n(Brighter = More Time Spent)')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    
    # TODO: Mark chest location
    
    return fig


# =============================================================================
# Stage 2: Crafting Subtask Visualizations
# =============================================================================

def plot_crafting_subtasks(metrics: Dict) -> plt.Figure:
    """Track completion of crafting sub-goals.
    
    Shows agent learning the sequence: craft → deposit
    
    Args:
        metrics: Dictionary from evaluate_policy() with keys:
            - 'crafting_events': List of per-episode craft counts
            - 'episode_returns': List of per-episode returns (hearts deposited)
            
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    crafting_events = metrics.get('crafting_events', [])
    episode_returns = metrics.get('episode_returns', [])
    
    if not crafting_events or not episode_returns:
        # No data to plot
        fig.text(0.5, 0.5, 'No subtask data available',
                ha='center', va='center', fontsize=14)
        return fig
    
    episodes = np.arange(len(crafting_events))
    window_size = min(20, max(5, len(crafting_events) // 10))
    
    # Subtask 1: Hearts Crafted per Episode
    axes[0, 0].plot(episodes, crafting_events, alpha=0.3, color='green', label='Raw')
    if len(crafting_events) >= window_size:
        axes[0, 0].plot(episodes, smooth_curve(crafting_events, window_size), 
                       color='green', linewidth=2, label=f'Smoothed ({window_size})')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Hearts Crafted')
    axes[0, 0].set_title('Subtask 1: Crafting Success')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Subtask 2: Hearts Deposited per Episode
    axes[0, 1].plot(episodes, episode_returns, alpha=0.3, color='red', label='Raw')
    if len(episode_returns) >= window_size:
        axes[0, 1].plot(episodes, smooth_curve(episode_returns, window_size), 
                       color='red', linewidth=2, label=f'Smoothed ({window_size})')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Hearts Deposited')
    axes[0, 1].set_title('Subtask 2: Depositing Success')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Subtask 3: Craft-to-Deposit Efficiency
    efficiency = [d / max(c, 1) for c, d in zip(crafting_events, episode_returns)]
    axes[1, 0].plot(episodes, efficiency, alpha=0.3, color='purple', label='Raw')
    if len(efficiency) >= window_size:
        axes[1, 0].plot(episodes, smooth_curve(efficiency, window_size), 
                       color='purple', linewidth=2, label=f'Smoothed ({window_size})')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Deposit/Craft Ratio')
    axes[1, 0].set_title('Subtask 3: Crafting → Depositing Efficiency')
    axes[1, 0].set_ylim([0, 1.1])
    axes[1, 0].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect (1.0)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Summary Statistics
    avg_crafted = np.mean(crafting_events[-20:]) if len(crafting_events) >= 20 else np.mean(crafting_events)
    avg_deposited = np.mean(episode_returns[-20:]) if len(episode_returns) >= 20 else np.mean(episode_returns)
    avg_efficiency = np.mean(efficiency[-20:]) if len(efficiency) >= 20 else np.mean(efficiency)
    
    summary_text = f"""Final Performance (last 20 episodes):
    
Hearts Crafted:  {avg_crafted:.2f} / episode
Hearts Deposited: {avg_deposited:.2f} / episode  
Efficiency:      {avg_efficiency:.2%}

Interpretation:
• High craft, low deposit = navigation issue
• Efficiency < 50% = losing hearts somehow
• Efficiency ~100% = perfect execution ✓
"""
    
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                   verticalalignment='center', transform=axes[1, 1].transAxes)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    return fig


def plot_inventory_timeline(inventory_history: Dict) -> plt.Figure:
    """Show resource flow during episode.
    
    Visualizes: Resources consumed → Heart created → Heart deposited
    
    Args:
        inventory_history: Dictionary with resource keys (carbon, oxygen, etc.)
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    if not inventory_history:
        return fig
    
    # Resource inventory
    steps = range(len(inventory_history.get('carbon', [])))
    if 'carbon' in inventory_history:
        axes[0].plot(steps, inventory_history['carbon'], label='Carbon', marker='o')
    if 'oxygen' in inventory_history:
        axes[0].plot(steps, inventory_history['oxygen'], label='Oxygen', marker='s')
    if 'germanium' in inventory_history:
        axes[0].plot(steps, inventory_history['germanium'], label='Germanium', marker='^')
    if 'silicon' in inventory_history:
        axes[0].plot(steps, inventory_history['silicon'], label='Silicon', marker='d')
    
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Quantity')
    axes[0].set_title('Crafting Resources Over Episode')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Heart inventory
    if 'heart' in inventory_history:
        axes[1].plot(steps, inventory_history['heart'],
                     color='red', linewidth=3, marker='o', markersize=8)
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('Hearts')
        axes[1].set_title('Heart Inventory Over Episode\n(Increase = Crafted, Decrease = Deposited)')
        axes[1].set_ylim(-0.5, max(inventory_history['heart']) + 0.5 if inventory_history['heart'] else 1)
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# =============================================================================
# Stage 3: Multi-Agent Coordination Visualizations
# =============================================================================

def plot_multiagent_returns(agent_histories: List[Dict], num_agents: int = 4) -> plt.Figure:
    """Compare individual agent and team performance.
    
    Args:
        agent_histories: List of dictionaries with 'returns' and 'episode_lengths'
        num_agents: Number of agents
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Individual agent returns
    for i in range(min(num_agents, len(agent_histories))):
        if 'returns' in agent_histories[i]:
            agent_returns = agent_histories[i]['returns']
            if len(agent_returns) >= 50:
                smoothed = smooth_curve(agent_returns, 50)
                x_smooth = range(49, len(agent_returns))
                axes[0, 0].plot(x_smooth, smoothed, label=f'Agent {i}', alpha=0.7, linewidth=2)
            else:
                axes[0, 0].plot(agent_returns, label=f'Agent {i}', alpha=0.7, linewidth=2)
    
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Return')
    axes[0, 0].set_title('Individual Agent Returns')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Team total return
    max_len = max(len(agent_histories[i].get('returns', [])) for i in range(min(num_agents, len(agent_histories))))
    team_returns = []
    for ep in range(max_len):
        episode_sum = sum(
            agent_histories[i]['returns'][ep]
            for i in range(min(num_agents, len(agent_histories)))
            if 'returns' in agent_histories[i] and ep < len(agent_histories[i]['returns'])
        )
        team_returns.append(episode_sum)
    
    if len(team_returns) >= 50:
        smoothed = smooth_curve(team_returns, 50)
        x_smooth = range(49, len(team_returns))
        axes[0, 1].plot(x_smooth, smoothed, color='purple', linewidth=2)
    else:
        axes[0, 1].plot(team_returns, color='purple', linewidth=2)
    
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Team Return')
    axes[0, 1].set_title('Team Total Return')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Episode length comparison
    for i in range(min(num_agents, len(agent_histories))):
        if 'episode_lengths' in agent_histories[i]:
            agent_lengths = agent_histories[i]['episode_lengths']
            if len(agent_lengths) >= 50:
                smoothed = smooth_curve(agent_lengths, 50)
                x_smooth = range(49, len(agent_lengths))
                axes[1, 0].plot(x_smooth, smoothed, label=f'Agent {i}', alpha=0.7, linewidth=2)
            else:
                axes[1, 0].plot(agent_lengths, label=f'Agent {i}', alpha=0.7, linewidth=2)
    
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Steps')
    axes[1, 0].set_title('Episode Lengths (Efficiency)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Success rate per agent
    for i in range(min(num_agents, len(agent_histories))):
        if 'returns' in agent_histories[i]:
            agent_returns = agent_histories[i]['returns']
            successes = [1 if r >= 1.0 else 0 for r in agent_returns]
            if len(successes) >= 50:
                success_rate = smooth_curve(successes, 50)
                x_smooth = range(49, len(successes))
                axes[1, 1].plot(x_smooth, success_rate, label=f'Agent {i}', alpha=0.7, linewidth=2)
            else:
                axes[1, 1].plot(successes, label=f'Agent {i}', alpha=0.7, linewidth=2)
    
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Success Rate')
    axes[1, 1].set_title('Per-Agent Success Rates')
    axes[1, 1].set_ylim(0, 1.1)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_agent_specialization(agent_activities: List[Dict], num_agents: int = 4) -> plt.Figure:
    """Visualize if agents develop specialized roles.
    
    Args:
        agent_activities: List of dictionaries with activity counts per agent
        num_agents: Number of agents
        
    Returns:
        Matplotlib figure
    """
    try:
        import seaborn as sns
    except ImportError:
        print("Warning: seaborn not available, using basic matplotlib")
        return plt.figure(figsize=(16, 6))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Activity heatmap
    activities = ['Extractor Visits', 'Assembler Visits', 'Chest Visits']
    activity_matrix = np.array([
        [agent_activities[i].get('extractor_visits', 0) for i in range(min(num_agents, len(agent_activities)))],
        [agent_activities[i].get('assembler_visits', 0) for i in range(min(num_agents, len(agent_activities)))],
        [agent_activities[i].get('chest_visits', 0) for i in range(min(num_agents, len(agent_activities)))],
    ])
    
    sns.heatmap(activity_matrix,
                xticklabels=[f'Agent {i}' for i in range(min(num_agents, len(agent_activities)))],
                yticklabels=activities,
                annot=True, fmt='.0f', cmap='Blues', ax=ax1)
    ax1.set_title('Agent Activity Patterns\n(Total Visits)')
    
    # Time spent distribution
    time_matrix = np.array([
        [agent_activities[i].get('time_at_extractors', 0) for i in range(min(num_agents, len(agent_activities)))],
        [agent_activities[i].get('time_at_assembler', 0) for i in range(min(num_agents, len(agent_activities)))],
        [agent_activities[i].get('time_at_chest', 0) for i in range(min(num_agents, len(agent_activities)))],
        [agent_activities[i].get('time_moving', 0) for i in range(min(num_agents, len(agent_activities)))],
    ])
    
    # Normalize to percentages
    col_sums = time_matrix.sum(axis=0)
    col_sums[col_sums == 0] = 1  # Avoid division by zero
    time_matrix_pct = time_matrix / col_sums * 100
    
    sns.heatmap(time_matrix_pct,
                xticklabels=[f'Agent {i}' for i in range(min(num_agents, len(agent_activities)))],
                yticklabels=['Extractors', 'Assembler', 'Chest', 'Moving'],
                annot=True, fmt='.1f', cmap='YlOrRd', ax=ax2,
                cbar_kws={'label': 'Percentage of Time'})
    ax2.set_title('Time Allocation Per Agent\n(Percentage)')
    
    plt.tight_layout()
    return fig


def plot_coordination_metrics(episode_data: List[Dict]) -> plt.Figure:
    """Measure coordination efficiency.
    
    Args:
        episode_data: List of episode dictionaries with coordination metrics
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Collision rate over training
    collision_rate = [
        ep.get('collisions', 0) / max(ep.get('steps', 1), 1)
        for ep in episode_data
    ]
    if len(collision_rate) >= 20:
        smoothed = smooth_curve(collision_rate, 20)
        x_smooth = range(19, len(collision_rate))
        axes[0, 0].plot(x_smooth, smoothed, color='red', linewidth=2)
    else:
        axes[0, 0].plot(collision_rate, color='red', linewidth=2)
    
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Collisions per Step')
    axes[0, 0].set_title('Agent Collision Rate\n(Lower = Better Coordination)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Resource gathering balance
    resource_balance = [
        np.std(ep.get('resources_per_agent', [0, 0, 0, 0]))
        for ep in episode_data
    ]
    if len(resource_balance) >= 20:
        smoothed = smooth_curve(resource_balance, 20)
        x_smooth = range(19, len(resource_balance))
        axes[0, 1].plot(x_smooth, smoothed, color='orange', linewidth=2)
    else:
        axes[0, 1].plot(resource_balance, color='orange', linewidth=2)
    
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Std Dev of Resources')
    axes[0, 1].set_title('Resource Gathering Balance\n(Lower = More Equal Distribution)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Task distribution (last 100 episodes)
    recent_episodes = episode_data[-100:] if len(episode_data) > 100 else episode_data
    if recent_episodes and 'hearts_per_agent' in recent_episodes[0]:
        hearts_per_agent = np.array([ep.get('hearts_per_agent', [0, 0, 0, 0]) for ep in recent_episodes])
        axes[1, 0].boxplot(hearts_per_agent, labels=[f'Agent {i}' for i in range(hearts_per_agent.shape[1])])
        axes[1, 0].set_ylabel('Hearts Deposited')
        axes[1, 0].set_title('Task Distribution (Last 100 Episodes)')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Concurrent activity
    concurrent_activity = [ep.get('avg_active_agents', 0) for ep in episode_data]
    if len(concurrent_activity) >= 20:
        smoothed = smooth_curve(concurrent_activity, 20)
        x_smooth = range(19, len(concurrent_activity))
        axes[1, 1].plot(x_smooth, smoothed, color='green', linewidth=2)
    else:
        axes[1, 1].plot(concurrent_activity, color='green', linewidth=2)
    
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Avg Active Agents')
    axes[1, 1].set_title('Concurrent Activity\n(Higher = Better Parallelization)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# =============================================================================
# Utility Functions
# =============================================================================

def create_metrics_table(
    episode_returns: List[float],
    episode_lengths: List[float],
    training_time: float
) -> pd.DataFrame:
    """Create summary table of training results.
    
    Args:
        episode_returns: List of episode returns
        episode_lengths: List of episode lengths
        training_time: Training time in seconds
        
    Returns:
        Pandas DataFrame with metrics
    """
    if not episode_returns:
        return pd.DataFrame({'Metric': [], 'Value': []})
    
    # Calculate metrics
    final_success_rate = np.mean([1 if r >= 2.0 else 0
                                   for r in episode_returns[-100:]])
    avg_return = np.mean(episode_returns[-100:]) if len(episode_returns) >= 100 else np.mean(episode_returns)
    avg_length = np.mean(episode_lengths[-100:]) if len(episode_lengths) >= 100 else np.mean(episode_lengths)
    best_return = np.max(episode_returns)
    
    metrics = pd.DataFrame({
        'Metric': [
            'Final Success Rate',
            'Avg Return (Last 100)',
            'Best Return',
            'Avg Episode Length',
            'Training Time',
            'Total Episodes'
        ],
        'Value': [
            f'{final_success_rate:.1%}',
            f'{avg_return:.2f}',
            f'{best_return:.2f}',
            f'{avg_length:.0f} steps',
            f'{training_time:.1f}s',
            f'{len(episode_returns)}'
        ]
    })
    
    return metrics


def evaluate_policy(
    config,
    policy,
    num_episodes: int = 100,
    max_steps: int = 200,
    seed: int = 42,
    track_inventory: bool = False,
    sample_episodes: int = 10
) -> Dict:
    """Evaluate a trained policy and collect metrics.
    
    Args:
        config: Game configuration
        policy: Trained policy to evaluate
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        seed: Random seed
        track_inventory: If True, track detailed inventory over time (slower)
        sample_episodes: Number of episodes to track inventory for (if track_inventory=True)
        
    Returns:
        Dictionary with episode returns, lengths, positions, and optionally inventory data
    """
    from mettagrid import MettaGridEnv
    import numpy as np
    
    env = MettaGridEnv(env_cfg=config)
    np.random.seed(seed)
    
    episode_returns = []
    episode_lengths = []
    all_positions = []
    
    # For detailed tracking (optional)
    inventory_timelines = []  # List of lists of inventory dicts
    crafting_events = []  # List of episode crafting counts
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        total_reward = 0.0
        prev_hearts = 0
        crafts_this_episode = 0
        
        # Track inventory for sample episodes
        episode_inventory = [] if (track_inventory and ep < sample_episodes) else None
        
        while not done and steps < max_steps:
            # Get agent position
            agent_pos = env.get_agent_positions()[0] if hasattr(env, 'get_agent_positions') else None
            if agent_pos is not None:
                all_positions.append(agent_pos)
            
            # Track inventory if requested
            if episode_inventory is not None and hasattr(env, 'agents'):
                try:
                    agent = env.agents[0] if hasattr(env, 'agents') else None
                    if agent and hasattr(agent, 'inventory'):
                        inv_snapshot = {
                            'step': steps,
                            'carbon': agent.inventory.get('carbon', 0),
                            'oxygen': agent.inventory.get('oxygen', 0),
                            'germanium': agent.inventory.get('germanium', 0),
                            'silicon': agent.inventory.get('silicon', 0),
                            'heart': agent.inventory.get('heart', 0),
                        }
                        episode_inventory.append(inv_snapshot)
                        
                        # Detect crafting (heart count increased)
                        current_hearts = inv_snapshot['heart']
                        if current_hearts > prev_hearts:
                            crafts_this_episode += 1
                        prev_hearts = current_hearts
                except (AttributeError, IndexError, KeyError):
                    pass  # Environment doesn't expose inventory in expected way
            
            # Get action from policy
            agent_policy = policy.agent_policy(0)  # Single agent
            action = agent_policy.step(obs[0])
            
            # Step environment
            obs, rewards, dones, truncated, info = env.step([action])
            total_reward += rewards[0]
            done = dones[0] or truncated[0]
            steps += 1
        
        episode_returns.append(total_reward)
        episode_lengths.append(steps)
        crafting_events.append(crafts_this_episode)
        
        if episode_inventory is not None:
            inventory_timelines.append(episode_inventory)
    
    env.close()
    
    result = {
        'episode_returns': episode_returns,
        'episode_lengths': episode_lengths,
        'positions': all_positions,
        'crafting_events': crafting_events,
    }
    
    if track_inventory and inventory_timelines:
        result['inventory_timelines'] = inventory_timelines
    
    return result


def create_gif_from_episode(
    env,
    policy,
    filename: str = 'episode.gif',
    max_steps: int = 200
) -> str:
    """Record episode and save as animated GIF.
    
    Note: GIF creation requires RGB rendering, which is not currently
    exposed by MettagGrid's Python API. Use the GUI visualizer instead:
        cogames play <game> --policy <policy> --policy-data <checkpoint>
    
    Args:
        env: Environment instance
        policy: Trained policy
        filename: Output filename
        max_steps: Maximum steps to record
        
    Returns:
        Path to saved GIF (empty placeholder)
    """
    print(f"ℹ️  GIF creation not available - MettagGrid uses mettascope GUI for visualization")
    print(f"   To visualize your trained agent, use:")
    print(f"   cogames play <game> --policy simple --policy-data <checkpoint>")
    return ""


# =============================================================================
# Enhanced Visualization Functions (using crafting_events from evaluate_policy)
# =============================================================================

def plot_crafting_analysis(metrics: Dict) -> plt.Figure:
    """Analyze crafting and depositing performance.
    
    Uses the 'crafting_events' data collected by evaluate_policy().
    
    Args:
        metrics: Dictionary from evaluate_policy() with crafting_events
        
    Returns:
        Matplotlib figure
    """
    # Use the existing plot_crafting_subtasks implementation
    return plot_crafting_subtasks(metrics)

