"""
Multi-Node Q-Learning for CliffWalking-v0 (Frozen lake should still work too :D)
Distributed Q-Learning across multiple machines using Ray Cluster
Run this on a RAY CLUSTER with multiple nodes

Setup Instructions:
1. On head node:
   ray start --head --port=6379
   
2. On worker nodes:
   ray start --address='<head-node-ip>:6379'
   
3. Run this script on head node:
   python multi_node_qlearn.py --episodes 1000 --workers 8

Usage:
    python multi_node_qlearn.py --episodes 1000 --workers 8 --address <head-node-ip>:6379
"""

import ray
import gymnasium as gym
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import time
import json
import argparse
import pickle
import socket


# ============================================================================
# RAY WORKER FOR Q-LEARNING, ctrl+c ctrl+v from single node :b
# ============================================================================

@ray.remote
class QLearningWorker:
    """
    Ray worker that collects experience using Q-Learning
    Can run on different physical machines in the cluster
    """
    def __init__(self, env_name, state_size, action_size, worker_id):
        self.env = gym.make(env_name)
        self.state_size = state_size
        self.action_size = action_size
        self.worker_id = worker_id
        self.hostname = socket.gethostname()  # Track which node this runs on
        
        # Initialize Q-table
        self.q_table = np.zeros((state_size, action_size))
        
        # Hyperparameters
        self.alpha = 0.1
        self.gamma = 0.99
        self.epsilon = 1.0
    
    def get_hostname(self):
        """Return hostname to verify distribution across nodes"""
        return self.hostname
    
    def set_q_table(self, q_table):
        """Update worker's Q-table from main process"""
        self.q_table = q_table.copy()
    
    def set_epsilon(self, epsilon):
        """Update exploration rate"""
        self.epsilon = epsilon
    
    def select_action(self, state):
        """Epsilon-greedy action selection"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        else:
            return np.argmax(self.q_table[state])
    
    def run_episode(self, max_steps=500):
        """Run one episode and return Q-table updates and metrics"""
        state = self.env.reset()[0]
        episode_reward = 0
        steps = 0
        
        # Track Q-table updates
        q_updates = []
        
        for step in range(max_steps):
            # Select action
            action = self.select_action(state)
            
            # Take action
            next_state, reward, done, truncated, _ = self.env.step(action)
            
            # Q-Learning update
            old_value = self.q_table[state, action]
            next_max = np.max(self.q_table[next_state])
            new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
            
            # Store update
            q_updates.append({
                'state': state,
                'action': action,
                'value': new_value
            })
            
            # Update local Q-table
            self.q_table[state, action] = new_value
            
            episode_reward += reward
            steps += 1
            state = next_state
            
            if done or truncated:
                break
        
        return {
            'q_updates': q_updates,
            'episode_reward': episode_reward,
            'steps': steps,
            'q_table': self.q_table,
            'hostname': self.hostname,
            'worker_id': self.worker_id
        }


# ============================================================================
# MAIN TRAINER
# ============================================================================

class QLearningTrainer: # same as single node, recycle, reuse, reduce ;)
    """
    Main Q-Learning trainer that aggregates Q-tables from distributed workers
    """
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = np.zeros((state_size, action_size))
    
    def aggregate_q_tables(self, q_tables):
        """
        Aggregate Q-tables from multiple workers across different nodes
        Uses averaging to combine updates
        """
        self.q_table = np.mean(q_tables, axis=0)
    
    def get_q_table(self):
        return self.q_table.copy()
    
    def save(self, filename='ray_qlearn/multi_node_qtable.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"Q-table saved to {filename}")
    
    def load(self, filename='ray_qlearn/multi_node_qtable.pkl'):
        with open(filename, 'rb') as f:
            self.q_table = pickle.load(f)
        print(f"Q-table loaded from {filename}")


def train_multi_node(env_name="CliffWalking-v0", num_episodes=1000, 
                     num_workers=8, render_every=100, ray_address=None):
    """
    Train Q-Learning on Ray cluster with multiple nodes
    
    Args:
        env_name: Gymnasium environment name
        num_episodes: Number of training episodes
        num_workers: Number of parallel workers (8-16+ for multi-node)
        render_every: Print stats every N episodes
        ray_address: Ray cluster address (e.g., '192.168.1.100:6379')
    """
    print(f"\n{'='*70}")
    print(f"MULTI-NODE Q-LEARNING (Ray Cluster)")
    print(f"Environment: {env_name}")
    print(f"Workers: {num_workers}")
    print(f"{'='*70}\n")
    
    # Connect to Ray cluster
    if ray_address:
        print(f"Connecting to Ray cluster at {ray_address}...")
        ray.init(address=ray_address, ignore_reinit_error=True)
    else:
        print("No cluster address provided, running in local mode...")
        ray.init(ignore_reinit_error=True, num_cpus=num_workers)
    
    # Print cluster info
    print(f"\nRay Cluster Info:")
    print(f"Available nodes: {len(ray.nodes())}")
    print(f"Available CPUs: {ray.cluster_resources().get('CPU', 0)}")
    print(f"Available GPUs: {ray.cluster_resources().get('GPU', 0)}\n")
    
    # Get environment info
    env = gym.make(env_name)
    state_size = env.observation_space.n
    action_size = env.action_space.n
    env.close()
    
    print(f"State space: {state_size}")
    print(f"Action space: {action_size}\n")
    
    # Create trainer
    trainer = QLearningTrainer(state_size, action_size)
    
    # Create distributed workers
    print(f"Creating {num_workers} distributed workers...")
    workers = [
        QLearningWorker.remote(env_name, state_size, action_size, i)
        for i in range(num_workers)
    ]
    
    # Verify worker distribution across nodes
    print("\nVerifying worker distribution:")
    hostnames = ray.get([worker.get_hostname.remote() for worker in workers])
    unique_hosts = set(hostnames)
    print(f"Workers distributed across {len(unique_hosts)} nodes:")
    for host in unique_hosts:
        count = hostnames.count(host)
        print(f"  {host}: {count} workers")
    print()
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    scores_window = deque(maxlen=100)
    worker_distribution = []
    
    # Epsilon decay
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    
    start_time = time.time()
    
    for episode in range(num_episodes):
        # Distribute current Q-table to all workers (across all nodes)
        current_q_table = trainer.get_q_table()
        ray.get([worker.set_q_table.remote(current_q_table) for worker in workers])
        ray.get([worker.set_epsilon.remote(epsilon) for worker in workers])
        
        # Run episodes in parallel across all nodes
        episode_futures = [worker.run_episode.remote() for worker in workers]
        results = ray.get(episode_futures)
        
        # Track which nodes processed which episodes
        if episode % render_every == 0:
            hosts_used = [r['hostname'] for r in results]
            worker_distribution.append({
                'episode': episode,
                'hosts': list(set(hosts_used)),
                'num_hosts': len(set(hosts_used))
            })
        
        # Aggregate Q-tables from all workers
        worker_q_tables = [r['q_table'] for r in results]
        trainer.aggregate_q_tables(worker_q_tables)
        
        # Collect metrics
        avg_reward = np.mean([r['episode_reward'] for r in results])
        avg_steps = np.mean([r['steps'] for r in results])
        
        episode_rewards.append(avg_reward)
        episode_lengths.append(avg_steps)
        scores_window.append(avg_reward)
        
        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        # Print progress
        if episode % render_every == 0 and episode > 0:
            mean_reward = np.mean(scores_window)
            hosts_this_batch = len(set([r['hostname'] for r in results]))
            print(f"Episode {episode:4d}: Avg Reward: {mean_reward:7.2f}, "
                  f"Epsilon: {epsilon:.3f}, Nodes used: {hosts_this_batch}")
    
    training_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Final average reward (last 100): {np.mean(scores_window):.2f}")
    print(f"Average workers per node: {num_workers / len(unique_hosts):.1f}")
    print(f"{'='*70}")
    
    # Save Q-table
    trainer.save('ray_qlearn/multi_node_qtable.pkl')
    
    # Save results
    results_dict = {
        'environment': env_name,
        'num_episodes': num_episodes,
        'num_workers': num_workers,
        'num_nodes': len(unique_hosts),
        'training_time': training_time,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'final_avg_reward': float(np.mean(scores_window)),
        'nodes_used': list(unique_hosts),
        'worker_distribution': worker_distribution,
    }
    
    with open('ray_qlearn/multi_node_results.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    ray.shutdown()
    
    return episode_rewards, episode_lengths, trainer, training_time


def evaluate_policy(env_name, q_table, num_episodes=10, render=True):
    """Evaluate trained Q-table"""
    print(f"\n{'='*70}")
    print("EVALUATION")
    print(f"{'='*70}\n")
    
    if render:
        env = gym.make(env_name, render_mode='human')
    else:
        env = gym.make(env_name)
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()[0]
        episode_reward = 0
        steps = 0
        
        for step in range(500):
            action = np.argmax(q_table[state])
            state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            steps += 1
            
            if done or truncated:
                break
        
        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {steps}")
    
    env.close()
    
    print(f"\nMean Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    
    return episode_rewards


def plot_results(episode_rewards, save_path='ray_qlearn/multi_node_training.png'):
    """Plot training results"""
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Raw rewards
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, alpha=0.3)
    
    # Moving average
    window = 50
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(episode_rewards)), moving_avg, 'r-', linewidth=2, label='50-episode MA')
    
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.title('Multi-Node Q-Learning Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Distribution of final rewards
    plt.subplot(1, 2, 2)
    final_rewards = episode_rewards[-100:] if len(episode_rewards) >= 100 else episode_rewards
    plt.hist(final_rewards, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Episode Reward')
    plt.ylabel('Frequency')
    plt.title('Distribution of Final 100 Episodes')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved to {save_path}")
    # plt.show()


def compare_single_vs_multi(single_results_file='ray_qlearn/single_node_results.json',
                           multi_results_file='ray_qlearn/multi_node_results.json'):
    """
    Compare single-node vs multi-node performance
    Load results from JSON files
    """
    try:
        with open(single_results_file, 'r') as f:
            single_results = json.load(f)
        with open(multi_results_file, 'r') as f:
            multi_results = json.load(f)
    except FileNotFoundError:
        print("Results files not found. Run both single and multi-node training first.")
        return
    
    print(f"\n{'='*70}")
    print("PERFORMANCE COMPARISON: Single-Node vs Multi-Node")
    print(f"{'='*70}")
    print(f"\nSingle Node:")
    print(f"  Workers: {single_results['num_workers']}")
    print(f"  Training time: {single_results['training_time']:.2f}s")
    print(f"  Final avg reward: {single_results['final_avg_reward']:.2f}")
    
    print(f"\nMulti-Node:")
    print(f"  Workers: {multi_results['num_workers']}")
    print(f"  Nodes: {multi_results['num_nodes']}")
    print(f"  Training time: {multi_results['training_time']:.2f}s")
    print(f"  Final avg reward: {multi_results['final_avg_reward']:.2f}")
    
    speedup = single_results['training_time'] / multi_results['training_time']
    efficiency = speedup / (multi_results['num_workers'] / single_results['num_workers'])
    
    print(f"\nSpeedup: {speedup:.2f}x")
    print(f"Parallel Efficiency: {efficiency:.2%}")
    print(f"{'='*70}")
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Training time
    ax = axes[0]
    categories = ['Single\nNode', 'Multi\nNode']
    times = [single_results['training_time'], multi_results['training_time']]
    bars = ax.bar(categories, times, color=['blue', 'red'], alpha=0.7)
    ax.set_ylabel('Training Time (seconds)')
    ax.set_title(f'Training Time Comparison\nSpeedup: {speedup:.2f}x')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, t in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{t:.1f}s', ha='center', va='bottom')
    
    # Plot 2: Final performance
    ax = axes[1]
    final_rewards = [single_results['final_avg_reward'], multi_results['final_avg_reward']]
    bars = ax.bar(categories, final_rewards, color=['blue', 'red'], alpha=0.7)
    ax.set_ylabel('Final Average Reward')
    ax.set_title('Final Performance Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, r in zip(bars, final_rewards):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{r:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('ray_qlearn/single_vs_multi_comparison.png', dpi=150)
    print("\nComparison plot saved to 'ray_qlearn/single_vs_multi_comparison.png'")
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi-Node Q-Learning with Ray Cluster')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes')
    parser.add_argument('--workers', type=int, default=8, help='Number of workers (8-16 for cluster)')
    parser.add_argument('--render-every', type=int, default=100, help='Print stats frequency')
    parser.add_argument('--address', type=str, default=None, help='Ray cluster address (e.g., 192.168.1.100:6379)')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate after training')
    parser.add_argument('--load', type=str, default=None, help='Load Q-table from file')
    parser.add_argument('--compare', action='store_true', help='Compare single vs multi-node results')
    
    args = parser.parse_args()
    
    if args.compare:
        # Compare existing results
        compare_single_vs_multi()
    elif args.load:
        # Load and evaluate existing Q-table
        trainer = QLearningTrainer(48, 4)  # CliffWalking dimensions
        trainer.load(args.load)
        evaluate_policy("CliffWalking-v0", trainer.q_table, num_episodes=10, render=True)
    else:
        # Train on cluster
        episode_rewards, episode_lengths, trainer, training_time = train_multi_node(
            env_name="CliffWalking-v0",
            num_episodes=args.episodes,
            num_workers=args.workers,
            render_every=args.render_every,
            ray_address=args.address
        )
        
        # Plot results
        plot_results(episode_rewards)
        
        # Evaluate if requested
        if args.evaluate:
            evaluate_policy("CliffWalking-v0", trainer.q_table, num_episodes=5, render=True)