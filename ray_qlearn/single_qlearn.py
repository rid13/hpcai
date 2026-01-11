"""
Single Node Q-Learning for CliffWalking-v0 (or Frozen Lake if you want, with proper modifications of cource ;) )
Traditional Q-Learning with Ray Core for distributed experience collection
Run this on a SINGLE machine/node

Usage:
    python single_node_qlearn.py --episodes 1000 --workers 2
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


# ============================================================================
# RAY WORKER FOR Q-LEARNING
# ============================================================================

@ray.remote
class QLearningWorker:
    """
    Ray worker that collects experience using Q-Learning
    Each worker has its own copy of Q-table
    """
    def __init__(self, env_name, state_size, action_size, worker_id):
        # To render during training, use (env_name,render_mode='human)'
        self.env = gym.make(env_name)
        self.state_size = state_size
        self.action_size = action_size
        self.worker_id = worker_id
        
        # Initialize Q-table
        self.q_table = np.zeros((state_size, action_size))
        
        # Hyperparameters
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
    
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
        """
        Run one episode and return Q-table updates and metrics
        """
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
            'q_table': self.q_table
        }


# ============================================================================
# MAIN TRAINER
# ============================================================================

class QLearningTrainer:
    """
    Main Q-Learning trainer that aggregates Q-tables from workers
    """
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = np.zeros((state_size, action_size))
    
    def aggregate_q_tables(self, q_tables):
        """
        Aggregate Q-tables from multiple workers
        Uses averaging to combine updates
        """
        self.q_table = np.mean(q_tables, axis=0)
    
    def get_q_table(self):
        return self.q_table.copy()
    
    def save(self, filename='ray_qlearn/single_node_qtable.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"Q-table saved to {filename}")
    
    def load(self, filename='ray_qlearn/single_node_qtable.pkl'):
        with open(filename, 'rb') as f:
            self.q_table = pickle.load(f)
        print(f"Q-table loaded from {filename}")


def train_single_node(env_name="CliffWalking-v0", num_episodes=1000, 
                      num_workers=2, render_every=100):
    """
    Train Q-Learning on single node with multiple workers
    
    Args:
        env_name: Gymnasium environment name
        num_episodes: Number of training episodes
        num_workers: Number of parallel workers (2-4 for single node)
        render_every: Print stats every N episodes
    """
    print(f"\n{'='*70}")
    print(f"SINGLE NODE Q-LEARNING")
    print(f"Environment: {env_name}")
    print(f"Workers: {num_workers}")
    print(f"{'='*70}\n")
    
    # Initialize Ray (local mode for single node)
    ray.init(ignore_reinit_error=True, num_cpus=num_workers)
    
    # Get environment info
    env = gym.make(env_name)
    state_size = env.observation_space.n
    action_size = env.action_space.n
    env.close()
    
    print(f"State space: {state_size}")
    print(f"Action space: {action_size}\n")
    
    # Create trainer
    trainer = QLearningTrainer(state_size, action_size)
    
    # Create workers
    workers = [
        QLearningWorker.remote(env_name, state_size, action_size, i)
        for i in range(num_workers)
    ]
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    scores_window = deque(maxlen=100)
    
    # Epsilon decay
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    
    start_time = time.time()
    
    for episode in range(num_episodes):
        # Distribute current Q-table to workers
        current_q_table = trainer.get_q_table()
        ray.get([worker.set_q_table.remote(current_q_table) for worker in workers])
        ray.get([worker.set_epsilon.remote(epsilon) for worker in workers])
        
        # Run episodes in parallel
        episode_futures = [worker.run_episode.remote() for worker in workers]
        results = ray.get(episode_futures)
        
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
            print(f"Episode {episode:4d}: Avg Reward: {mean_reward:7.2f}, "
                  f"Epsilon: {epsilon:.3f}, Avg Steps: {avg_steps:.1f}")
    
    training_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Final average reward (last 100): {np.mean(scores_window):.2f}")
    print(f"{'='*70}")
    
    # Save Q-table
    trainer.save('ray_qlearn/single_node_qtable.pkl')
    
    # Save results
    results_dict = {
        'environment': env_name,
        'num_episodes': num_episodes,
        'num_workers': num_workers,
        'training_time': training_time,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'final_avg_reward': float(np.mean(scores_window)),
    }
    
    with open('ray_qlearn/single_node_results.json', 'w') as f:
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


def plot_results(episode_rewards, save_path='ray_qlearn/single_node_training.png'):
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
    plt.title('Single Node Q-Learning Training')
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Single Node Q-Learning')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes')
    parser.add_argument('--workers', type=int, default=2, help='Number of workers')
    parser.add_argument('--render-every', type=int, default=100, help='Print stats frequency')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate after training')
    parser.add_argument('--load', type=str, default=None, help='Load Q-table from file')
    
    args = parser.parse_args()
    
    if args.load:
        # Load and evaluate existing Q-table
        trainer = QLearningTrainer(48, 4)  # CliffWalking dimensions
        trainer.load(args.load)
        evaluate_policy("CliffWalking-v0", trainer.q_table, num_episodes=10, render=True)
    else:
        # Train
        episode_rewards, episode_lengths, trainer, training_time = train_single_node(
            env_name="CliffWalking-v0",
            num_episodes=args.episodes,
            num_workers=args.workers,
            render_every=args.render_every
        )
        
        # Plot results
        plot_results(episode_rewards)
        
        # Evaluate if requested
        if args.evaluate:
            evaluate_policy("CliffWalking-v0", trainer.q_table, num_episodes=10, render=True)