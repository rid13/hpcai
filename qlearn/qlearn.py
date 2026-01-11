"""
Basic tabular Q-learning implementation.
A sort of POC if you want before introducing Ray-based parallelism
"""
import gymnasium as gym
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import time

class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, gamma=0.99, 
                 epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        """
        Q-Learning Agent
        
        Args:
            state_size: Number of states
            action_size: Number of actions
            learning_rate: Learning rate (alpha)
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Epsilon decay rate per episode
        """
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Initialize Q-table with zeros
        self.q_table = np.zeros((state_size, action_size))
    
    def select_action(self, state):
        """Select action using epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(self.action_size)
        else:
            # Exploit: best action from Q-table
            return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state, done):
        """Update Q-table using Q-learning update rule"""
        # Q-learning update rule:
        # Q(s,a) = Q(s,a) + lr * (reward + gamma * max(Q(s',a')) - Q(s,a))
        
        current_q = self.q_table[state, action]
        
        if done:
            # Terminal state: no future rewards
            target_q = reward
        else:
            # Non-terminal: add discounted max future Q-value
            max_next_q = np.max(self.q_table[next_state])
            target_q = reward + self.gamma * max_next_q
        
        # Update Q-value
        self.q_table[state, action] = current_q + self.lr * (target_q - current_q)
    
    def decay_epsilon(self):
        """Decay epsilon after each episode"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train_qlearning(env_no_render, env_render, agent, n_episodes=1000, max_steps=1000, 
                    print_every=100, render_every=50):
    """
    Train Q-learning agent with selective rendering
    
    Args:
        env_no_render: Gym environment without rendering
        env_render: Gym environment with rendering
        agent: QLearningAgent instance
        n_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        print_every: Print stats every N episodes
        render_every: Render every N episodes
    """
    scores = []
    scores_deque = deque(maxlen=100)
    epsilons = []
    
    for episode in range(1, n_episodes + 1):
        # Choose which environment to use
        should_render = (episode % render_every == 0)
        # should_render = 0 # switch betweeen these two lines to enable rendering or not
        current_env = env_render if should_render else env_no_render
        
        state = current_env.reset()[0]
        total_reward = 0
        
        for step in range(max_steps):
            # Select action
            action = agent.select_action(state)
            
            # Take action
            next_state, reward, done, truncated, info = current_env.step(action)
            total_reward += reward
            
            # Update Q-table
            agent.update(state, action, reward, next_state, done)
            
            state = next_state
            
            if done or truncated:
                break
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Record statistics
        scores.append(total_reward)
        scores_deque.append(total_reward)
        epsilons.append(agent.epsilon)
        
        # Print progress
        if episode % print_every == 0:
            avg_score = np.mean(scores_deque)
            print(f'Episode {episode}\tAverage Score: {avg_score:.2f}\tEpsilon: {agent.epsilon:.3f}')
    
    return scores, epsilons


def evaluate_agent(env, agent, n_episodes=10, max_steps=1000):
    """
    Evaluate the trained agent
    
    Args:
        env: Gym environment (with render_mode='human')
        agent: Trained QLearningAgent
        n_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
    """
    episode_rewards = []
    
    for episode in range(n_episodes):
        state = env.reset()[0]
        total_reward = 0
        
        for step in range(max_steps):
            # Always exploit during evaluation (epsilon=0)
            action = np.argmax(agent.q_table[state])
            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            state = next_state
            
            if done or truncated:
                break
        
        episode_rewards.append(total_reward)
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    
    return mean_reward, std_reward


def plot_training_results(scores, epsilons):
    """Plot training scores and epsilon decay"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot scores
    ax1.plot(scores, alpha=0.6, label='Episode Score')
    # Plot moving average
    window = 100
    if len(scores) >= window:
        moving_avg = np.convolve(scores, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(scores)), moving_avg, 'r-', linewidth=2, label=f'{window}-Episode Moving Avg')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')
    ax1.set_title('Training Scores')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot epsilon decay
    ax2.plot(epsilons, 'g-', linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Epsilon')
    ax2.set_title('Exploration Rate (Epsilon) Decay')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('qlearn/qlearning_training_results.png', dpi=150, bbox_inches='tight')
    print("Training plots saved to 'qlearning_training_results.png'")
    # plt.show()


if __name__ == '__main__':
    env_id = "CliffWalking-v0"
    # env_id = "FrozenLake-v1"
    
    # Create TWO environments: one for rendering, one for fast training
    env_no_render = gym.make(env_id)  # No rendering - fast training
    env_render = gym.make(env_id, render_mode='human')  # With rendering
    
    # Get state and action space sizes
    state_size = env_no_render.observation_space.n
    action_size = env_no_render.action_space.n
    
    print(f"State space size: {state_size}")
    print(f"Action space size: {action_size}")
    
    # parametres
    parametres = {
        "learning_rate": 0.1,       # Alpha
        "gamma": 0.99,              # Discount factor
        "epsilon_start": 10.0,       # Initial exploration rate
        "epsilon_min": 0.01,        # Minimum exploration rate
        "epsilon_decay": 0.995,     # Epsilon decay per episode
        "n_episodes": 1000,         # Number of training episodes
        "max_steps": 1000,          # Max steps per episode
        "n_eval_episodes": 10,      # Number of evaluation episodes
    }
    
    # Create Q-learning agent
    agent = QLearningAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=parametres["learning_rate"],
        gamma=parametres["gamma"],
        epsilon_start=parametres["epsilon_start"],
        epsilon_min=parametres["epsilon_min"],
        epsilon_decay=parametres["epsilon_decay"]
    )
    
    print(f"\n{'='*70}")
    print("\nStarting Q-learning training...")
    print("Rendering every 50th episode to show progress.\n")
    print(f"\n{'='*70}")
    
    # Train the agent
    start = time.time()
    scores, epsilons = train_qlearning(
        env_no_render=env_no_render,
        env_render=env_render,
        agent=agent,
        n_episodes=parametres["n_episodes"],
        max_steps=parametres["max_steps"],
        print_every=100,
        render_every=50  # Only render every 50th episode
    )
    end = time.time()
    print(f"Training completed in {end-start:.2f} seconds")
    
    # Save Q-table
    np.save("qlearn/cliff_walking_qtable.npy", agent.q_table)
    print("\nQ-table saved to 'cliff_walking_qtable.npy'")
    
    env_no_render.close()
    env_render.close()
    
    # Plot training results
    plot_training_results(scores, epsilons)
    
    # Evaluate the trained agent
    print(f"\n{'='*70}")
    print("\nEvaluating trained agent...")
    print(f"\n{'='*70}")

    eval_env = gym.make(env_id, render_mode='human')

    start = time.time()
    mean_reward, std_reward = evaluate_agent(
        eval_env,
        agent,
        n_episodes=parametres["n_eval_episodes"],
        max_steps=parametres["max_steps"]
    )
    end = time.time()
    print(f"Evaluation completed in {end-start:.2f} seconds")
    
    print(f"\nEvaluation Results:")
    print(f"Mean Reward: {mean_reward:.2f}")
    print(f"Std Reward: {std_reward:.2f}")
    
    eval_env.close()