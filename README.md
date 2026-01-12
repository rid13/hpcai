# Report

This is the report for the hpcai lab, by Rida El Goumiri

---

# Environment

The chosen environment for this lab is [Cliff Walking](https://gymnasium.farama.org/environments/toy_text/cliff_walking/).
The code is inspired by the [tutorial([https://gymnasium.farama.org/introduction/train_agent/](https://gymnasium.farama.org/introduction/train_agent/))] for blackjack provided by gymnasium.

*Ps: the code should be appropriate also for the [Frozen Lake](https://gymnasium.farama.org/environments/toy_text/frozen_lake/) environment, as both of them are based on similar action spaces.*

---

# Implementation

The focus was mainly on implementing a Qlearn algorithme using ray from scratch, which explains the use of a simple enviroment.

## First version

The first version implements tabular Q-learning using Gymnasium and Numpy. The objective is validating the algorithm's correctness and learning behavior before introducing Ray-based parallelism.
The agent follows an ε-greedy policy with exponential decay and updates values using the standard off-policy Q-learning rule.
The learned Q-table and training plots are saved for reproducibility. Evaluation is conducted with ε set to zero to measure the learned policy.

---

## Ray-core version

### Single node

This version implments tabular Q-learning using Ray core on a single node to introduce parallelism while keeping the algorithme relatively unchanged. Multiple Ray actors are used to train in parallel, each running an independant instance of the environment.

EAch worker maints a local Q-table and executes one episode per training iteraton using an ε-greedy policy. After each episode, the central process agregates all workers' Q-tables by calculating the mean value. A shared global Q-table is then produced and redistributed amongst the workers. Exploration rate ε is synchronized across workers and decayed centrally.

This serves as validation for the Ray-based execution model, thus serving as an intermediate step before scaling to multi-node deployments.

### Multi node

The final version extends the single-node Ray implementation into a multi-node Ray cluster. The Q-learning remains unchanged, but Ray actors are now distributed across multiples nodes.

The workers split the work load in the same manner as the single node variant, except ,well, being on seperated machines instead of the same one.

Worker hostnames are recorded to verify effective distribution across nodes, ensuring that experience collection occurs on multiple machines. Training metrics and artifacts are saved for reproducibility and comparison.

Because the environment is on the smaller side (48 states, action space of size 4), scalling efficiency is limited by communications and synchronisation overhead. Nontheless, this version demonstrates correct multi-node execution with Ray Core and completes the required progression from a single-process baseline to a fully distributed reinforcement learning setup.

---

## Commands used for multi node execution (2 nodes)

Reserve 2 nodes on Grid5000:

```bash
oarsub -I -l nodes=2,walltime=02:00:00
```

Identify the nodes:

```bash
cat $OAR_NODEFILE
```

The first node is the head node, while the second node is the worker node.

On head node:

```bash
ray stop
ray start --head --port=6379
```

On worker node:

```bash
ray stop
ray start --address=<head ip>:6379
```

On head node:

```bash
python multi_node_qlearn.py --workers 2 --episodes 1000 --address <head ip>:6379
```

---

**PS:** All three implementations allow rendering while training to see the progress as the agent progresses, this can be disabled or enabled in the code by commenting the relevant lines in the training functions.

---

# Performance

Convergence and comparaison graphs are present alongside the code.

## Performance Comparison and Discussion

### Single-Node vs Multi-Node Ray Execution

| Configuration | Workers | Nodes | Training Time (s) | Final Avg Reward |
| ------------- | ------- | ----- | ----------------- | ---------------- |
| Single-node   | 2       | 1     | 3.30              | -16.32           |
| Multi-node    | 2       | 2     | 7.48              | -16.28           |

The multi-node execution does **not provide a speedup** for this experiment. With only two workers and a very small tabular problem, communication and synchronization overhead dominate computation time, resulting in a slowdown (speedup = 0.44×, parallel efficiency ≈ 44%).

Final policy performance is comparable across configurations, confirming that **distribution does not affect learning correctness**, only execution efficiency. This behavior is expected for lightweight environments such as *CliffWalking-v0*.

---

### Baseline Q-Learning Results (No Ray)

The standalone Q-learning implementation converges steadily over 1000 episodes. Average episode reward improves from very large negative values (due to frequent cliff falls during early exploration) to approximately **−50** during training.

Final evaluation of the learned policy yields:

* **Training time**: 4.50 s
* **Mean reward**: −13.0
* **Standard deviation**: 0.0

This corresponds to a near-optimal policy for *CliffWalking-v0*, where the optimal path still incurs a negative return. The baseline confirms correct algorithmic behavior and provides a reference point for Ray-based implementations.

---

### Key Takeaways

* Tabular Q-learning converges reliably on *CliffWalking-v0*.
* Ray Core enables single-node and multi-node parallelism without changing learning outcomes.
* For small environments, **distributed execution is slower** due to overhead.
* Multi-node execution mainly demonstrates **scalability mechanics**, not performance gains, in this setting.


