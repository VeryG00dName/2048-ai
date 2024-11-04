# 2048 AI with Genetic Algorithm Optimization

This project implements an AI for the game [2048](https://gabrielecirulli.github.io/2048/) using a genetic algorithm to optimize the evaluation function's weights. The AI is capable of reaching high scores and achieving the 2048 tile and beyond.

## **Table of Contents**

- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Optimization Details](#optimization-details)
- [Contributing](#contributing)
- [License](#license)

## **Features**

- **AI Player for 2048:**
  - Implements an AI that plays 2048 using the expectimax algorithm.
  - Uses an evaluation function with weighted heuristics.

- **Genetic Algorithm Optimization:**
  - Optimizes the weights of the AI's evaluation function.
  - Supports interruption and resuming of the optimization process.
  - Utilizes multiprocessing for efficient computation.

- **Visualization:**
  - Provides a graphical interface to watch the AI play the game.
  - Built using Tkinter for easy setup and cross-platform compatibility.

- **Reproducibility:**
  - Allows setting seeds for random number generators to reproduce results.
  - Consistent evaluation across runs with the same seed.

## **Demo**

![2048 AI Demo](demo.gif)

*The AI playing 2048 and achieving high scores.*

## **Installation**

### **Prerequisites**

- **Python 3.6 or higher**
- **Required Python Packages:**
  - `numpy`
  - `tkinter` (usually included with Python)
  - Any other packages used in your project

### **Clone the Repository**

```bash
git clone https://github.com/yourusername/2048-AI-Genetic-Algorithm.git
cd 2048-AI-Genetic-Algorithm
```

### **Install Dependencies**

```bash
pip install -r requirements.txt
```

## **Usage**

### **Running the AI with Visualization**

To watch the AI play 2048 using the optimized weights:

```bash
python main.py --seed SEED
```

**Example:**

```bash
python main.py --seed 42
```

**Parameters:**

- `--seed`: Seed for the random number generators (optional).

### **Optimizing Weights with Genetic Algorithm**

To run the genetic algorithm and optimize the weights:

```bash
python optimize_weights.py
```

**Options:**
You can adjust parameters like population size, number of generations, mutation rate, etc., by modifying the `GeneticOptimizer` initialization in `optimize_weights.py`.

### **Command-Line Arguments for `main.py`**

```bash
usage: main.py [--seed SEED]

optional arguments:
  --seed SEED           Seed for the random number generators.
```

## **Project Structure**

- `main.py`: Runs the AI with visualization.
- `game.py`: Contains the `Game` class with the game logic.
- `ai.py`: Implements the `AI` class with the expectimax algorithm and evaluation function.
- `visualize.py`: Handles the graphical interface using Tkinter.
- `optimize_weights.py`: Contains the `GeneticOptimizer` class to optimize weights using a genetic algorithm.
- `README.md`: Project documentation.
- `requirements.txt`: List of required Python packages.

## **Optimization Details**

The genetic algorithm optimizes the weights for the AI's evaluation function, which considers several heuristics:

- **Corner Heuristic**: Encourages keeping the highest tile in a corner.
- **Adjacency Heuristic**: Rewards having adjacent tiles with the same value.
- **Empty Cells Heuristic**: Favors boards with more empty cells.
- **Monotonicity Heuristic**: Prefers boards where values increase or decrease smoothly.

### **Features of the Genetic Algorithm**

- **Population Management**: Maintains a population of candidate weight sets.
- **Selection**: Uses roulette wheel selection based on fitness scores.
- **Crossover and Mutation**: Generates new candidates through crossover and mutation, with constraints to keep weights within desired ranges.
- **Fitness Evaluation**: Averages scores over multiple games to assess each candidate.
- **State Saving**: Saves progress after each generation, allowing the optimization to be paused and resumed.

## **Contributing**

Contributions are welcome! Please follow these steps:

### **Fork the Repository**

### **Create a Feature Branch**

```bash
git checkout -b feature/YourFeature
```

### **Commit Your Changes**

```bash
git commit -am 'Add new feature'
```

### **Push to the Branch**

```bash
git push origin feature/YourFeature
```

### **Create a Pull Request**

## **License**

This project is licensed under the Apache-2.0 license - see the [LICENSE](LICENSE) file for details.
