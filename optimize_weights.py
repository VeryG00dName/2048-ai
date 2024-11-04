# optimize_weights.py

import multiprocessing
import random
import time
import numpy as np
import os
import pickle
from game import Game
from ai import AI

def run_game(args):
    """
    Run a single game and return the score.
    """
    weights, search_depth, game_seed = args
    try:
        # Set the seed for reproducibility
        random.seed(game_seed)
        np.random.seed(game_seed)
        # Run the game
        game = Game()
        ai = AI(weights=weights, search_depth=search_depth)
        while not game.is_game_over():
            move = ai.decide_move(game)
            game.move(move)
        return game.score
    except Exception as e:
        print(f"Exception in run_game: {e}")
        return None  # Indicate failure

class GeneticOptimizer:
    def __init__(self, population_size=20, generations=10, mutation_rate=0.1,
                 crossover_rate=0.7, num_games=5, search_depth=3,
                 save_file='optimizer_state.pkl', seed=42, weight_constraints=None):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.num_games = num_games  # Number of games to average for fitness
        self.search_depth = search_depth  # Depth for AI search
        self.num_weights = 4  # Number of weights in the evaluation function
        self.save_file = save_file  # File to save the optimizer state
        self.current_generation = 0  # Initialize current generation

        self.seed = seed
        self.random_state = np.random.RandomState(seed)
        random.seed(seed)

        # Initialize weight constraints
        if weight_constraints is None:
            # Default constraints: all weights >= 0
            self.weight_constraints = [(0, None)] * self.num_weights
        else:
            self.weight_constraints = weight_constraints

        # Initialize population and fitnesses
        self.population = None
        self.fitnesses = None

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            # Random weights within specified constraints
            weights = []
            for i in range(self.num_weights):
                min_wt, max_wt = self.weight_constraints[i]
                if min_wt is None:
                    min_wt = -10
                if max_wt is None:
                    max_wt = 10
                weight = self.random_state.uniform(min_wt, max_wt)
                weights.append(weight)
            population.append(np.array(weights))
        return population

    def fitness_function(self, weights, individual_index, total_individuals):
        """
        Evaluate the average score over multiple games with a per-game timeout.
        """
        time_limit_per_game = 30  # Time limit per game in seconds
        scores = []
        start_time = time.time()
        completed_games = 0

        # Prepare arguments for each game, including unique seeds
        args_list = []
        for i in range(self.num_games):
            game_seed = self.seed + individual_index * self.num_games + i
            args_list.append((weights, self.search_depth, game_seed))

        with multiprocessing.Pool() as pool:
            # Submit all game tasks to the pool
            results = [pool.apply_async(run_game, args=(args,)) for args in args_list]

            # Process results with per-game timeout
            for i, result in enumerate(results):
                try:
                    score = result.get(timeout=time_limit_per_game)
                    if score is not None:
                        scores.append(score)
                    completed_games += 1
                except multiprocessing.TimeoutError:
                    # Game exceeded time limit
                    print(f"Game {i + 1}/{self.num_games} for Individual {individual_index} timed out.")
                    completed_games += 1
                except Exception as e:
                    print(f"Exception in game {i + 1}/{self.num_games} for Individual {individual_index}: {e}")
                    completed_games += 1

                # Print current stats every 10 seconds
                elapsed_time = time.time() - start_time
                if elapsed_time >= 10 and int(elapsed_time) % 10 == 0:
                    if scores:
                        average_score = np.mean(scores)
                    else:
                        average_score = 0
                    print(f"Individual {individual_index}/{total_individuals} is taking long to evaluate...")
                    print(f"Elapsed Time: {int(elapsed_time)}s, Completed Games: {completed_games}/{self.num_games}, Current Average Score: {average_score}")

        average_score = np.mean(scores) if scores else 0
        return average_score

    def selection(self, population, fitnesses):
        """
        Select two individuals based on fitness using roulette wheel selection.
        """
        total_fitness = sum(fitnesses)
        if total_fitness == 0:
            probabilities = [1 / len(fitnesses)] * len(fitnesses)
        else:
            probabilities = [f / total_fitness for f in fitnesses]
        selected_indices = self.random_state.choice(len(population), size=2, p=probabilities)
        return population[selected_indices[0]], population[selected_indices[1]]

    def crossover(self, parent1, parent2):
        """
        Perform single-point crossover between two parents,
        enforcing constraints on the weights.
        """
        if self.random_state.rand() < self.crossover_rate:
            point = self.random_state.randint(1, self.num_weights - 1)
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
        else:
            child1, child2 = parent1.copy(), parent2.copy()

        # Enforce weight constraints
        for child in [child1, child2]:
            for i in range(self.num_weights):
                min_wt, max_wt = self.weight_constraints[i]
                if min_wt is not None:
                    child[i] = max(min_wt, child[i])
                if max_wt is not None:
                    child[i] = min(max_wt, child[i])
        return child1, child2

    def mutate(self, individual):
        """
        Mutate an individual's weights with small random changes,
        enforcing constraints on the weights.
        """
        for i in range(self.num_weights):
            if self.random_state.rand() < self.mutation_rate:
                individual[i] += self.random_state.normal(0, 0.1)
                # Enforce weight constraints
                min_wt, max_wt = self.weight_constraints[i]
                if min_wt is not None:
                    individual[i] = max(min_wt, individual[i])
                if max_wt is not None:
                    individual[i] = min(max_wt, individual[i])
        return individual

    def save_state(self):
        """
        Save the current state of the optimizer to a file.
        """
        state = {
            'current_generation': self.current_generation,
            'population': self.population,
            'fitnesses': self.fitnesses,
            'parameters': {
                'population_size': self.population_size,
                'generations': self.generations,
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate,
                'num_games': self.num_games,
                'search_depth': self.search_depth,
                'num_weights': self.num_weights,
                'seed': self.seed,
                'weight_constraints': self.weight_constraints
            }
        }
        with open(self.save_file, 'wb') as f:
            pickle.dump(state, f)
        print(f"State saved to {self.save_file} at generation {self.current_generation}.")

    def load_state(self):
        """
        Load the optimizer state from a file.
        """
        with open(self.save_file, 'rb') as f:
            state = pickle.load(f)
        self.current_generation = state['current_generation']
        self.population = state['population']
        self.fitnesses = state['fitnesses']
        # Load parameters to ensure consistency
        params = state['parameters']
        self.population_size = params['population_size']
        self.generations = params['generations']
        self.mutation_rate = params['mutation_rate']
        self.crossover_rate = params['crossover_rate']
        self.num_games = params['num_games']
        self.search_depth = params['search_depth']
        self.num_weights = params['num_weights']
        self.seed = params['seed']
        self.weight_constraints = params['weight_constraints']
        # Reset random state
        self.random_state = np.random.RandomState(self.seed)
        random.seed(self.seed)
        print(f"Resuming from saved state at generation {self.current_generation}.")

    def optimize(self):
        """
        Run the genetic algorithm to optimize the weights.
        """
        # Check if a saved state exists
        if os.path.exists(self.save_file):
            self.load_state()
        else:
            self.current_generation = 0
            self.population = self.initialize_population()
            self.fitnesses = [0] * self.population_size

        for generation in range(self.current_generation, self.generations):
            self.current_generation = generation
            print(f"Generation {generation + 1}/{self.generations}")
            fitnesses = []
            total_individuals = len(self.population)
            for idx, individual in enumerate(self.population):
                fitness = self.fitness_function(individual, idx + 1, total_individuals)
                fitnesses.append(fitness)
                print(f"Individual {idx + 1}/{total_individuals} Fitness: {fitness}")
            self.fitnesses = fitnesses  # Update fitnesses

            # Save state after each generation
            self.save_state()

            new_population = []
            while len(new_population) < self.population_size:
                parent1, parent2 = self.selection(self.population, self.fitnesses)
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                new_population.extend([child1, child2])
            self.population = new_population[:self.population_size]

        # After all generations, return the best individual
        best_fitness = max(self.fitnesses)
        best_index = self.fitnesses.index(best_fitness)
        best_weights = self.population[best_index]
        print(f"Best Weights: {best_weights} Fitness: {best_fitness}")

        # Clean up saved state file
        if os.path.exists(self.save_file):
            os.remove(self.save_file)
            print(f"Saved state file {self.save_file} removed after completion.")

        return best_weights

if __name__ == "__main__":
    # On Windows, you might need this line to support multiprocessing
    multiprocessing.freeze_support()
    optimizer = GeneticOptimizer(
        population_size=20,
        generations=10,
        mutation_rate=0.1,
        crossover_rate=0.7,
        num_games=5,
        search_depth=4,
        save_file='optimizer_state.pkl',
        seed=42,
        weight_constraints=[
            (0, None),  # corner_weight >= 0
            (0, None),  # adjacency_weight >= 0
            (0, None),  # empty_cells_weight >= 0
            (0, None),  # monotonicity_weight >= 0
        ]
    )
    best_weights = optimizer.optimize()
    print("Optimization Completed.")
    print(f"Best Weights Found: {best_weights}")
