"""
Genetic Algorithm for Feature Selection

This module provides genetic algorithm-based methods for feature selection.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, Callable, Tuple
import random
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from autofeature.feature_selection.base import BaseSelector
from autofeature.feature_selection.wrapper_methods import clone_model


class GeneticSelector(BaseSelector):
    """Genetic algorithm based feature selector.
    
    This selector uses a genetic algorithm to find an optimal feature subset.
    """
    
    def __init__(self, model_type: str = 'auto',
                 model: Optional[Any] = None,
                 cv: int = 5,
                 scoring: Union[str, Callable] = None,
                 n_generations: int = 10,
                 population_size: int = 50,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.7,
                 tournament_size: int = 3,
                 elite_size: int = 5,
                 n_jobs: int = -1,
                 verbose: int = 0,
                 early_stopping_rounds: Optional[int] = None,
                 random_state: Optional[int] = None,
                 min_features: int = 1):
        """Initialize the genetic selector.
        
        Args:
            model_type: Type of model to use ('regression', 'classification', 'auto')
            model: Pre-configured model to use (if None, a default model is used)
            cv: Number of cross-validation folds
            scoring: Scoring metric for cross-validation
            n_generations: Number of generations
            population_size: Size of the population
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            tournament_size: Size of tournament for selection
            elite_size: Number of top individuals to carry over to next generation
            n_jobs: Number of parallel jobs for cross-validation
            verbose: Verbosity level
            early_stopping_rounds: Stop if no improvement after this many generations
            random_state: Random seed for reproducibility
            min_features: Minimum number of features to select
        """
        super().__init__()
        self.model_type = model_type
        self.model = model
        self.cv = cv
        self.scoring = scoring
        self.n_generations = n_generations
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elite_size = elite_size
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.early_stopping_rounds = early_stopping_rounds
        self.min_features = min_features
        self.random_state = random_state
        
        # Set random seed if provided
        if self.random_state is not None:
            random.seed(self.random_state)
            np.random.seed(self.random_state)
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'GeneticSelector':
        """Fit the selector to the data.
        
        Args:
            X: Input features
            y: Target variable
            
        Returns:
            self: The fitted selector
        """
        self._validate_input(X)
        
        # Initialize feature importance dict
        self.feature_importances_ = {col: 0.0 for col in X.columns}
        
        # Detect task type if auto
        if self.model_type == 'auto':
            is_regression = self._is_regression_task(y)
            self.model_type = 'regression' if is_regression else 'classification'
        
        # Setup model and scoring
        model = self._setup_model()
        scoring = self._setup_scoring()
        
        # Run genetic algorithm
        self.selected_features, self.feature_importances_ = self._run_genetic_algorithm(
            X, y, model, scoring
        )
        
        self.is_fitted = True
        return self
    
    def _setup_model(self) -> Any:
        """Set up the model for feature selection.
        
        Returns:
            Any: Model instance
        """
        if self.model is not None:
            return self.model
        
        # Create default model based on task type
        if self.model_type == 'regression':
            return RandomForestRegressor(n_estimators=100, n_jobs=self.n_jobs, random_state=self.random_state)
        else:
            return RandomForestClassifier(n_estimators=100, n_jobs=self.n_jobs, random_state=self.random_state)
    
    def _setup_scoring(self) -> Union[str, Callable]:
        """Set up scoring metric for cross-validation.
        
        Returns:
            Union[str, Callable]: Scoring metric
        """
        if self.scoring is not None:
            return self.scoring
        
        # Default scoring based on task type
        if self.model_type == 'regression':
            return 'r2'
        else:
            return 'accuracy'
    
    def _run_genetic_algorithm(self, X: pd.DataFrame, y: pd.Series, 
                              model: Any, scoring: Union[str, Callable]) -> Tuple[List[str], Dict[str, float]]:
        """Run the genetic algorithm for feature selection.
        
        Args:
            X: Input features
            y: Target variable
            model: Model to use
            scoring: Scoring metric
            
        Returns:
            Tuple: Selected features and feature importances
        """
        n_features = X.shape[1]
        feature_names = list(X.columns)
        
        # Initialize population randomly
        population = self._initialize_population(n_features)
        
        # Track best solution overall
        best_solution = None
        best_score = -np.inf
        best_feature_mask = None
        
        # For early stopping
        generations_no_improve = 0
        
        # Generation loop
        for generation in range(self.n_generations):
            # Evaluate population
            fitness_scores = []
            for individual in population:
                # Get features corresponding to 1s in the individual
                selected_features = [feature_names[i] for i in range(n_features) if individual[i] == 1]
                
                # If we have too few features, penalize heavily
                if len(selected_features) < self.min_features:
                    fitness_scores.append(-np.inf)
                    continue
                
                # Evaluate feature set
                score = self._evaluate_feature_set(X, y, selected_features, model, scoring)
                fitness_scores.append(score)
            
            # Find best individual in this generation
            best_idx = np.argmax(fitness_scores)
            generation_best_score = fitness_scores[best_idx]
            generation_best_solution = population[best_idx]
            
            # Update overall best if better
            if generation_best_score > best_score:
                best_score = generation_best_score
                best_solution = generation_best_solution
                best_feature_mask = [i for i in range(n_features) if generation_best_solution[i] == 1]
                generations_no_improve = 0
            else:
                generations_no_improve += 1
            
            # Check for early stopping
            if self.early_stopping_rounds is not None and generations_no_improve >= self.early_stopping_rounds:
                if self.verbose > 0:
                    print(f"Early stopping at generation {generation+1} (no improvement for {generations_no_improve} generations)")
                break
            
            if self.verbose > 0:
                selected_count = sum(generation_best_solution)
                print(f"Generation {generation+1}/{self.n_generations}: "
                     f"Best score = {generation_best_score:.4f}, "
                     f"Features = {selected_count}/{n_features}")
            
            # Create next generation
            next_population = []
            
            # Elitism: Keep best individuals
            elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
            for idx in elite_indices:
                next_population.append(population[idx])
            
            # Fill the rest through selection and crossover
            while len(next_population) < self.population_size:
                # Select parents
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                
                # Crossover
                if random.random() < self.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1, parent2
                
                # Mutation
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                
                # Add to next generation
                next_population.append(child1)
                if len(next_population) < self.population_size:
                    next_population.append(child2)
            
            # Update population
            population = next_population
        
        # Get selected features from best solution
        selected_features = [feature_names[i] for i in range(n_features) if best_solution[i] == 1]
        
        # Calculate feature importances based on selection frequency in final population
        importances = {name: 0.0 for name in feature_names}
        for individual in population:
            for i, val in enumerate(individual):
                if val == 1:
                    importances[feature_names[i]] += 1
        
        # Normalize importances
        max_imp = max(importances.values()) if importances else 1.0
        importances = {k: v / max_imp for k, v in importances.items()}
        
        # Ensure selected features have high importance
        for feat in selected_features:
            importances[feat] = max(importances[feat], 0.8)
        
        return selected_features, importances
    
    def _initialize_population(self, n_features: int) -> List[List[int]]:
        """Initialize a random population.
        
        Args:
            n_features: Number of features
            
        Returns:
            List: Population as list of binary feature masks
        """
        population = []
        
        # Create random individuals
        for _ in range(self.population_size):
            # Random binary vector
            individual = [random.randint(0, 1) for _ in range(n_features)]
            
            # Ensure at least one feature is selected
            if sum(individual) == 0:
                individual[random.randint(0, n_features - 1)] = 1
                
            population.append(individual)
        
        return population
    
    def _evaluate_feature_set(self, X: pd.DataFrame, y: pd.Series, 
                             features: List[str], model: Any, 
                             scoring: Union[str, Callable]) -> float:
        """Evaluate a feature set using cross-validation.
        
        Args:
            X: Input features
            y: Target variable
            features: Features to evaluate
            model: Model to use
            scoring: Scoring metric
            
        Returns:
            float: Mean cross-validation score
        """
        if not features:
            return -np.inf
            
        # Create a copy of the model to avoid fitting the same model multiple times
        model_copy = clone_model(model)
        
        try:
            # Evaluate using cross-validation
            scores = cross_val_score(
                model_copy, X[features], y, 
                cv=self.cv, scoring=scoring, n_jobs=self.n_jobs
            )
            return np.mean(scores)
        except Exception:
            # Return very low score on error
            return -np.inf
    
    def _tournament_selection(self, population: List[List[int]], fitness_scores: List[float]) -> List[int]:
        """Select an individual using tournament selection.
        
        Args:
            population: Population
            fitness_scores: Fitness scores for the population
            
        Returns:
            List: Selected individual
        """
        # Select tournament_size random individuals
        tournament_indices = random.sample(range(len(population)), min(self.tournament_size, len(population)))
        
        # Select the best from the tournament
        best_idx = tournament_indices[0]
        best_score = fitness_scores[best_idx]
        
        for idx in tournament_indices[1:]:
            if fitness_scores[idx] > best_score:
                best_idx = idx
                best_score = fitness_scores[idx]
        
        return population[best_idx]
    
    def _crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Perform crossover between two parents.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Tuple: Two children
        """
        n = len(parent1)
        
        # Select crossover point
        point = random.randint(1, n - 1)
        
        # Create children
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        
        return child1, child2
    
    def _mutate(self, individual: List[int]) -> List[int]:
        """Mutate an individual.
        
        Args:
            individual: Individual to mutate
            
        Returns:
            List: Mutated individual
        """
        mutated = individual.copy()
        
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                # Flip the bit
                mutated[i] = 1 - mutated[i]
        
        # Ensure at least one feature is selected
        if sum(mutated) == 0:
            mutated[random.randint(0, len(mutated) - 1)] = 1
            
        return mutated
    
    def _is_regression_task(self, y: pd.Series) -> bool:
        """Determine if this is a regression or classification task.
        
        Args:
            y: Target variable
            
        Returns:
            bool: True if regression, False if classification
        """
        unique_values = y.nunique()
        
        # Heuristic: if few unique values or object/category type, likely classification
        if unique_values <= 10 or y.dtype in ['object', 'category', 'bool']:
            return False
        
        # If float type with many unique values, likely regression
        if y.dtype.kind in 'fc' and unique_values > 10:
            return True
            
        # Default to classification
        return False 