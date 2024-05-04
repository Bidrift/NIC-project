import numpy as np
import pandas as pd

from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import FunctionTransformer
from sko import GA
import random

## Functions taken and inspired from GeeksForGeeks, for better experimenting
def get_name(x: pd.Series) -> tuple[np.array, str]:
    return x.values, x.name

def neutral(x: pd.Series) -> pd.DataFrame:
    values, col_name = get_name(x)
    return pd.DataFrame(data=values, columns = [f"{col_name}"])

def polynomial(x: pd.Series, degree: int) -> pd.DataFrame:
    values, col_name = get_name(x)
    data = values ** degree
    return pd.DataFrame(data=data, columns = [f"{col_name}**{degree}"])

def log(x: pd.Series) -> pd.DataFrame:
    values, col_name = get_name(x)
    ft = FunctionTransformer(func=np.log1p)
    data = np.sign(values) * ft.fit_transform(np.abs(values))
    return pd.DataFrame(data=data, columns=[f'log({col_name})'])
        
def reciprocal(x: pd.Series) -> pd.DataFrame:
    values, col_name = get_name(x)
    data = np.sign(values) / (1e-6 + np.abs(values))
    return pd.DataFrame(data=data, columns=[f'recip({col_name})'])

def sin(x: pd.Series) -> pd.DataFrame:
    values, col_name = get_name(x)
    data = np.sin(values)
    return pd.DataFrame(data=data, columns=[f'sin({col_name})'])

def square_root(x: pd.Series) -> pd.DataFrame:
    values, col_name = get_name(x)
    data = np.sign(values) * np.sqrt(np.abs(values))
    return pd.DataFrame(data=data, columns=[f'sqrt({col_name})'])

def cos(x: pd.Series) -> pd.DataFrame:
    values, col_name = get_name(x)
    data = np.cos(values)
    return pd.DataFrame(data=data, columns=[f'cos({col_name})'])

def tan(x: pd.Series) -> pd.DataFrame:
    values, col_name = get_name(x)
    data = np.tan(values)
    return pd.DataFrame(data=data, columns=[f'tan({col_name})'])
        
def box_cox(x: pd.Series) -> pd.DataFrame:
    values, col_name = get_name(x)
    pt = PowerTransformer(method='box-cox', standardize=False)
    transformed_data = np.abs(values.reshape(-1, 1) + 2)
    assert(transformed_data > 0).all()
    transformed_data = pt.fit_transform(np.abs(values.reshape(-1, 1) + 2))
    data = np.sign(values.reshape(-1, 1)) * transformed_data
    return pd.DataFrame(data=data, columns=[f'bc({col_name})'])

def yeo_johnson(x: pd.Series) -> pd.DataFrame:
    values, col_name = get_name(x)
    pt = PowerTransformer(method='yeo-johnson')
    transformed_data = pt.fit_transform(values.reshape(-1, 1))
    data = transformed_data
    return pd.DataFrame(data=data, columns=[f'yj({col_name})'])

def quantile_transformation(x: pd.Series) -> pd.DataFrame:
    values, col_name = get_name(x)
    pt = QuantileTransformer(random_state=0)
    transformed_data = pt.fit_transform(values.reshape(-1, 1))
    data = transformed_data
    return pd.DataFrame(data=data, columns=[f'qt({col_name})'])

def get_transformations(poly_degree:int) -> list:
    transformations = list()
    transformations.append(lambda x: tan(x))
    transformations.append(lambda x: square_root(x))
    transformations.append(lambda x: log(x))
    transformations.append(lambda x: yeo_johnson(x))
    transformations.append(lambda x: sin(x))
    transformations.append(lambda x: reciprocal(x))
    transformations.append(lambda x: cos(x))
    transformations.append(lambda x: box_cox(x))

    for degree in range(1, poly_degree + 1):
        transformations.append(lambda x, degree=degree: polynomial(x, degree))

    return transformations

class FeatureTransformation:
    def __init__(self, df: pd.DataFrame=None, y: np.array=None, population:int=20, max_iter:int=200, mutation_freq:float=0.05, n_poly:int=4, target_name:list=None) -> None:
        
        # Data loaders
        self.df = df
        self.y = y
        self.transformers = get_transformations(n_poly)
        if (target_name is None):
            self.target_name = 'y'
            
        # Parameters for GP from lab
        self.population = population
        self.max_iter = max_iter
        self.mutation_freq = mutation_freq
        
    def _get_mutation_pattern(self, value_function, transformers):
        while True:
            new_value = random.choices(list(range(len(transformers))), k=1)[0]
            if new_value != value_function:
                return new_value
    
    def _genic_mutation_chromosome(self, chromosome):
        pos = random.choices(list(range(len(chromosome))), k=1)[0]
        if (len(chromosome[pos]) <= 0):
            return chromosome
        gene_pos = random.choices(list(range(len(chromosome[pos]))), k=1)[0]
        chromosome[pos][gene_pos] = self._get_mutation_pattern(chromosome[pos][gene_pos], self.transformers)
        return chromosome
    
    def _chromosomic_mutation_chromosome(self, chromosome):
        pos = random.choices(list(range(len(chromosome))), k=1)[0]
        gene_size = random.randint(0, 4)
        chromosome[pos] = random.sample(range(0, len(self.transformers)), k=gene_size)
        return chromosome
    
    def _gp_mutation(self, generation):
        for i in range(len(generation)):
            if (np.random.rand() < self.mutation_freq):
                generation[i] = self._chromosomic_mutation_chromosome(generation[i])
    
        for i in range(len(generation)):
            if (np.random.rand() < self.mutation_freq and generation[i]):
                generation[i] = self._genic_mutation_chromosome(generation[i])
        return generation
        
            
    def _new_features(self, chromosome: np.array, df:pd.DataFrame=None):
        if (df is None):
            df = self.df
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(data=df, columns=range(df.shape[1]))
        
        # The chromosome is assumed to be a numpy array of size : number of features of the data field
        # iterate through the chromosome: each value maps to a function
        # apply this function on the corresponding column
        new_features = []
        
        for d, value_functions in zip(df.columns, chromosome):
            if (len(value_functions) == 0):
                new_features.append(neutral(df[d]))
                continue
            new_feature = self.transformers[int(value_functions[0])](df[d])
            for i in range(1, len(value_functions)):
                value_function = value_functions[i]
                new_feature = self.transformers[int(value_function)](new_feature[new_feature.columns[0]])
            new_features.append(new_feature)

        # Concatenate all the new features into a single dataframe
        all_data =  pd.concat(new_features, axis=1, ignore_index=False)
        
        return all_data
    
    def _get_correlation(self, chromosome) -> pd.Series:
        # Get the new features from the chromosome
        new_features = self._new_features(chromosome)
        # Add the target variable's values as a column to the "new_features" dataframe
        new_features[self.target_name] = self.y.copy()
        
        # Compute the correlation matrix (linear correlation)
        linear_corr = np.abs(new_features.corr()[self.target_name])
        
        return linear_corr
    
    def _gp_fitness_chromosome(self, chromosome):
        arr = chromosome
        linear_corr = self._get_correlation(arr)
        # the score is the reverse of the average score of the best "num_feats" new features
        return 1 / (linear_corr.mean())
    
    def _gp_sort_population_by_fitness(self):
        # Sorting the population based on fitness (lower is better)
        # your code here
        # implement sorting method to sort population based on fitness value
        # hint, use argsort or sort by lambda
        sorted_indices = np.argsort(self.fitness)
        sorted_population = [self.generation[i] for i in sorted_indices]
        sorted_fitness = [self.fitness[i] for i in sorted_indices]
        self.generation = sorted_population
        self.fitness = np.array(sorted_fitness)
    
    def _gp_fitness(self):
        self.fitness = np.zeros(self.population)
        self.probs = np.zeros(self.population)
        total = 0
        
        # Looping over all solutions computing the fitness for each solution (chromosome)
        for i in range(self.population):
            self.fitness[i] = self._gp_fitness_chromosome(self.generation[i])
            total += self.fitness[i]
        self._gp_sort_population_by_fitness()
        self.probs = self.fitness/total
        
    
    def _init_pop(self, dim:int):
        self.generation = list()
        for i in range(self.population):
            self.generation.append([])
            for j in range(dim):
                gene_size = random.randint(0, 4)
                self.generation[i].append(random.sample(range(0, len(self.transformers)), k=gene_size))
        
    # Crossover function
    def _gp_crossover(self, generation, probs):
        new_gen = list(generation[:len(generation)//2])
        while (len(new_gen) < len(generation)):
            # Choose fathers using roulette
            father1 = random.choices(generation, weights=probs, k=1)[0]
            father2 = random.choices(generation, weights=probs, k=1)[0]
            while True:
                if (father1 != father2):
                    break
                father2 = random.choices(generation, weights=probs, k=1)[0]
            # Exchange programs
            pos = random.choices(list(range(len(generation[0]))), k=1)[0]
            child1 = list()
            child2 = list()
            for i in range(0, pos):
                child1.append(father1[i])
                child2.append(father2[i])
            for i in range(pos, len(father1)):
                child1.append(father2[i])
                child2.append(father1[i])
            new_gen.append(child1)
            new_gen.append(child2)
        if (len(new_gen) > len(generation)):
            new_gen = new_gen[:len(generation)]
        return new_gen
        
                
    
    def fit(self, df:pd.DataFrame, y: np.array):
        num_feats = df.shape[1]
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(data=df, columns=range(num_feats))
        self.df, self.y = df, y
        
        self._init_pop(num_feats)
        self._gp_fitness()
        
        for _ in range(self.max_iter):
            self.generation = self._gp_crossover(self.generation, self.probs)
            self.generation = self._gp_mutation(self.generation)
            self._gp_fitness()
            
        
        self.x = self.generation[0]
    
    def transform(self, df:pd.DataFrame) -> pd.DataFrame:
        return self._new_features(self.x, df)
    
    def fit_transform(self, df:pd.DataFrame, y: np.array) -> pd.DataFrame:
        self.fit(df, y)
        return self.transform(df)
    
