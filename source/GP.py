import numpy as np
import pandas as pd

from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer
from sko import GA
import random

def get_name(x: pd.Series) -> tuple[np.array, str]:
    return x.values, x.name

def polynomial(x: pd.Series, degree: int) -> pd.DataFrame:
    values, col_name = get_name(x)
    data = values ** degree
    return pd.DataFrame(data=data, columns = [f"{col_name}**{degree}"])

def square_root(x: pd.Series) -> pd.DataFrame:
    values, col_name = get_name(x)
    data = np.sign(values) * np.sqrt(np.abs(values))
    return pd.DataFrame(data=data, columns=[f'sqrt({col_name})'])
        
def reciprocal(x: pd.Series) -> pd.DataFrame:
    values, col_name = get_name(x)
    data = np.sign(values) / (1e-6 + np.abs(values))
    return pd.DataFrame(data=data, columns=[f'recip({col_name})'])
        
def box_cox(x: pd.Series) -> pd.DataFrame:
    values, col_name = get_name(x)
    pt = PowerTransformer(method='box-cox', standardize=False)
    transformed_data = pt.fit_transform(np.abs(values).reshape(-1, 1))
    data = np.sign(values).reshape(-1, 1) * transformed_data
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
    transformations = [lambda x: square_root(x),
              lambda x: reciprocal(x),
              lambda x: box_cox(x),
              lambda x: yeo_johnson(x)]

    for degree in range(1, poly_degree + 1):
        transformations.append(lambda x, degree=degree: polynomial(x, degree))

    return transformations

print(get_transformations(5))

class FeatureTransformation:
    def __init__(self, population=10, max_iter=200, mutation_freq=0.05, df: pd.DataFrame=None, y: np.array=None, n_poly:int=4, target_names:list=None) -> None:
        
        # Data loaders
        self.df = df
        self.y = y
        self.transformers = get_transformations(n_poly)
        if (target_names is None):
            self.target_names = ['y', 'target', 'dependent_variable']
            
        # Parameters for GP from lab
        self.population = population
        self.max_iter = max_iter
        self.mutation_freq = mutation_freq
        
    def _get_mutation_pattern(self, value_function, transformers):
        if value_function < 4:
            while True:
                new_value = random.choices(list(range(4)), k=1)[0]
                if new_value != value_function:
                    return new_value
        prob = random.random()
        if prob < 0.5:
            value_function += 1
        else:
            value_function -= 1
            value_function % (len(transformers) - 4)
        return 4 + value_function
    
    def _mutation_chromosome(self, chromosome):
        pos = random.choices(list(range(len(chromosome))), k=1)[0]
        chromosome[pos] = self._get_mutation_pattern(chromosome[pos], self.transformers)
        return chromosome
    
    def _gp_mutation(self, generation):
        for i in range(generation.size_pop):
            if (np.random.rand() < self.mutation_freq):
                generation.Chrom[i] = self._mutation_chromosome(generation.Chrom[i])
        return generation.Chrom
    
    def _find_target_name(self):
        for name in self.target_names:
            if name not in self.df.columns:
                target = name
                return target
            
    def _new_features(self, chromosome: np.array, df:pd.DataFrame=None):
        if df is None:
            df = self.df

        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(data=df, columns=range(df.shape[1]))
        
        # The chromosome is assumed to be a numpy array of size : number of features of the data field
        # iterate through the chromosome: each value maps to a function
        # apply this function on the corresponding column
        new_features = [self.transformers[int(value_function)](df[d]) for d, value_function in zip(df.columns, chromosome)]

        # Concatenate all the new features into a single dataframe
        all_data =  pd.concat(new_features, axis=1, ignore_index=False)
        
        return all_data
    
    def _get_correlation(self, chromosome) -> pd.Series:
        # Get the new features from the chromosome
        new_features = self._new_features(chromosome)
        
        # Retrieve the target name
        target_name = self._find_target_name()
        
        # Add the target variable's values as a column to the "new_features" dataframe
        new_features[target_name] = self.y.copy()
        
        # Compute the correlation matrix (linear correlation)
        linear_corr = np.abs(new_features.corr()[target_name])

        # Order the columns by their correlation to the target
        linear_corr.sort_values(ascending=False, inplace=True)
        linear_corr.drop(target_name, inplace=True)
        return linear_corr
    
    def _gp_fitness(self, chromosome: np.array):
        linear_corr = self._get_correlation(chromosome)
        # the score is the reverse of the average score of the best "num_feats" new features
        return 1 / (linear_corr.mean())
    
    def fit(self, df:pd.DataFrame, y: np.array):
        num_feats = df.shape[1]
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(data=df, columns=range(num_feats))
        self.df, self.y = df, y
        
        gp_function = lambda x: self._gp_fitness(x)
        
        # Define the lower and upper bounds for the chromosomes
        lower_bound = np.zeros(num_feats)
        upper_bound = np.full(shape=(num_feats, ), fill_value=len(self.transformers) - 1)
        # Define the precision so that values in chromosome objects are integers
        precision = np.full(shape=(num_feats, ), fill_value=1)
        
        gp = GA.GA(func=gp_function, n_dim=num_feats, size_pop=self.population, max_iter=self.max_iter, prob_mut=self.mutation_freq, lb=lower_bound, ub=upper_bound, precision=precision)
        
        gp.register(operator_name='mutation', operator=lambda x: self._gp_mutation(x))
        
        x, y = gp.run()
        self.x = x
    
    def transform(self, df:pd.DataFrame) -> pd.DataFrame:
        return self._new_features(self.x, df)
    
    def fit_transform(self, df:pd.DataFrame, y: np.array) -> pd.DataFrame:
        self.fit(df, y)
        return self.transform(df)
