import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import floor
import seaborn as sns
from scipy.stats import spearmanr
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sko.PSO import PSO
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import random

def display_heat_map(data: pd.DataFrame, title: str):
    plt.figure(figsize=(10, 10))
    sns.heatmap(data, linewidth=1, annot=True)
    plt.title(title)
    plt.show()
    
class MultiPSO:
    def set_data(self, df):
        self.df = df
        # calculate the spearmanr correlation of the dataframe's features
        corr = spearmanr(df).correlation
        # make sure it is symmetric
        corr = (corr + corr.T) / 2
        # fill the diagonal with 1s
        np.fill_diagonal(corr, 1)
        # transform the matrix to a dataframe that represents how similar each feature it is to another
        self.dist_matrix = pd.DataFrame(data= (1 - np.abs (corr)), columns=list(df.columns), index=list(df.columns))
        # have a dictionary mapping the column's order to its name
        self.columns_dict = dict(list(zip(range(len(df.columns)), df.columns)))
        # set the number of features for later reference
        self.num_feats = len(df.columns)
        # save the column names for later reference
        self.columns = list(df.columns) 
    
    def __init__(self, df: pd.DataFrame, max_iter = 200, vif = 2.5, epsilon = 0.1,
                 min_fraction=0.40, max_fraction=0.76, step=0.05) -> None:
        self.max_iter = max_iter
        self.df = df
        corr = spearmanr(df).correlation
        np.fill_diagonal((corr + corr.T) / 2, 1)
        self.num_features = len(df.columns)
        self.pso = None
        self.vif = vif
        self.epsilon = epsilon
        self.min_fraction = min_fraction
        self.max_fraction = max_fraction
        self.step = step

    def _calculate_vif(self, df: pd.DataFrame= None):
        if (df is None) :
            df = self.df
        vif = pd.DataFrame()
        vif['variables'] = self.df.columns
        vif['VIF'] = [variance_inflation_factor(df.values, i) for i in range(len(df.columns))]
        return vif.set_index('variables')
    
    def _get_clusters(self, particle: np.array):
        n = len(particle)
        discrete = np.array([int(x) for x in particle])
        cluster_feats = {}
        for i in range(n):
            if (discrete[i] not in cluster_feats):
                cluster_feats[discrete[i]] = []
            cluster_feats[discrete[i]].append(i)
        return cluster_feats
    
    def _calculate_score(self, cluster_feats: dict):
        cluster_names = {}
        for c, feats in cluster_feats.items():
            cluster_names[c] = [self.columns_dict[i] for i in feats]
        inner_cluster_score = 0
        for c, names in cluster_feats.items():
            inner_cluster_score += (1 + np.exp(self.dist_matrix.iloc[names, names].values.sum())) / np.log(len(names) + np.exp(1))
        return inner_cluster_score
    
    def _pso_function(self, particle: np.array):
        return self._calculate_score(self._get_clusters(particle))
    
    def _cluster_pso(self, num_clusters):
        pso_function = lambda x: self._pso_function(x)
        lower_bound = np.zeros(self.num_feats)
        upper_bound = np.full(shape=self.num_feats, fill_value=num_clusters, dtype="float")
        pso = PSO(func=pso_function, dim=self.num_feats, pop=15, max_iter=self.max_iter, lb=lower_bound,
                          ub=upper_bound, c1=1.5, c2=1.5)
        pso.run()
        x, y = pso.gbest_x, pso.gbest_y
        cluster_feats = self._get_clusters(x)
        return x, y
    
    def _find_best_cluster(self):
        best_score = np.inf
        best_x = None
        last_num_clusters = 0
        for fraction in np.arange(self.min_fraction, self.max_fraction, self.step):
            num_clusters = max(floor(fraction * self.num_feats), 3)
            if num_clusters == last_num_clusters:
                continue
            last_num_clusters = num_clusters
            x, y = self._cluster_pso(num_clusters)
            if y < best_score:
                best_score = y
                best_x = x
        return best_x
    
    def _get_new_df(self, best_particle):
        pca = PCA(n_components=1)
        clusters = self._get_clusters(best_particle)
        new_dfs = []
        for _, feats in clusters.items():
            new_feats = pd.DataFrame(data=pca.fit_transform(self.df.loc[:, feats]), index=list(self.df.index))
            new_dfs.append(new_feats)
        return pd.concat(new_dfs, axis=1, ignore_index=True)

    def eliminate_multicollinearity(self, df: pd.DataFrame):
        vif = self._calculate_vif(df)
        collinear = list(vif[vif['VIF'] >= self.vif].index)
        collinear_df = df.loc[:, collinear]
        non_collinear = [c for c in df.columns if c not in collinear]
        non_collinear_df = df.loc[:, non_collinear]
        if not collinear:
            return df
        self.set_data(collinear_df)
        best_x = self._find_best_cluster()
        new_collinear_df = self._get_new_df(best_x)
        return pd.concat([non_collinear_df, new_collinear_df], axis=1)
    

with open ("test.txt", "a") as f:
    solution_counter = 0
    improved_solution_counter = 0
    degraded_solution_counter = 0
    for i in range(20):
        features_1 = 0
        features_2 = 0
        features_removed = 0
        random_state = random.randint(0, 200)
        n_informative = random.randint(5, 14)
        n_redundant = random.randint(6, 12)
        X1, y = make_classification(n_samples=4000, n_features=25,n_informative=n_informative, n_redundant=n_redundant, n_repeated=0, shuffle=False, random_state=random_state)

        X1 = pd.DataFrame(data=X1, columns=range(X1.shape[1]))

        pso = MultiPSO(X1.copy())

        new_X1 = pso.eliminate_multicollinearity(X1.copy())
        
        X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y, test_size=0.2, random_state=11)
        features_1 = len(X_train1.columns)
        lr = LogisticRegression(max_iter=5000)
        lr.fit(X_train1, y_train1)
        y_pred1 = lr.predict(X_test1)
        a1 = accuracy_score(y_test1, y_pred1)
        
        X_train1, X_test1, y_train1, y_test1 = train_test_split(new_X1,y,  test_size=0.2, random_state=11)
        features_2 = len(X_train1.columns)
        features_removed += features_1 - features_2
        lr.fit(X_train1, y_train1)
        y_pred1 = lr.predict(X_test1)
        a2 = (accuracy_score(y_test1, y_pred1))
        print(f"a1:\t{a1}")
        print(f"a2:\t{a2}")
        
        f.write(f"test {i}\n")
        f.write(f"random state: {random_state}, inf feat: {n_informative}, red feat: {n_redundant}\n")
        f.write(f"accuracy with {features_1} features :{a1}\n")
        f.write(f"accuracy with {features_1-features_2} features removed:{a2}\n")
        f.write("\n")
        solution_counter += 1
        
        solution_difference = a2-a1
    
        if (a2 > a1):
            improved_solution_counter += 1
        elif (solution_difference < 0.001):
            degraded_solution_counter += 1
    features_removed = features_removed/solution_counter
    f.write(f"Among {solution_counter} tests we had {improved_solution_counter} improved solutions and {degraded_solution_counter} worse solutions")
    f.write(f"Apprx. {solution_difference} features was removed per test")
        
        
        