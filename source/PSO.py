import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from statsmodels.stats.outliers_influence import variance_inflation_factor

def display_heat_map(data: pd.DataFrame, title: str):
    plt.figure(figsize=(10, 10))
    sns.heatmap(data, linewidth=1, annot=True)
    plt.title(title)
    plt.show()
    
class PSO:
    def __init__(self, df: pd.DataFrame, max_iter:int, vif:float, epsilon:int,
                 min_fraction, max_fraction, step) -> None:
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

    def get_vif(self):
        vif = pd.DataFrame()
        vif['variables'] = self.df.columns
        for i in range(self.num_features):
            vif['VIF'] = [variance_inflation_factor(self.df.values, i)]
        return vif.set_index('variables')
    
    def get_clusters(self, particle: np.array):
        n = len(particle)
        discrete = np.array([int(x) for x in particle])
        cluster_feats = {}
        for i in range(n):
            if (discrete[i] not in cluster_feats):
                cluster_feats[discrete[i]] = []
            cluster_feats[discrete[i]].append(i)
        return cluster_feats
    
    def calculate_score(self, particle: np.array):
        clusters = {}
        new_order = []
        title = ""
        
        
        
        