import numpy as np
import sklearn.linear_model

def relu(x):
    return x * (x>0)

class ELM:
    def __init__(self, num_random_features, activation_function = relu, ridge_penalty = 0):
        self.num_random_features = num_random_features
        self.activation_function = activation_function
        self.ridge_penalty = ridge_penalty
    def random_features(self, X):
        augmented_feature_matrix = np.vstack([X.T,np.ones(X.shape[0])]).T
        return self.activation_function(augmented_feature_matrix @ self.random_weights.T)
    def fit(self, X, y):
        self.random_weights = np.random.normal(size = (self.num_random_features, X.shape[1] + 1))
        self.linear_model = sklearn.linear_model.Ridge(self.ridge_penalty)
        random_features = self.random_features(X)
        self.linear_model.fit(random_features, y)
        return self
    def predict(self, X):
        random_features = self.random_features(X)
        return self.linear_model.predict(random_features)
        
