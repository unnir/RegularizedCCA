# Regularized Canonical Correlation Analysis (rCCA)

## Overview
Regularized Canonical Correlation Analysis (rCCA) is a statistical method for finding linear relationships between two sets of variables. It extends the traditional Canonical Correlation Analysis (CCA) by adding regularization, making it more suitable for high-dimensional datasets or situations with multicollinearity.

## Features
- **Regularization**: Addresses overfitting and multicollinearity in high-dimensional data.
- **Flexibility**: Customizable regularization parameters for each dataset.
- **Scikit-learn Compatible**: Follows scikit-learn API conventions for easy integration into machine learning pipelines.

## Installation
Copy it to your code: 
```py
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.linalg import eigh

class RegularizedCCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2, regularization_x=0.1, regularization_y=0.1):
        self.n_components = n_components
        self.regularization_x = regularization_x
        self.regularization_y = regularization_y
        self.x_weights_ = None
        self.y_weights_ = None

    def fit(self, X, Y):
        n = X.shape[0]

        # Centering the data
        X = X - np.mean(X, axis=0)
        Y = Y - np.mean(Y, axis=0)

        # Compute covariance matrices
        C_xx = np.dot(X.T, X) / n + self.regularization_x * np.eye(X.shape[1])
        C_yy = np.dot(Y.T, Y) / n + self.regularization_y * np.eye(Y.shape[1])
        C_xy = np.dot(X.T, Y) / n

        # Solve the generalized eigenvalue problem
        inv_C_xx = np.linalg.inv(C_xx)
        inv_C_yy = np.linalg.inv(C_yy)
        R = np.dot(np.dot(inv_C_xx, C_xy), np.dot(inv_C_yy, C_xy.T))

        eigvals, eigvecs = eigh(R, eigvals=(X.shape[1] - self.n_components, X.shape[1] - 1))
        self.x_weights_ = np.dot(inv_C_xx, eigvecs)

        # Compute y_weights
        self.y_weights_ = np.dot(np.dot(inv_C_yy, C_xy.T), eigvecs)
        self.y_weights_ /= np.linalg.norm(self.y_weights_, axis=0)

        return self

    def transform(self, X, Y):
        X_transformed = np.dot(X - np.mean(X, axis=0), self.x_weights_)
        Y_transformed = np.dot(Y - np.mean(Y, axis=0), self.y_weights_)
        return X_transformed, Y_transformed
    
    def fit_transform(self, X, Y):
        self.fit(X, Y)
        return self.transform(X, Y)
```

## Usage 
```py
import numpy as np
from rcca import RegularizedCCA

# Initialize the rCCA model
rcca = RegularizedCCA(n_components=2, regularization_x=0.1, regularization_y=0.1)

# Fit the model and transform the datasets
X_c, Y_c = rcca.fit_transform(X, Y)

correlation = np.corrcoef(X_c.T, Y_c.T)
correlation
```


