#!/usr/bin/env python3
import numpy as np
import ELM
import plotly.graph_objects as go
import plotly.express as px

def test_function(X):
    return np.cos(np.sum(X**2,axis=1))

if __name__ == "__main__":
    X_train, X_test = np.random.normal(size=(1000,2)), np.random.normal(size=(5000,2))
    y_train, y_test = test_function(X_train), test_function(X_test)

    model = ELM.ELM(1000,ridge_penalty = 10)
    model.fit(X_train, y_train)

    #Plotting
    x, y, z_act, z_pred = *X_test.T, y_test, model.predict(X_test)

    fig = px.scatter_3d(title = 'Actual vs. Predicted')
    fig.add_scatter3d(x=x,y=y,z=z_act,name='Actual',mode='markers',marker=dict(color='blue'))
    fig.add_scatter3d(x=x,y=y,z=z_pred,name='Predicted',mode='markers',marker=dict(color='red'))
    fig.show()

    fig = px.scatter_3d(title = 'Residuals')
    fig.add_scatter3d(x=x,y=y,z=z_pred-z_act,mode='markers',marker=dict(color='blue'))
    fig.show()
