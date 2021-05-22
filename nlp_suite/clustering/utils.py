from re import template
from ipywidgets.widgets.widget_layout import Layout
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from collections.abc import Iterable

def plot_3d_clusters(pca_proj, labels, max_points=10000, point_size=2, theme='plotly_dark'):
    '''
    Plot 3D pca projections on a plotly graph
    '''
    idx = np.random.choice(range(pca_proj.shape[0]), size=max_points, replace=False)

    fig = go.FigureWidget([go.Scatter3d(x=pca_proj[idx, 0],
                                 y=pca_proj[idx, 1],
                                 z=pca_proj[idx, 2], 
                                 mode='markers',
                                 marker=dict(size=point_size,
                                             color=labels[idx],
                                             colorscale='Rainbow'),
                                 text=[f'Cluster {l}' for l in labels[idx]],
                                 hoverinfo='text',
                                )], 
                            layout=go.Layout(template=theme)
                            )

    fig.layout.update(scene=dict(
                      xaxis=dict(showticklabels=False, title='', showspikes=False, showgrid=False),
                      yaxis=dict(showticklabels=False, title='', showspikes=False, showgrid=False),
                      zaxis=dict(showticklabels=False, title='', showspikes=False, showgrid=False),
                                ))


    return fig


def classify_text(vectorizer, clusters, text_docs):
    '''
    Classify documents of text given a learned vectorizer and cluster centers
    '''
    assert isinstance(vectorizer, TfidfVectorizer)
    assert isinstance(clusters, MiniBatchKMeans)
    text_docs = [text_docs] if isinstance(text_docs, str) else text_docs
    assert isinstance(text_docs, Iterable)

    return clusters.predict(vectorizer.transform(text_docs))
