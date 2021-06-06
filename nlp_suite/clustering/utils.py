from re import template
from ipywidgets.widgets.widget_layout import Layout
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from collections.abc import Iterable

def plot_3d_clusters(pca_proj, labels, max_points=10000, point_size=2, theme='plotly_dark'):
    '''Plot 3D pca projections on a plotly graph

    :param pca_proj: Numpy array consisting of 3D projections for each data point; Shape [N, 3]
    :type pca_proj: numpy.ndarray
    :param labels: Numpy array consisting of clustering label for each data point; Shape [N,]
    :type labels: numpy.ndarray
    :param max_points: Maximum number of points to plot, defaults to 10000
    :type max_points: int, optional
    :param point_size: Size of plotted points, defaults to 2
    :type point_size: int, optional
    :param theme: Plotly theme for generated plot, defaults to 'plotly_dark'
    :type theme: str, optional
    :return: Plotly FigureWidget object contatining a cluster plot
    :rtype: plotly.graph_objs._figurewidget.FigureWidget
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
    '''Classify documents of text given a learned vectorizer and cluster centers

    :param vectorizer: Vectorizer computed using sklearn used to encode text
    :type vectorizer: sklearn.feature_extraction.text.TfidfVectorizer
    :param clusters: K-Means clusters computed by sklearn
    :type clusters: sklearn.cluster._kmeans.MiniBatchKMeans
    :param text_docs: Iterable containing text fragments to be classified
    :type text_docs: collections.abc.Iterable
    :return: Numpy array of class predictions of the same length as text_docs
    :rtype: numpy.ndarray
    '''    
    assert isinstance(vectorizer, TfidfVectorizer)
    assert isinstance(clusters, MiniBatchKMeans)
    text_docs = [text_docs] if isinstance(text_docs, str) else text_docs
    assert isinstance(text_docs, Iterable)

    return clusters.predict(vectorizer.transform(text_docs))
