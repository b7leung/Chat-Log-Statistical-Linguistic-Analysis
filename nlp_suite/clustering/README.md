# Introduction

This directory contains all utilities and notebooks for running cluster analysis. Follow the steps in this file to properly generate the required files.

# Setup

Follow these steps to generate the clustering data.

1. Download the dataset: https://www.kaggle.com/jef1056/discord-data (Only v1 used in uploaded example)

2. Use the requirements.txt file to ensure you have the appropriate python modules

3. In the clustering directoy, run the following command: ``python preprocessing.py --datapath <path to downloaded data>``
   You may also specify a save directory, name of the fine, and minimum number of sentences required: ``--outpath``, ``--fname``, ``--minchats``
   This will output a csv file with all the extracted data.

4. In the clustering directoy, run the following command: ``python compute_tfidf.py --datapath <path csv file generated in step 3>``
   You may also specify minimum word frequency ``--minfreq``, maximum number of features ``--maxfeat``, and name of the output file ``--fname``
   This will output a pkl file containing a clean dataframe, vectorizer object, and encodings of the training data

5. In the clustering directoy, run the following command: ``python compute_clusters.py --datapath <path pkl file generated in step 4>``
   **This file is already included for v1 directory in discord data** 
   You may also specify number of clusters ``--numclusters``, and output file name ``--fname``
   This will output a pkl file containing the cluster data, labels for the training data, and 3D PCA projections of the encodings

6. Run the cells in Clustering.ipynb to generate the visualizations


