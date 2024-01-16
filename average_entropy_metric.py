import numpy as np

def average_entropy(df):
    """
    Calculate the average entropy for the clustering result. 

    Parameters:
    - df: A DataFrame with columns 'subject' and 'CLUSTER'.

    Returns:
    - custom_metric_value: The calculated custom evaluation metric.
    """

    # Calculate the number of observations in each cluster
    cluster_sizes = df['CLUSTER'].value_counts().sort_index()

    M = len(cluster_sizes)  # Total number of clusters
    N = len(df)
    
    weighted_entropy = 0.0

    for cluster_id, cluster_size in cluster_sizes.items():
        # Filter DataFrame for the current cluster
        cluster_df = df[df['CLUSTER'] == cluster_id]

        # Calculate the frequency of each subject in the cluster
        # These are the N_ij-values
        N_ij = cluster_df['main_subject'].value_counts().values

        # Calculate the relative frequency of each subject in the cluster
        pi = N_ij / cluster_size

        # Calculate the entropy for the cluster
        cluster_entropy = -np.sum(pi * np.log2(pi+ 1e-10))  # Add a small epsilon to avoid log(0)

        # Adjust for cluster size using the logarithm of cluster size. 
        # adjusted_cluster_entropy = cluster_entropy / np.log(Ni)

        # Accumulate the adjusted entropy for all clusters
        weighted_entropy += (cluster_size / N) * cluster_entropy

    # Calculate the average across all clusters
    weighted_entropy /= M

    return weighted_entropy