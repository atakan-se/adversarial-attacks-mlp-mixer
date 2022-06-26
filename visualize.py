from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
from math import ceil

def graph_3d_norms_over_dataset(all_norms, patch_counts, figure_width=18):
    normalized_all_norms = (all_norms - torch.mean(all_norms, dim=2).unsqueeze(2))/torch.std(all_norms,dim=2).unsqueeze(2)
    sum_all_norms = torch.sum(normalized_all_norms, dim=0)

    ph,pw = (patch_counts, patch_counts) if isinstance(patch_counts, int) else patch_counts
    block_count = sum_all_norms.shape[0]
    fig_y_count = ceil(block_count/3) if block_count>3 else 1
    fig = plt.figure(figsize=(figure_width, fig_y_count * (figure_width//3) ))
    x = np.arange(0, pw)
    y = np.arange(0, ph)
    X,Y = np.meshgrid(x,y)

    for i, block in enumerate(sum_all_norms):
        block = block.reshape((ph,pw))
        ax = fig.add_subplot(fig_y_count, 3,  i+1, projection='3d')
        # Plot a 3D surface
        ax.plot_surface(X, Y, block.cpu().numpy())
        fig.show()

def graph_heatmap_norms_over_dataset(all_norms, patch_counts, figure_width=18):
    normalized_all_norms = (all_norms - torch.mean(all_norms, dim=2).unsqueeze(2))/torch.std(all_norms,dim=2).unsqueeze(2)
    sum_all_norms = torch.sum(normalized_all_norms, dim=0)

    ph,pw = (patch_counts, patch_counts) if isinstance(patch_counts, int) else patch_counts
    block_count = sum_all_norms.shape[0]
    fig_y_count = ceil(block_count/3) if block_count>3 else 1
    fig = plt.figure(figsize=(figure_width, fig_y_count * (figure_width//3) ))
    x = np.arange(0, pw)
    y = np.arange(0, ph)
    X,Y = np.meshgrid(x,y)

    for i, block in enumerate(sum_all_norms):
        block = block.reshape((ph,pw))
        ax = fig.add_subplot(fig_y_count, 3,  i+1)
        # Plot a 3D surface
        sns.heatmap(ax=ax, data=block)
        fig.show()
