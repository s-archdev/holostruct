"""
Comprehensive visualization suite for Mistral 4B model internals.
Provides plotting functions for weights, activations, attention patterns, and embeddings.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class WeightVisualizer:
    """Visualize weight distributions across model layers."""
    
    def __init__(self, data: Dict[str, Any]):
        self.data = data
        self.weights = data.get('weights', {})
        self.metadata = data.get('metadata', {})
    
    def plot_weight_distributions(self, figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """Plot histogram of weight distributions for all layers."""
        if not self.weights:
            raise ValueError("No weight data available")
        
        n_layers = len(self.weights)
        n_cols = 4
        n_rows = (n_layers + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_layers > 1 else [axes]
        
        layer_indices = sorted(self.weights.keys())
        
        for i, layer_idx in enumerate(layer_indices):
            layer_weights = self.weights[layer_idx]
            
            # Combine all weights in this layer
            all_weights = []
            for weight_name, weight_tensor in layer_weights.items():
                all_weights.extend(weight_tensor.flatten())
            
            all_weights = np.array(all_weights)
            
            axes[i].hist(all_weights, bins=50, alpha=0.7, density=True)
            axes[i].set_title(f'Layer {layer_idx}')
            axes[i].set_xlabel('Weight Value')
            axes[i].set_ylabel('Density')
            
            # Add statistics
            mean_val = np.mean(all_weights)
            std_val = np.std(all_weights)
            axes[i].axvline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'μ={mean_val:.3f}')
            axes[i].axvline(mean_val + std_val, color='orange', linestyle=':', alpha=0.7)
            axes[i].axvline(mean_val - std_val, color='orange', linestyle=':', alpha=0.7, label=f'σ={std_val:.3f}')
            axes[i].legend()
        
        # Hide unused subplots
        for i in range(len(layer_indices), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.suptitle('Weight Distributions Across Layers', y=1.02, fontsize=16)
        return fig
    
    def plot_weight_statistics(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """Plot weight statistics (mean, std, min, max) across layers."""
        if not self.weights:
            raise ValueError("No weight data available")
        
        layer_indices = sorted(self.weights.keys())
        stats = {'layer': [], 'mean': [], 'std': [], 'min': [], 'max': [], 'component': []}
        
        for layer_idx in layer_indices:
            layer_weights = self.weights[layer_idx]
            
            for weight_name, weight_tensor in layer_weights.items():
                flat_weights = weight_tensor.flatten()
                stats['layer'].append(layer_idx)
                stats['mean'].append(np.mean(flat_weights))
                stats['std'].append(np.std(flat_weights))
                stats['min'].append(np.min(flat_weights))
                stats['max'].append(np.max(flat_weights))
                stats['component'].append(weight_name)
        
        df = pd.DataFrame(stats)
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Mean values
        axes[0, 0].scatter(df['layer'], df['mean'], alpha=0.6, c=df['component'].astype('category').cat.codes)
        axes[0, 0].set_title('Weight Means by Layer')
        axes[0, 0].set_xlabel('Layer')
        axes[0, 0].set_ylabel('Mean Weight')
        
        # Standard deviations
        axes[0, 1].scatter(df['layer'], df['std'], alpha=0.6, c=df['component'].astype('category').cat.codes)
        axes[0, 1].set_title('Weight Standard Deviations by Layer')
        axes[0, 1].set_xlabel('Layer')
        axes[0, 1].set_ylabel('Weight Std')
        
        # Range (max - min)
        df['range'] = df['max'] - df['min']
        axes[1, 0].scatter(df['layer'], df['range'], alpha=0.6, c=df['component'].astype('category').cat.codes)
        axes[1, 0].set_title('Weight Ranges by Layer')
        axes[1, 0].set_xlabel('Layer')
        axes[1, 0].set_ylabel('Weight Range')
        
        # Component comparison
        component_means = df.groupby('component')['mean'].mean().sort_values()
        axes[1, 1].barh(range(len(component_means)), component_means.values)
        axes[1, 1].set_yticks(range(len(component_means)))
        axes[1, 1].set_yticklabels(component_means.index, rotation=0)
        axes[1, 1].set_title('Average Weight by Component')
        axes[1, 1].set_xlabel('Mean Weight')
        
        plt.tight_layout()
        return fig

class ActivationVisualizer:
    """Visualize activation patterns across layers and tokens."""
    
    def __init__(self, data: Dict[str, Any]):
        self.data = data
        self.activations = data.get('activations', {})
        self.metadata = data.get('metadata', {})
    
    def plot_activation_heatmap(self, layer_idx: int, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """Plot activation heatmap for a specific layer."""
        if layer_idx not in self.activations:
            raise ValueError(f"No activation data for layer {layer_idx}")
        
        layer_acts = self.activations[layer_idx]
        token_indices = sorted(layer_acts.keys())
        
        # Create activation matrix [tokens, hidden_dim]
        activation_matrix = np.array([layer_acts[token_idx] for token_idx in token_indices])
        