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
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Full activation heatmap
        im1 = ax1.imshow(activation_matrix, aspect='auto', cmap='RdBu_r', interpolation='nearest')
        ax1.set_title(f'Layer {layer_idx} Activations')
        ax1.set_xlabel('Hidden Dimension')
        ax1.set_ylabel('Token Position')
        plt.colorbar(im1, ax=ax1)
        
        # Activation statistics per token
        token_means = np.mean(activation_matrix, axis=1)
        token_stds = np.std(activation_matrix, axis=1)
        
        ax2.plot(token_indices, token_means, 'b-', label='Mean', linewidth=2)
        ax2.fill_between(token_indices, 
                        token_means - token_stds, 
                        token_means + token_stds, 
                        alpha=0.3, label='±1 Std')
        ax2.set_title(f'Layer {layer_idx} Activation Statistics')
        ax2.set_xlabel('Token Position')
        ax2.set_ylabel('Activation Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_activation_evolution(self, figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """Plot how activations evolve across layers."""
        if not self.activations:
            raise ValueError("No activation data available")
        
        layer_indices = sorted(self.activations.keys())
        
        # Calculate statistics for each layer
        layer_stats = []
        for layer_idx in layer_indices:
            layer_acts = self.activations[layer_idx]
            all_activations = np.concatenate([acts for acts in layer_acts.values()])
            
            stats = {
                'layer': layer_idx,
                'mean': np.mean(all_activations),
                'std': np.std(all_activations),
                'min': np.min(all_activations),
                'max': np.max(all_activations),
                'sparsity': np.mean(np.abs(all_activations) < 0.01)  # Fraction near zero
            }
            layer_stats.append(stats)
        
        df = pd.DataFrame(layer_stats)
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Mean activation evolution
        axes[0, 0].plot(df['layer'], df['mean'], 'bo-', linewidth=2, markersize=6)
        axes[0, 0].set_title('Mean Activation by Layer')
        axes[0, 0].set_xlabel('Layer')
        axes[0, 0].set_ylabel('Mean Activation')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Standard deviation evolution
        axes[0, 1].plot(df['layer'], df['std'], 'ro-', linewidth=2, markersize=6)
        axes[0, 1].set_title('Activation Std by Layer')
        axes[0, 1].set_xlabel('Layer')
        axes[0, 1].set_ylabel('Activation Std')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Range evolution
        axes[1, 0].fill_between(df['layer'], df['min'], df['max'], alpha=0.3, color='green')
        axes[1, 0].plot(df['layer'], df['min'], 'g-', label='Min')
        axes[1, 0].plot(df['layer'], df['max'], 'g-', label='Max')
        axes[1, 0].set_title('Activation Range by Layer')
        axes[1, 0].set_xlabel('Layer')
        axes[1, 0].set_ylabel('Activation Value')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Sparsity evolution
        axes[1, 1].plot(df['layer'], df['sparsity'], 'mo-', linewidth=2, markersize=6)
        axes[1, 1].set_title('Activation Sparsity by Layer')
        axes[1, 1].set_xlabel('Layer')
        axes[1, 1].set_ylabel('Fraction Near Zero')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

class AttentionVisualizer:
    """Visualize attention patterns and head behaviors."""
    
    def __init__(self, data: Dict[str, Any]):
        self.data = data
        self.attention = data.get('attention', {})
        self.metadata = data.get('metadata', {})
    
    def plot_attention_heatmap(self, layer_idx: int, head_idx: int = 0, figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """Plot attention heatmap for specific layer and head."""
        if layer_idx not in self.attention:
            raise ValueError(f"No attention data for layer {layer_idx}")
        
        attention_scores = self.attention[layer_idx]  # [n_heads, seq_len, seq_len]
        
        if head_idx >= attention_scores.shape[0]:
            raise ValueError(f"Head {head_idx} not available. Max head index: {attention_scores.shape[0]-1}")
        
        head_attention = attention_scores[head_idx]  # [seq_len, seq_len]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(head_attention, cmap='Blues', interpolation='nearest')
        ax.set_title(f'Attention Pattern - Layer {layer_idx}, Head {head_idx}')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Attention Weight')
        
        # Add grid for better readability
        ax.set_xticks(range(0, head_attention.shape[1], max(1, head_attention.shape[1]//10)))
        ax.set_yticks(range(0, head_attention.shape[0], max(1, head_attention.shape[0]//10)))
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_attention_head_comparison(self, layer_idx: int, figsize: Tuple[int, int] = (16, 12)) -> plt.Figure:
        """Compare attention patterns across all heads in a layer."""
        if layer_idx not in self.attention:
            raise ValueError(f"No attention data for layer {layer_idx}")
        
        attention_scores = self.attention[layer_idx]  # [n_heads, seq_len, seq_len]
        n_heads = attention_scores.shape[0]
        
        n_cols = 4
        n_rows = (n_heads + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_heads > 1 else [axes]
        
        for head_idx in range(n_heads):
            head_attention = attention_scores[head_idx]
            
            im = axes[head_idx].imshow(head_attention, cmap='Blues', interpolation='nearest')
            axes[head_idx].set_title(f'Head {head_idx}')
            axes[head_idx].set_xlabel('Key')
            axes[head_idx].set_ylabel('Query')
        
        # Hide unused subplots
        for i in range(n_heads, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.suptitle(f'Attention Heads Comparison - Layer {layer_idx}', y=1.02, fontsize=16)
        return fig
    
    def plot_attention_statistics(self, figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """Plot attention statistics across layers and heads."""
        if not self.attention:
            raise ValueError("No attention data available")
        
        layer_indices = sorted(self.attention.keys())
        
        # Calculate statistics
        stats = []
        for layer_idx in layer_indices:
            attention_scores = self.attention[layer_idx]
            n_heads = attention_scores.shape[0]
            
            for head_idx in range(n_heads):
                head_attention = attention_scores[head_idx]
                
                # Calculate various attention metrics
                entropy = -np.sum(head_attention * np.log(head_attention + 1e-8), axis=-1).mean()
                max_attention = np.max(head_attention, axis=-1).mean()
                attention_spread = np.std(head_attention, axis=-1).mean()
                
                stats.append({
                    'layer': layer_idx,
                    'head': head_idx,
                    'entropy': entropy,
                    'max_attention': max_attention,
                    'spread': attention_spread
                })
        
        df = pd.DataFrame(stats)
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Entropy by layer (averaged across heads)
        entropy_by_layer = df.groupby('layer')['entropy'].mean()
        axes[0, 0].plot(entropy_by_layer.index, entropy_by_layer.values, 'bo-', linewidth=2)
        axes[0, 0].set_title('Attention Entropy by Layer')
        axes[0, 0].set_xlabel('Layer')
        axes[0, 0].set_ylabel('Average Entropy')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Max attention by layer
        max_att_by_layer = df.groupby('layer')['max_attention'].mean()
        axes[0, 1].plot(max_att_by_layer.index, max_att_by_layer.values, 'ro-', linewidth=2)
        axes[0, 1].set_title('Max Attention by Layer')
        axes[0, 1].set_xlabel('Layer')
        axes[0, 1].set_ylabel('Average Max Attention')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Attention spread by layer
        spread_by_layer = df.groupby('layer')['spread'].mean()
        axes[1, 0].plot(spread_by_layer.index, spread_by_layer.values, 'go-', linewidth=2)
        axes[1, 0].set_title('Attention Spread by Layer')
        axes[1, 0].set_xlabel('Layer')
        axes[1, 0].set_ylabel('Average Spread')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Head diversity within layers
        for layer_idx in layer_indices[::4]:  # Sample every 4th layer
            layer_data = df[df['layer'] == layer_idx]
            axes[1, 1].scatter([layer_idx] * len(layer_data), layer_data['entropy'], alpha=0.6)
        
        axes[1, 1].set_title('Head Entropy Diversity')
        axes[1, 1].set_xlabel('Layer')
        axes[1, 1].set_ylabel('Head Entropy')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

class EmbeddingVisualizer:
    """Visualize embedding spaces and token representations."""
    
    def __init__(self, data: Dict[str, Any]):
        self.data = data
        self.embeddings = data.get('embeddings', {})
        self.metadata = data.get('metadata', {})
    
    def plot_embedding_pca(self, embedding_type: str = 'input_embeddings', figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """Plot PCA projection of embeddings."""
        if embedding_type not in self.embeddings:
            raise ValueError(f"Embedding type {embedding_type} not available")
        
        embeddings = self.embeddings[embedding_type]
        
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        projected = pca.fit_transform(embeddings)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # PCA projection
        scatter = ax1.scatter(projected[:, 0], projected[:, 1], 
                            c=range(len(projected)), cmap='viridis', alpha=0.7)
        ax1.set_title(f'PCA Projection - {embedding_type}')
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.colorbar(scatter, ax=ax1, label='Token Position')
        
        # Explained variance
        n_components = min(10, embeddings.shape[1])
        pca_full = PCA(n_components=n_components)
        pca_full.fit(embeddings)
        
        ax2.bar(range(1, n_components + 1), pca_full.explained_variance_ratio_)
        ax2.set_title('PCA Explained Variance')
        ax2.set_xlabel('Principal Component')
        ax2.set_ylabel('Explained Variance Ratio')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_embedding_tsne(self, embedding_type: str = 'input_embeddings', figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """Plot t-SNE projection of embeddings."""
        if embedding_type not in self.embeddings:
            raise ValueError(f"Embedding type {embedding_type} not available")
        
        embeddings = self.embeddings[embedding_type]
        
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
        projected = tsne.fit_transform(embeddings)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        scatter = ax.scatter(projected[:, 0], projected[:, 1], 
                           c=range(len(projected)), cmap='viridis', alpha=0.7, s=50)
        ax.set_title(f't-SNE Projection - {embedding_type}')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        plt.colorbar(scatter, ax=ax, label='Token Position')
        
        # Add token position annotations for small sequences
        if len(projected) <= 20:
            for i, (x, y) in enumerate(projected):
                ax.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points', 
                           fontsize=8, alpha=0.7)
        
        plt.tight_layout()
        return fig

class InteractiveVisualizer:
    """Create interactive visualizations using Plotly."""
    
    def __init__(self, data: Dict[str, Any]):
        self.data = data
        self.weights = data.get('weights', {})
        self.activations = data.get('activations', {})
        self.attention = data.get('attention', {})
        self.embeddings = data.get('embeddings', {})
        self.metadata = data.get('metadata', {})
    
    def create_interactive_attention_viz(self, layer_idx: int) -> go.Figure:
        """Create interactive attention visualization."""
        if layer_idx not in self.attention:
            raise ValueError(f"No attention data for layer {layer_idx}")
        
        attention_scores = self.attention[layer_idx]
        n_heads, seq_len, _ = attention_scores.shape
        
        # Create subplots for each head
        fig = make_subplots(
            rows=(n_heads + 3) // 4, cols=4,
            subplot_titles=[f'Head {i}' for i in range(n_heads)],
            specs=[[{'type': 'heatmap'} for _ in range(4)] for _ in range((n_heads + 3) // 4)]
        )
        
        for head_idx in range(n_heads):
            row = head_idx // 4 + 1
            col = head_idx % 4 + 1
            
            head_attention = attention_scores[head_idx]
            
            heatmap = go.Heatmap(
                z=head_attention,
                colorscale='Blues',
                showscale=(head_idx == 0),
                hovertemplate='Query: %{y}<br>Key: %{x}<br>Attention: %{z:.3f}<extra></extra>'
            )
            
            fig.add_trace(heatmap, row=row, col=col)
        
        fig.update_layout(
            title=f'Interactive Attention Patterns - Layer {layer_idx}',
            height=300 * ((n_heads + 3) // 4)
        )
        
        return fig
    
    def create_3d_embedding_viz(self, embedding_type: str = 'input_embeddings') -> go.Figure:
        """Create 3D embedding visualization."""
        if embedding_type not in self.embeddings:
            raise ValueError(f"Embedding type {embedding_type} not available")
        
        embeddings = self.embeddings[embedding_type]
        
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        projected = pca.fit_transform(embeddings)
        
        fig = go.Figure(data=[go.Scatter3d(
            x=projected[:, 0],
            y=projected[:, 1],
            z=projected[:, 2],
            mode='markers+text',
            marker=dict(
                size=8,
                color=range(len(projected)),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Token Position")
            ),
            text=[f'Token {i}' for i in range(len(projected))],
            textposition="top center",
            hovertemplate='Token %{text}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>PC3: %{z:.3f}<extra></extra>'
        )])
        
        fig.update_layout(
            title=f'3D PCA Projection - {embedding_type}',
            scene=dict(
                xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.2%})',
                yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.2%})',
                zaxis_title=f'PC3 ({pca.explained_variance_ratio_[2]:.2%})'
            )
        )
        
        return fig

# Utility class for comprehensive analysis
class ModelAnalyzer:
    """High-level interface for comprehensive model analysis."""
    
    def __init__(self, data_path: str):
        from patched_llama import load_captured_data
        self.data = load_captured_data(data_path)
        
        self.weight_viz = WeightVisualizer(self.data)
        self.activation_viz = ActivationVisualizer(self.data)
        self.attention_viz = AttentionVisualizer(self.data)
        self.embedding_viz = EmbeddingVisualizer(self.data)
        self.interactive_viz = InteractiveVisualizer(self.data)
    
    def generate_all_plots(self, output_dir: str = "./plots", save_format: str = 'png'):
        """Generate all visualization plots and save them."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        plots_generated = []
        
        try:
            # Weight visualizations
            if self.data.get('weights'):
                fig = self.weight_viz.plot_weight_distributions()
                fig.savefig(f"{output_dir}/weight_distributions.{save_format}", dpi=300, bbox_inches='tight')
                plt.close(fig)
                plots_generated.append("weight_distributions")
                
                fig = self.weight_viz.plot_weight_statistics()
                fig.savefig(f"{output_dir}/weight_statistics.{save_format}", dpi=300, bbox_inches='tight')
                plt.close(fig)
                plots_generated.append("weight_statistics")
        except Exception as e:
            print(f"Error generating weight plots: {e}")
        
        try:
            # Activation visualizations
            if self.data.get('activations'):
                layer_indices = sorted(self.data['activations'].keys())
                
                # Plot first layer as example
                if layer_indices:
                    fig = self.activation_viz.plot_activation_heatmap(layer_indices[0])
                    fig.savefig(f"{output_dir}/activation_heatmap_layer_{layer_indices[0]}.{save_format}", dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    plots_generated.append(f"activation_heatmap_layer_{layer_indices[0]}")
                
                fig = self.activation_viz.plot_activation_evolution()
                fig.savefig(f"{output_dir}/activation_evolution.{save_format}", dpi=300, bbox_inches='tight')
                plt.close(fig)
                plots_generated.append("activation_evolution")
        except Exception as e:
            print(f"Error generating activation plots: {e}")
        
        try:
            # Attention visualizations
            if self.data.get('attention'):
                layer_indices = sorted(self.data['attention'].keys())
                
                if layer_indices:
                    # Plot first layer attention
                    fig = self.attention_viz.plot_attention_heatmap(layer_indices[0])
                    fig.savefig(f"{output_dir}/attention_heatmap_layer_{layer_indices[0]}.{save_format}", dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    plots_generated.append(f"attention_heatmap_layer_{layer_indices[0]}")
                    
                    fig = self.attention_viz.plot_attention_head_comparison(layer_indices[0])
                    fig.savefig(f"{output_dir}/attention_heads_layer_{layer_indices[0]}.{save_format}", dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    plots_generated.append(f"attention_heads_layer_{layer_indices[0]}")
                
                fig = self.attention_viz.plot_attention_statistics()
                fig.savefig(f"{output_dir}/attention_statistics.{save_format}", dpi=300, bbox_inches='tight')
                plt.close(fig)
                plots_generated.append("attention_statistics")
        except Exception as e:
            print(f"Error generating attention plots: {e}")
        
        try:
            # Embedding visualizations
            if self.data.get('embeddings'):
                for emb_type in self.data['embeddings'].keys():
                    fig = self.embedding_viz.plot_embedding_pca(emb_type)
                    fig.savefig(f"{output_dir}/embedding_pca_{emb_type}.{save_format}", dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    plots_generated.append(f"embedding_pca_{emb_type}")
                    
                    if len(self.data['embeddings'][emb_type]) > 3:  # t-SNE needs at least 4 points
                        fig = self.embedding_viz.plot_embedding_tsne(emb_type)
                        fig.savefig(f"{output_dir}/embedding_tsne_{emb_type}.{save_format}", dpi=300, bbox_inches='tight')
                        plt.close(fig)
                        plots_generated.append(f"embedding_tsne_{emb_type}")
        except Exception as e:
            print(f"Error generating embedding plots: {e}")
        
        return plots_generated
    
    def print_summary(self):
        """Print a summary of the captured data."""
        print("=== Model Analysis Summary ===")
        
        # Metadata
        if self.data.get('metadata'):
            print(f"Prompt: {self.data['metadata'].get('prompt', 'N/A')}")
            print(f"Model layers: {self.data['metadata'].get('n_layers', 'N/A')}")
            print(f"Hidden dim: {self.data['metadata'].get('n_embd', 'N/A')}")
            print(f"Attention heads: {self.data['metadata'].get('n_head', 'N/A')}")
        
        # Data availability
        print(f"\nData captured:")
        print(f"- Weights: {len(self.data.get('weights', {}))} layers")
        print(f"- Activations: {len(self.data.get('activations', {}))} layers")
        print(f"- Attention: {len(self.data.get('attention', {}))} layers")
        print(f"- Embeddings: {list(self.data.get('embeddings', {}).keys())}")
        
        # Size estimates
        total_size = 0
        for category in ['weights', 'activations', 'attention', 'embeddings']:
            category_data = self.data.get(category, {})
            if isinstance(category_data, dict):
                for key, value in category_data.items():
                    if isinstance(value, dict):
                        for subvalue in value.values():
                            if hasattr(subvalue, 'nbytes'):
                                total_size += subvalue.nbytes
                    elif hasattr(value, 'nbytes'):
                        total_size += value.nbytes
        
        print(f"\nTotal data size: ~{total_size / (1024**2):.1f} MB")

# Example usage
if __name__ == "__main__":
    # Example of how to use the visualization suite
    
    # Load data from a capture file
    data_path = "./test_dumps/capture_20241201_120000.h5"  # Replace with actual path
    
    try:
        analyzer = ModelAnalyzer(data_path)
        
        # Print summary
        analyzer.print_summary()
        
        # Generate all plots
        plots = analyzer.generate_all_plots("./example_plots")
        print(f"\nGenerated plots: {plots}")
        
        # Create some interactive visualizations
        if analyzer.data.get('attention'):
            layer_idx = sorted(analyzer.data['attention'].keys())[0]
            interactive_fig = analyzer.interactive_viz.create_interactive_attention_viz(layer_idx)
            interactive_fig.write_html(f"./example_plots/interactive_attention_layer_{layer_idx}.html")
            print(f"Created interactive attention plot for layer {layer_idx}")
        
        if analyzer.data.get('embeddings'):
            emb_type = list(analyzer.data['embeddings'].keys())[0]
            interactive_3d = analyzer.interactive_viz.create_3d_embedding_viz(emb_type)
            interactive_3d.write_html(f"./example_plots/interactive_3d_{emb_type}.html")
            print(f"Created 3D embedding plot for {emb_type}")
            
    except FileNotFoundError:
        print(f"Capture file not found: {data_path}")
        print("Run the patched_llama.py script first to generate capture data.")
    except Exception as e:
        print(f"Error: {e}")
        