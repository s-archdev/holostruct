#!/usr/bin/env python3
"""
Complete example script demonstrating the Mistral 4B model internals auditing system.
This script shows how to use all components together for comprehensive analysis.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Optional
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our custom modules
try:
    from patched_llama import PatchedLlama, CaptureConfig, load_captured_data
    from visualization_suite import ModelAnalyzer
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Make sure all required modules are in the same directory or PYTHONPATH")
    sys.exit(1)

def run_basic_analysis(model_path: str, prompt: str, output_dir: str = "./analysis_output"):
    """Run a basic analysis workflow."""
    logger.info("Starting basic analysis workflow...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure capture settings
    config = CaptureConfig(
        capture_weights=True,
        capture_activations=True,
        capture_attention=True,
        capture_embeddings=True,
        capture_layers=[0, 1, 15, 31],  # Sample layers: first, second, middle, last
        max_tokens=100,
        output_dir=os.path.join(output_dir, "dumps"),
        compress=True,
        verbose=True
    )
    
    # Initialize model
    logger.info(f"Loading model from {model_path}")
    try:
        model = PatchedLlama(
            model_path=model_path,
            capture_config=config,
            n_ctx=2048,
            n_threads=4,
            verbose=False
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None
    
    # Generate with capture
    logger.info(f"Processing prompt: '{prompt}'")
    try:
        output_text, dump_file = model.generate_with_capture(prompt, max_tokens=50)
        logger.info(f"Generated text: '{output_text.strip()}'")
        logger.info(f"Captured data saved to: {dump_file}")
    except Exception as e:
        logger.error(f"Failed to generate with capture: {e}")
        return None
    
    # Analyze captured data
    logger.info("Analyzing captured data...")
    try:
        analyzer = ModelAnalyzer(dump_file)
        analyzer.print_summary()
        
        # Generate all visualizations
        plots_dir = os.path.join(output_dir, "plots")
        plots = analyzer.generate_all_plots(plots_dir)
        logger.info(f"Generated {len(plots)} plots in {plots_dir}")
        
        # Create interactive visualizations
        interactive_dir = os.path.join(output_dir, "interactive")
        os.makedirs(interactive_dir, exist_ok=True)
        
        # Attention visualization
        if analyzer.data.get('attention'):
            layer_idx = sorted(analyzer.data['attention'].keys())[0]
            fig = analyzer.interactive_viz.create_interactive_attention_viz(layer_idx)
            fig.write_html(os.path.join(interactive_dir, f"attention_layer_{layer_idx}.html"))
            logger.info(f"Created interactive attention plot for layer {layer_idx}")
        
        # 3D embedding visualization
        if analyzer.data.get('embeddings'):
            emb_type = list(analyzer.data['embeddings'].keys())[0]
            fig = analyzer.interactive_viz.create_3d_embedding_viz(emb_type)
            fig.write_html(os.path.join(interactive_dir, f"embeddings_3d_{emb_type}.html"))
            logger.info(f"Created 3D embedding plot for {emb_type}")
        
        return dump_file, plots_dir, interactive_dir
        
    except Exception as e:
        logger.error(f"Failed to analyze data: {e}")
        return None

def run_comparative_analysis(model_path: str, prompts: List[str], output_dir: str = "./comparative_analysis"):
    """Run comparative analysis across multiple prompts."""
    logger.info("Starting comparative analysis...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    config = CaptureConfig(
        capture_weights=False,  # Skip weights for comparative analysis
        capture_activations=True,
        capture_attention=True,
        capture_embeddings=True,
        capture_layers=[0, 15, 31],  # Fewer layers for efficiency
        max_tokens=50,
        output_dir=os.path.join(output_dir, "dumps"),
        compress=True,
        verbose=True
    )
    
    # Initialize model
    model = PatchedLlama(
        model_path=model_path,
        capture_config=config,
        n_ctx=2048,
        verbose=False
    )
    
    results = []
    
    # Process each prompt
    for i, prompt in enumerate(prompts):
        logger.info(f"Processing prompt {i+1}/{len(prompts)}: '{prompt}'")
        
        try:
            output_text, dump_file = model.generate_with_capture(prompt, max_tokens=30)
            results.append({
                'prompt': prompt,
                'output': output_text,
                'dump_file': dump_file
            })
        except Exception as e:
            logger.error(f"Failed to process prompt {i+1}: {e}")
            continue
    
    # Comparative analysis
    if len(results) >= 2:
        logger.info("Performing comparative analysis...")
        
        # Load all data
        all_data = []
        for result in results:
            data = load_captured_data(result['dump_file'])
            data['prompt'] = result['prompt']
            all_data.append(data)
        
        # Compare attention patterns
        compare_attention_patterns(all_data, output_dir)
        
        # Compare activation statistics
        compare_activation_statistics(all_data, output_dir)
        
        # Compare embedding similarities
        compare_embeddings(all_data, output_dir)
    
    return results

def compare_attention_patterns(all_data: List[dict], output_dir: str):
    """Compare attention patterns across different prompts."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    logger.info("Comparing attention patterns...")
    
    # Get common layers
    common_layers = set(all_data[0].get('attention', {}).keys())
    for data in all_data[1:]:
        common_layers &= set(data.get('attention', {}).keys())
    
    if not common_layers:
        logger.warning("No common attention layers found")
        return
    
    for layer_idx in sorted(common_layers):
        fig, axes = plt.subplots(1, len(all_data), figsize=(5 * len(all_data), 4))
        if len(all_data) == 1:
            axes = [axes]
        
        for i, data in enumerate(all_data):
            attention_scores = data['attention'][layer_idx]
            # Average across heads for visualization
            avg_attention = np.mean(attention_scores, axis=0)
            
            im = axes[i].imshow(avg_attention, cmap='Blues', interpolation='nearest')
            axes[i].set_title(f"Prompt {i+1}\n{data['prompt'][:30]}...")
            axes[i].set_xlabel('Key Position')
            axes[i].set_ylabel('Query Position')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'attention_comparison_layer_{layer_idx}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close(fig)

def compare_activation_statistics(all_data: List[dict], output_dir: str):
    """Compare activation statistics across prompts."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    logger.info("Comparing activation statistics...")
    
    # Collect statistics
    stats_data = []
    
    for data_idx, data in enumerate(all_data):
        activations = data.get('activations', {})
        
        for layer_idx, layer_acts in activations.items():
            all_acts = np.concatenate([acts for acts in layer_acts.values()])
            
            stats_data.append({
                'prompt_idx': data_idx,
                'prompt': data['prompt'][:30] + "...",
                'layer': layer_idx,
                'mean': np.mean(all_acts),
                'std': np.std(all_acts),
                'sparsity': np.mean(np.abs(all_acts) < 0.01)
            })
    
    if not stats_data:
        logger.warning("No activation statistics to compare")
        return
    
    df = pd.DataFrame(stats_data)
    
    # Plot comparisons
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Mean activations by layer and prompt
    for prompt_idx in df['prompt_idx'].unique():
        prompt_data = df[df['prompt_idx'] == prompt_idx]
        axes[0].plot(prompt_data['layer'], prompt_data['mean'], 
                    'o-', label=f"Prompt {prompt_idx + 1}", alpha=0.7)
    
    axes[0].set_title('Mean Activations by Layer')
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('Mean Activation')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Standard deviations
    for prompt_idx in df['prompt_idx'].unique():
        prompt_data = df[df['prompt_idx'] == prompt_idx]
        axes[1].plot(prompt_data['layer'], prompt_data['std'], 
                    'o-', label=f"Prompt {prompt_idx + 1}", alpha=0.7)
    
    axes[1].set_title('Activation Std by Layer')
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('Activation Std')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Sparsity
    for prompt_idx in df['prompt_idx'].unique():
        prompt_data = df[df['prompt_idx'] == prompt_idx]
        axes[2].plot(prompt_data['layer'], prompt_data['sparsity'], 
                    'o-', label=f"Prompt {prompt_idx + 1}", alpha=0.7)
    
    axes[2].set_title('Activation Sparsity by Layer')
    axes[2].set_xlabel('Layer')
    axes[2].set_ylabel('Fraction Near Zero')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'activation_statistics_comparison.png'), 
               dpi=300, bbox_inches='tight')
    plt.close(fig)

def compare_embeddings(all_data: List[dict], output_dir: str):
    """Compare embedding similarities across prompts."""
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.decomposition import PCA
    
    logger.info("Comparing embeddings...")
    
    # Collect embeddings
    all_embeddings = []
    labels = []
    
    for data_idx, data in enumerate(all_data):
        embeddings = data.get('embeddings', {})
        
        for emb_type, emb_data in embeddings.items():
            # Take mean embedding for each prompt
            mean_emb = np.mean(emb_data, axis=0)
            all_embeddings.append(mean_emb)
            labels.append(f"P{data_idx+1}_{emb_type}")
    
    if len(all_embeddings) < 2:
        logger.warning("Not enough embeddings to compare")
        return
    
    all_embeddings = np.array(all_embeddings)
    
    # Compute similarity matrix
    similarity_matrix = cosine_similarity(all_embeddings)
    
    # Plot similarity matrix
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    im = ax1.imshow(similarity_matrix, cmap='RdYlBu_r', vmin=-1, vmax=1)
    ax1.set_title('Embedding Cosine Similarities')
    ax1.set_xticks(range(len(labels)))
    ax1.set_yticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=45)
    ax1.set_yticklabels(labels)
    plt.colorbar(im, ax=ax1)
    
    # Add similarity values as text
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax1.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                           ha="center", va="center", color="black" if abs(similarity_matrix[i, j]) < 0.5 else "white")
    
    # PCA projection
    if all_embeddings.shape[1] > 2:
        pca = PCA(n_components=2)
        projected = pca.fit_transform(all_embeddings)
        
        scatter = ax2.scatter(projected[:, 0], projected[:, 1], 
                            c=range(len(projected)), cmap='viridis', s=100, alpha=0.7)
        ax2.set_title('PCA Projection of Mean Embeddings')
        ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        
        # Add labels
        for i, label in enumerate(labels):
            ax2.annotate(label, (projected[i, 0], projected[i, 1]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'embedding_comparisons.png'), 
               dpi=300, bbox_inches='tight')
    plt.close(fig)

def run_interactive_dashboard(model_path: str, output_dir: str = "./dashboard"):
    """Create an interactive dashboard for live analysis."""
    try:
        import dash
        from dash import dcc, html, Input, Output, State
        import plotly.graph_objects as go
    except ImportError:
        logger.error("Dash not installed. Run: pip install dash")
        return
    
    logger.info("Starting interactive dashboard...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize Dash app
    app = dash.Dash(__name__)
    
    # Configure model
    config = CaptureConfig(
        capture_weights=False,
        capture_activations=True,
        capture_attention=True,
        capture_embeddings=True,
        capture_layers=[0, 15, 31],
        max_tokens=50,
        output_dir=os.path.join(output_dir, "live_dumps"),
        compress=True,
        verbose=False
    )
    
    model = PatchedLlama(model_path=model_path, capture_config=config, verbose=False)
    
    # App layout
    app.layout = html.Div([
        html.H1("Mistral 4B Live Analysis Dashboard"),
        
        html.Div([
            html.Label("Enter prompt:"),
            dcc.Textarea(
                id='prompt-input',
                value='The future of artificial intelligence is',
                style={'width': '100%', 'height': 100}
            ),
            html.Button('Analyze', id='analyze-button', n_clicks=0),
        ], style={'margin': '20px'}),
        
        html.Div(id='output-text', style={'margin': '20px'}),
        
        dcc.Tabs(id='tabs', value='attention-tab', children=[
            dcc.Tab(label='Attention Patterns', value='attention-tab'),
            dcc.Tab(label='Activation Statistics', value='activation-tab'),
            dcc.Tab(label='Embeddings', value='embedding-tab'),
        ]),
        
        html.Div(id='tab-content')
    ])
    
    @app.callback(
        [Output('output-text', 'children'),
         Output('tab-content', 'children')],
        [Input('analyze-button', 'n_clicks')],
        [State('prompt-input', 'value')]
    )
    def update_analysis(n_clicks, prompt):
        if n_clicks == 0:
            return "Enter a prompt and click Analyze", html.Div()
        
        try:
            # Generate with capture
            output_text, dump_file = model.generate_with_capture(prompt, max_tokens=30)
            
            # Load and analyze data
            analyzer = ModelAnalyzer(dump_file)
            
            # Create visualizations based on current tab
            attention_content = create_attention_content(analyzer)
            activation_content = create_activation_content(analyzer)
            embedding_content = create_embedding_content(analyzer)
            
            output_div = html.Div([
                html.H3("Generated Text:"),
                html.P(output_text),
                html.Hr(),
                attention_content,
                activation_content,
                embedding_content
            ])
            
            return f"Generated: {output_text}", output_div
            
        except Exception as e:
            return f"Error: {str(e)}", html.Div()
    
    def create_attention_content(analyzer):
        if not analyzer.data.get('attention'):
            return html.Div("No attention data available")
        
        layer_idx = sorted(analyzer.data['attention'].keys())[0]
        fig = analyzer.interactive_viz.create_interactive_attention_viz(layer_idx)
        
        return html.Div([
            html.H3(f"Attention Patterns - Layer {layer_idx}"),
            dcc.Graph(figure=fig)
        ])
    
    def create_activation_content(analyzer):
        if not analyzer.data.get('activations'):
            return html.Div("No activation data available")
        
        fig = analyzer.activation_viz.plot_activation_evolution()
        
        return html.Div([
            html.H3("Activation Evolution"),
            dcc.Graph(figure=fig)
        ])
    
    def create_embedding_content(analyzer):
        if not analyzer.data.get('embeddings'):
            return html.Div("No embedding data available")
        
        emb_type = list(analyzer.data['embeddings'].keys())[0]
        fig = analyzer.interactive_viz.create_3d_embedding_viz(emb_type)
        
        return html.Div([
            html.H3(f"3D Embeddings - {emb_type}"),
            dcc.Graph(figure=fig)
        ])
    
    # Run the app
    logger.info("Dashboard available at http://127.0.0.1:8050/")
    app.run_server(debug=True, host='127.0.0.1', port=8050)

def main():
    """Main entry point with command-line interface."""
    parser = argparse.ArgumentParser(description="Mistral 4B Model Internals Auditing System")
    parser.add_argument("model_path", help="Path to Mistral 4B GGUF file")
    parser.add_argument("--mode", choices=["basic", "comparative", "dashboard"], 
                       default="basic", help="Analysis mode")
    parser.add_argument("--prompt", default="The future of artificial intelligence is",
                       help="Prompt for basic analysis")
    parser.add_argument("--prompts", nargs="+", 
                       default=["Hello, how are you?", "Explain quantum computing", "Write a short story"],
                       help="Multiple prompts for comparative analysis")
    parser.add_argument("--output-dir", default="./analysis_results",
                       help="Output directory for results")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate model path
    if not os.path.exists(args.model_path):
        logger.error(f"Model file not found: {args.model_path}")
        sys.exit(1)
    
    # Run selected mode
    try:
        if args.mode == "basic":
            result = run_basic_analysis(args.model_path, args.prompt, args.output_dir)
            if result:
                dump_file, plots_dir, interactive_dir = result
                logger.info(f"Analysis complete! Results saved to {args.output_dir}")
                logger.info(f"- Data dump: {dump_file}")
                logger.info(f"- Static plots: {plots_dir}")
                logger.info(f"- Interactive plots: {interactive_dir}")
        
        elif args.mode == "comparative":
            results = run_comparative_analysis(args.model_path, args.prompts, args.output_dir)
            logger.info(f"Comparative analysis complete! Results saved to {args.output_dir}")
            logger.info(f"Processed {len(results)} prompts")
        
        elif args.mode == "dashboard":
            run_interactive_dashboard(args.model_path, args.output_dir)
    
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

# Additional utility functions for batch processing
def batch_process_prompts(model_path: str, prompts_file: str, output_dir: str = "./batch_results"):
    """Process a large batch of prompts from a file."""
    logger.info(f"Starting batch processing from {prompts_file}")
    
    # Read prompts from file
    with open(prompts_file, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Found {len(prompts)} prompts to process")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure for efficient batch processing
    config = CaptureConfig(
        capture_weights=False,  # Skip weights for efficiency
        capture_activations=True,
        capture_attention=False,  # Skip attention for speed
        capture_embeddings=True,
        capture_layers=[0, 31],  # Only first and last layers
        max_tokens=20,  # Short generations
        output_dir=os.path.join(output_dir, "dumps"),
        compress=True,
        verbose=False
    )
    
    model = PatchedLlama(model_path=model_path, capture_config=config, verbose=False)
    
    results = []
    failed_prompts = []
    
    from tqdm import tqdm
    
    for i, prompt in enumerate(tqdm(prompts, desc="Processing prompts")):
        try:
            output_text, dump_file = model.generate_with_capture(prompt, max_tokens=20)
            results.append({
                'index': i,
                'prompt': prompt,
                'output': output_text,
                'dump_file': dump_file
            })
        except Exception as e:
            logger.warning(f"Failed to process prompt {i}: {e}")
            failed_prompts.append((i, prompt, str(e)))
    
    # Save results summary
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, "batch_results.csv"), index=False)
    
    if failed_prompts:
        failed_df = pd.DataFrame(failed_prompts, columns=['index', 'prompt', 'error'])
        failed_df.to_csv(os.path.join(output_dir, "failed_prompts.csv"), index=False)
    
    logger.info(f"Batch processing complete: {len(results)} successful, {len(failed_prompts)} failed")
    
    return results, failed_prompts

# Performance monitoring utilities
def monitor_memory_usage():
    """Monitor memory usage during analysis."""
    import psutil
    import time
    import threading
    
    def log_memory():
        process = psutil.Process()
        while True:
            memory_mb = process.memory_info().rss / 1024 / 1024
            logger.info(f"Memory usage: {memory_mb:.1f} MB")
            time.sleep(10)
    
    thread = threading.Thread(target=log_memory, daemon=True)
    thread.start()
    return thread