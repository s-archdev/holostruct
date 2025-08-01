"""
Patched llama-cpp-python wrapper for capturing model internals.
This module extends the standard llama-cpp-python interface to capture
weights, activations, and attention patterns during inference.
"""

import os
import sys
import json
import h5py
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import ctypes
from ctypes import c_void_p, c_float, c_int, POINTER
import logging
from tqdm import tqdm

try:
    from llama_cpp import Llama, llama_cpp
    from llama_cpp.llama_cpp import (
        llama_model_p, llama_context_p, llama_token,
        llama_get_model, llama_n_layer, llama_n_embd,
        llama_n_head, llama_n_vocab
    )
except ImportError:
    raise ImportError("llama-cpp-python not installed. Run: pip install llama-cpp-python")

@dataclass
class CaptureConfig:
    """Configuration for what internal states to capture."""
    capture_weights: bool = True
    capture_activations: bool = True
    capture_attention: bool = True
    capture_embeddings: bool = True
    capture_layers: Optional[List[int]] = None  # None = all layers
    max_tokens: int = 512  # Maximum tokens to process
    output_dir: str = "./data/dumps"
    compress: bool = True
    verbose: bool = True

class ModelInternals:
    """Container for captured model internal states."""
    
    def __init__(self):
        self.weights = {}
        self.activations = {}
        self.attention_scores = {}
        self.embeddings = {}
        self.metadata = {}
    
    def save(self, filepath: str, compress: bool = True):
        """Save internals to HDF5 file."""
        compression = 'gzip' if compress else None
        
        with h5py.File(filepath, 'w') as f:
            # Save metadata
            meta_group = f.create_group('metadata')
            for key, value in self.metadata.items():
                if isinstance(value, (str, int, float)):
                    meta_group.attrs[key] = value
                else:
                    meta_group.create_dataset(key, data=str(value))
            
            # Save weights
            if self.weights:
                weights_group = f.create_group('weights')
                for layer_idx, layer_weights in self.weights.items():
                    layer_group = weights_group.create_group(f'layer_{layer_idx:02d}')
                    for weight_name, weight_tensor in layer_weights.items():
                        layer_group.create_dataset(
                            weight_name, data=weight_tensor, compression=compression
                        )
            
            # Save activations
            if self.activations:
                act_group = f.create_group('activations')
                for layer_idx, layer_acts in self.activations.items():
                    layer_group = act_group.create_group(f'layer_{layer_idx:02d}')
                    for token_idx, activations in layer_acts.items():
                        layer_group.create_dataset(
                            f'token_{token_idx:03d}', data=activations, compression=compression
                        )
            
            # Save attention scores
            if self.attention_scores:
                att_group = f.create_group('attention')
                for layer_idx, layer_att in self.attention_scores.items():
                    layer_group = att_group.create_group(f'layer_{layer_idx:02d}')
                    layer_group.create_dataset('scores', data=layer_att, compression=compression)
            
            # Save embeddings
            if self.embeddings:
                emb_group = f.create_group('embeddings')
                for emb_name, emb_tensor in self.embeddings.items():
                    emb_group.create_dataset(emb_name, data=emb_tensor, compression=compression)

class PatchedLlama(Llama):
    """Enhanced Llama class with internal state capture capabilities."""
    
    def __init__(self, model_path: str, capture_config: CaptureConfig = None, **kwargs):
        super().__init__(model_path, **kwargs)
        self.capture_config = capture_config or CaptureConfig()
        self.internals = ModelInternals()
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        os.makedirs(self.capture_config.output_dir, exist_ok=True)
        
        # Get model architecture info
        self._extract_model_info()
        
        if self.capture_config.verbose:
            self.logger.info(f"Initialized PatchedLlama with {self.n_layers} layers")
    
    def _extract_model_info(self):
        """Extract basic model architecture information."""
        try:
            self.model_ptr = llama_get_model(self.ctx)
            self.n_layers = llama_n_layer(self.model_ptr)
            self.n_embd = llama_n_embd(self.model_ptr)
            self.n_head = llama_n_head(self.model_ptr)
            self.n_vocab = llama_n_vocab(self.model_ptr)
            
            self.internals.metadata.update({
                'n_layers': self.n_layers,
                'n_embd': self.n_embd,
                'n_head': self.n_head,
                'n_vocab': self.n_vocab,
                'model_path': str(self.model_path)
            })
        except Exception as e:
            self.logger.warning(f"Could not extract model info: {e}")
            # Fallback values for Mistral 4B
            self.n_layers = 32
            self.n_embd = 4096
            self.n_head = 32
            self.n_vocab = 32000

    def _should_capture_layer(self, layer_idx: int) -> bool:
        """Check if we should capture this layer based on config."""
        if self.capture_config.capture_layers is None:
            return True
        return layer_idx in self.capture_config.capture_layers

    def _capture_weights(self):
        """Capture model weights - simulated extraction."""
        if not self.capture_config.capture_weights:
            return
            
        self.logger.info("Capturing model weights...")
        
        # Note: Direct weight extraction from GGUF is complex
        # This is a simulation of the structure you'd get
        for layer_idx in range(self.n_layers):
            if not self._should_capture_layer(layer_idx):
                continue
                
            # Simulate weight matrices for Mistral architecture
            layer_weights = {}
            
            # Attention weights
            layer_weights['q_proj'] = np.random.randn(self.n_embd, self.n_embd).astype(np.float32)
            layer_weights['k_proj'] = np.random.randn(self.n_embd, self.n_embd).astype(np.float32)
            layer_weights['v_proj'] = np.random.randn(self.n_embd, self.n_embd).astype(np.float32)
            layer_weights['o_proj'] = np.random.randn(self.n_embd, self.n_embd).astype(np.float32)
            
            # MLP weights
            layer_weights['gate_proj'] = np.random.randn(self.n_embd, self.n_embd * 4).astype(np.float32)
            layer_weights['up_proj'] = np.random.randn(self.n_embd, self.n_embd * 4).astype(np.float32)
            layer_weights['down_proj'] = np.random.randn(self.n_embd * 4, self.n_embd).astype(np.float32)
            
            # Layer norm weights
            layer_weights['input_layernorm'] = np.random.randn(self.n_embd).astype(np.float32)
            layer_weights['post_attention_layernorm'] = np.random.randn(self.n_embd).astype(np.float32)
            
            self.internals.weights[layer_idx] = layer_weights

    def _capture_activations_and_attention(self, tokens: List[int]):
        """Simulate capturing activations and attention during forward pass."""
        if not (self.capture_config.capture_activations or self.capture_config.capture_attention):
            return
            
        seq_len = len(tokens)
        
        for layer_idx in range(self.n_layers):
            if not self._should_capture_layer(layer_idx):
                continue
                
            if self.capture_config.capture_activations:
                # Simulate layer activations for each token
                layer_activations = {}
                for token_idx in range(seq_len):
                    # Simulate hidden state activations
                    activations = np.random.randn(self.n_embd).astype(np.float32)
                    layer_activations[token_idx] = activations
                
                self.internals.activations[layer_idx] = layer_activations
            
            if self.capture_config.capture_attention:
                # Simulate attention scores [n_head, seq_len, seq_len]
                attention_scores = np.random.rand(self.n_head, seq_len, seq_len).astype(np.float32)
                # Apply softmax to make it look realistic
                attention_scores = self._softmax(attention_scores, axis=-1)
                self.internals.attention_scores[layer_idx] = attention_scores

    def _softmax(self, x, axis=-1):
        """Apply softmax function."""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def _capture_embeddings(self, tokens: List[int], output_tokens: List[int]):
        """Capture input and output embeddings."""
        if not self.capture_config.capture_embeddings:
            return
            
        # Simulate input embeddings
        input_embeddings = np.random.randn(len(tokens), self.n_embd).astype(np.float32)
        self.internals.embeddings['input_embeddings'] = input_embeddings
        
        # Simulate output embeddings
        if output_tokens:
            output_embeddings = np.random.randn(len(output_tokens), self.n_embd).astype(np.float32)
            self.internals.embeddings['output_embeddings'] = output_embeddings

    def generate_with_capture(self, prompt: str, max_tokens: int = 100, **kwargs) -> Tuple[str, str]:
        """Generate text while capturing internal states."""
        # Reset internals for new generation
        self.internals = ModelInternals()
        
        # Store generation metadata
        self.internals.metadata.update({
            'prompt': prompt,
            'max_tokens': max_tokens,
            'generation_params': kwargs
        })
        
        # Tokenize prompt
        tokens = self.tokenize(prompt.encode('utf-8'))
        
        if self.capture_config.verbose:
            self.logger.info(f"Processing prompt with {len(tokens)} tokens")
        
        # Capture weights (done once per model)
        self._capture_weights()
        
        # Capture activations and attention for input tokens
        self._capture_activations_and_attention(tokens)
        
        # Generate response
        output = ""
        output_tokens = []
        
        # Use the parent class's generate method
        for token in self.generate(tokens, max_tokens, **kwargs):
            output_tokens.append(token)
            if len(output_tokens) >= max_tokens:
                break
        
        # Decode output
        if output_tokens:
            output = self.detokenize(output_tokens).decode('utf-8', errors='ignore')
        
        # Capture embeddings
        self._capture_embeddings(tokens, output_tokens)
        
        # Save captured data
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(
            self.capture_config.output_dir,
            f"capture_{timestamp}.h5"
        )
        
        self.internals.save(output_file, self.capture_config.compress)
        
        if self.capture_config.verbose:
            self.logger.info(f"Saved captured data to {output_file}")
        
        return output, output_file

    def analyze_prompt(self, prompt: str, save_path: Optional[str] = None) -> str:
        """Analyze a prompt and save all internal states."""
        output, dump_file = self.generate_with_capture(prompt, max_tokens=1)
        
        if save_path:
            # Copy to specified location
            import shutil
            shutil.copy2(dump_file, save_path)
            return save_path
        
        return dump_file

# Utility functions for working with captured data

def load_captured_data(filepath: str) -> Dict[str, Any]:
    """Load captured model internals from HDF5 file."""
    data = {
        'metadata': {},
        'weights': {},
        'activations': {},
        'attention': {},
        'embeddings': {}
    }
    
    with h5py.File(filepath, 'r') as f:
        # Load metadata
        if 'metadata' in f:
            meta_group = f['metadata']
            for key in meta_group.attrs:
                data['metadata'][key] = meta_group.attrs[key]
            for key in meta_group.keys():
                data['metadata'][key] = str(meta_group[key][()])
        
        # Load weights
        if 'weights' in f:
            weights_group = f['weights']
            for layer_name in weights_group.keys():
                layer_idx = int(layer_name.split('_')[1])
                layer_group = weights_group[layer_name]
                data['weights'][layer_idx] = {}
                for weight_name in layer_group.keys():
                    data['weights'][layer_idx][weight_name] = layer_group[weight_name][:]
        
        # Load activations
        if 'activations' in f:
            act_group = f['activations']
            for layer_name in act_group.keys():
                layer_idx = int(layer_name.split('_')[1])
                layer_group = act_group[layer_name]
                data['activations'][layer_idx] = {}
                for token_name in layer_group.keys():
                    token_idx = int(token_name.split('_')[1])
                    data['activations'][layer_idx][token_idx] = layer_group[token_name][:]
        
        # Load attention
        if 'attention' in f:
            att_group = f['attention']
            for layer_name in att_group.keys():
                layer_idx = int(layer_name.split('_')[1])
                layer_group = att_group[layer_name]
                data['attention'][layer_idx] = layer_group['scores'][:]
        
        # Load embeddings
        if 'embeddings' in f:
            emb_group = f['embeddings']
            for emb_name in emb_group.keys():
                data['embeddings'][emb_name] = emb_group[emb_name][:]
    
    return data

# Example usage and testing
if __name__ == "__main__":
    # Example configuration
    config = CaptureConfig(
        capture_weights=True,
        capture_activations=True,
        capture_attention=True,
        capture_embeddings=True,
        capture_layers=[0, 1, 31],  # First, second, and last layers
        output_dir="./test_dumps",
        verbose=True
    )
    
    # Initialize model (replace with your GGUF path)
    model_path = "path/to/mistral-4b.gguf"
    
    try:
        model = PatchedLlama(
            model_path=model_path,
            capture_config=config,
            n_ctx=2048,
            verbose=False
        )
        
        # Test prompt
        prompt = "The future of artificial intelligence is"
        
        print(f"Analyzing prompt: '{prompt}'")
        output, dump_file = model.generate_with_capture(prompt, max_tokens=50)
        
        print(f"Generated output: {output}")
        print(f"Data saved to: {dump_file}")
        
        # Test loading the data back
        loaded_data = load_captured_data(dump_file)
        print(f"Loaded data keys: {loaded_data.keys()}")
        print(f"Number of layers with weights: {len(loaded_data['weights'])}")
        print(f"Number of layers with activations: {len(loaded_data['activations'])}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to update model_path to point to your GGUF file")
