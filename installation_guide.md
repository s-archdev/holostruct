# Mistral 4B Model Internals Auditing System - Installation Guide

## Prerequisites

### System Requirements

**Minimum Requirements:**
- **RAM**: 16GB (32GB recommended for full analysis)
- **Storage**: 50GB free space (model + analysis data)
- **Python**: 3.8 or higher
- **OS**: Linux, macOS, or Windows with WSL2

**Recommended for Optimal Performance:**
- **RAM**: 32GB or more
- **Storage**: SSD with 100GB+ free space
- **CPU**: 8+ cores for faster processing
- **GPU**: Optional but helps with inference speed

### Hardware-Specific Considerations

**Consumer Hardware (16-32GB RAM):**
- Enable selective layer capture
- Use compressed storage
- Process shorter sequences
- Batch operations carefully

**High-Memory Systems (64GB+ RAM):**
- Full model analysis possible
- Real-time visualization supported
- Cross-layer comparisons enabled

## Installation Steps

### 1. Create Virtual Environment

```bash
# Create and activate virtual environment
python -m venv mistral_audit_env
source mistral_audit_env/bin/activate  # Linux/macOS
# or
mistral_audit_env\Scripts\activate     # Windows
```

### 2. Install Core Dependencies

```bash
# Core ML and analysis libraries
pip install numpy>=1.21.0
pip install pandas>=1.3.0
pip install matplotlib>=3.5.0
pip install seaborn>=0.11.0
pip install scikit-learn>=1.0.0

# Visualization libraries
pip install plotly>=5.0.0
pip install bokeh>=2.4.0

# Data handling
pip install h5py>=3.0.0
pip install tqdm>=4.60.0

# Dimensionality reduction
pip install umap-learn>=0.5.0

# Interactive dashboard (optional)
pip install dash>=2.0.0
pip install dash-bootstrap-components
```

### 3. Install llama-cpp-python

**Option A: CPU Only (Recommended for most users)**
```bash
pip install llama-cpp-python
```

**Option B: GPU Acceleration (CUDA)**
```bash
# For NVIDIA GPUs with CUDA
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --no-cache-dir
```

**Option C: Metal Acceleration (macOS)**
```bash
# For Apple Silicon Macs
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --no-cache-dir
```

### 4. Verify Installation

```bash
python -c "import llama_cpp; print('llama-cpp-python installed successfully')"
python -c "import h5py, matplotlib, plotly; print('All dependencies installed')"
```

## Download and Setup

### 1. Get the Code

```bash
# Create project directory
mkdir mistral_audit_system
cd mistral_audit_system

# Copy the provided Python files:
# - patched_llama.py
# - visualization_suite.py  
# - example_usage_script.py
```

### 2. Get Mistral 4B GGUF Model

**Option A: Download from Hugging Face**
```bash
# Install huggingface-hub
pip install huggingface-hub

# Download model (choose appropriate quantization)
huggingface-cli download microsoft/DialoGPT-medium --local-dir ./models/
# Note: Replace with actual Mistral 4B GGUF model repository
```

**Option B: Convert from Other Formats**
```bash
# If you have the model in another format, use llama.cpp conversion tools
# Follow llama.cpp documentation for conversion
```

### 3. Create Directory Structure

```bash
mkdir -p data/dumps
mkdir -p data/processed
mkdir -p plots
mkdir -p interactive
mkdir -p examples
```

## Configuration

### 1. Basic Configuration File

Create `config.yaml`:
```yaml
model:
  path: "./models/mistral-4b-q4_0.gguf"
  context_size: 2048
  threads: 4

capture:
  weights: true
  activations: true
  attention: true
  embeddings: true
  layers: [0, 1, 15, 31]  # Sample layers
  max_tokens: 512
  compress: true

output:
  base_dir: "./analysis_results"
  save_plots: true
  save_interactive: true
  format: "png"  # png, pdf, svg
```

### 2. Memory-Optimized Configuration

For systems with limited RAM, create `low_memory_config.yaml`:
```yaml
model:
  path: "./models/mistral-4b-q4_0.gguf"
  context_size: 1024  # Reduced context
  threads: 2

capture:
  weights: false      # Skip weights to save memory
  activations: true
  attention: false    # Skip attention to save memory
  embeddings: true
  layers: [0, 31]     # Only first and last layers
  max_tokens: 128     # Shorter sequences
  compress: true

output:
  base_dir: "./analysis_results"
  save_plots: true
  save_interactive: false  # Skip interactive plots
  format: "png"
```

## First Run Test

### 1. Basic Functionality Test

```bash
python -c "
from patched_llama import PatchedLlama, CaptureConfig
config = CaptureConfig(
    capture_weights=False,
    capture_activations=True,
    capture_attention=False,
    capture_embeddings=True,
    capture_layers=[0],
    max_tokens=10,
    output_dir='./test_output',
    verbose=True
)
print('Configuration created successfully')
"
```

### 2. Model Loading Test

```bash
python example_usage_script.py path/to/your/mistral-4b.gguf --mode basic --prompt "Hello" --output-dir ./test_run
```

### 3. Visualization Test

```bash
python -c "
from visualization_suite import ModelAnalyzer
# This will fail if no data exists yet, but tests imports
print('Visualization suite imported successfully')
"
```

## Troubleshooting

### Common Issues and Solutions

**Issue 1: llama-cpp-python compilation fails**
```bash
# Solution: Install build tools
# Ubuntu/Debian:
sudo apt-get install build-essential cmake

# macOS:
xcode-select --install

# Windows: Install Visual Studio Build Tools
```

**Issue 2: Out of memory errors**
```bash
# Solution: Use memory-optimized configuration
# Reduce context size, skip weights, use fewer layers
# Monitor memory usage with:
python -c "import psutil; print(f'Available RAM: {psutil.virtual_memory().available/1024**3:.1f} GB')"
```

**Issue 3: Model file not found**
```bash
# Solution: Verify model path
ls -la path/to/your/model.gguf
# Ensure GGUF format (not other formats like .bin, .safetensors)
```

**Issue 4: Slow inference**
```bash
# Solution: Optimize model settings
# - Reduce context size
# - Increase thread count (up to CPU cores)
# - Use GPU acceleration if available
# - Use lower precision quantization (Q4_0 instead of Q8_0)
```

**Issue 5: Visualization errors**
```bash
# Solution: Check display backend
# For headless systems:
export MPLBACKEND=Agg

# For remote systems:
pip install 'matplotlib[backend_agg]'
```

### Performance Optimization

**For CPU-Only Systems:**
```python
# Optimize thread usage
import os
os.environ['OMP_NUM_