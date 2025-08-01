#!/bin/bash

# Mistral 4B CLI Installer & Runner for macOS
# Self-contained installation and execution script

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
INSTALL_DIR="$HOME/.mistral-cli"
VENV_DIR="$INSTALL_DIR/venv"
MODEL_DIR="$INSTALL_DIR/models"
PYTHON_SCRIPT="$INSTALL_DIR/mistral_cli.py"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install Homebrew if not present
install_homebrew() {
    if ! command_exists brew; then
        print_status "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        
        # Add Homebrew to PATH for Apple Silicon Macs
        if [[ $(uname -m) == "arm64" ]]; then
            echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zshrc
            eval "$(/opt/homebrew/bin/brew shellenv)"
        fi
    else
        print_success "Homebrew already installed"
    fi
}

# Function to install Python if not present
install_python() {
    if ! command_exists python3; then
        print_status "Installing Python 3..."
        brew install python@3.11
    else
        print_success "Python 3 already installed"
    fi
}

# Function to create directory structure
create_directories() {
    print_status "Creating directory structure..."
    mkdir -p "$INSTALL_DIR"
    mkdir -p "$MODEL_DIR"
    mkdir -p "$VENV_DIR"
}

# Function to create Python virtual environment
create_venv() {
    print_status "Creating Python virtual environment..."
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip
}

# Function to install Python dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."
    source "$VENV_DIR/bin/activate"
    
    # Install core dependencies
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install transformers accelerate bitsandbytes sentencepiece protobuf
    pip install huggingface-hub tokenizers safetensors
    pip install rich prompt-toolkit
}

# Function to create the Python CLI script
create_python_script() {
    print_status "Creating Python CLI script..."
    
    cat > "$PYTHON_SCRIPT" << 'EOF'
#!/usr/bin/env python3
"""
Mistral 4B CLI Interface
Interactive command-line interface for Mistral 4B model
"""

import os
import sys
import json
import argparse
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import warnings
warnings.filterwarnings("ignore")

console = Console()

class MistralCLI:
    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        
    def download_model(self):
        """Download Mistral 4B model"""
        model_name = "mistralai/Mistral-7B-Instruct-v0.2"  # Using 7B as 4B isn't available
        
        console.print(Panel.fit("🚀 Mistral Model Downloader", style="bold blue"))
        console.print(f"Model: {model_name}")
        console.print(f"Device: {self.device}")
        console.print(f"Storage: {self.model_dir}")
        
        if not Confirm.ask("Do you want to download the model? (This may take several GB)"):
            console.print("❌ Download cancelled")
            return False
            
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                # Download tokenizer
                task1 = progress.add_task("Downloading tokenizer...", total=None)
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=self.model_dir
                )
                progress.update(task1, completed=True)
                
                # Configure for memory efficiency
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                # Download model
                task2 = progress.add_task("Downloading model...", total=None)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    cache_dir=self.model_dir,
                    quantization_config=quantization_config,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    device_map="auto"
                )
                progress.update(task2, completed=True)
                
            console.print("✅ Model downloaded successfully!")
            return True
            
        except Exception as e:
            console.print(f"❌ Error downloading model: {str(e)}")
            return False
    
    def load_model(self):
        """Load the model if already downloaded"""
        model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Loading model...", total=None)
                
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=self.model_dir
                )
                
                # Configure for memory efficiency
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                # Load model
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    cache_dir=self.model_dir,
                    quantization_config=quantization_config,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    device_map="auto"
                )
                
                progress.update(task, completed=True)
                
            return True
            
        except Exception as e:
            console.print(f"❌ Error loading model: {str(e)}")
            return False
    
    def generate_response(self, prompt, max_length=512, temperature=0.7):
        """Generate response from the model"""
        if not self.model or not self.tokenizer:
            console.print("❌ Model not loaded!")
            return None
            
        try:
            # Format prompt for Mistral
            formatted_prompt = f"<s>[INST] {prompt} [/INST]"
            
            # Tokenize
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
            if self.device != "cpu":
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the input prompt from response
            response = response.replace(formatted_prompt, "").strip()
            
            return response
            
        except Exception as e:
            console.print(f"❌ Error generating response: {str(e)}")
            return None
    
    def interactive_chat(self):
        """Start interactive chat session"""
        console.print(Panel.fit("💬 Mistral Interactive Chat", style="bold green"))
        console.print("Type 'quit', 'exit', or 'q' to end the session")
        console.print("Type 'clear' to clear the screen")
        console.print("Type 'settings' to adjust parameters\n")
        
        settings = {
            'max_length': 512,
            'temperature': 0.7
        }
        
        while True:
            try:
                user_input = Prompt.ask("\n[bold blue]You[/bold blue]")
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    console.print("👋 Goodbye!")
                    break
                    
                elif user_input.lower() == 'clear':
                    os.system('clear')
                    console.print(Panel.fit("💬 Mistral Interactive Chat", style="bold green"))
                    continue
                    
                elif user_input.lower() == 'settings':
                    self.show_settings(settings)
                    continue
                
                if not user_input.strip():
                    continue
                
                # Generate response
                with Progress(
                    SpinnerColumn(),
                    TextColumn("Generating response..."),
                    console=console
                ) as progress:
                    task = progress.add_task("", total=None)
                    response = self.generate_response(
                        user_input, 
                        settings['max_length'], 
                        settings['temperature']
                    )
                
                if response:
                    console.print(f"\n[bold green]Mistral[/bold green]: {response}")
                else:
                    console.print("❌ Failed to generate response")
                    
            except KeyboardInterrupt:
                console.print("\n👋 Goodbye!")
                break
            except Exception as e:
                console.print(f"❌ Error: {str(e)}")
    
    def show_settings(self, settings):
        """Show and modify settings"""
        console.print(Panel.fit("⚙️ Settings", style="bold yellow"))
        console.print(f"Max Length: {settings['max_length']}")
        console.print(f"Temperature: {settings['temperature']}")
        
        if Confirm.ask("Do you want to modify settings?"):
            try:
                new_max_length = Prompt.ask("Max Length", default=str(settings['max_length']))
                settings['max_length'] = int(new_max_length)
                
                new_temperature = Prompt.ask("Temperature (0.1-2.0)", default=str(settings['temperature']))
                settings['temperature'] = float(new_temperature)
                
                console.print("✅ Settings updated!")
            except ValueError:
                console.print("❌ Invalid input, settings unchanged")

def main():
    parser = argparse.ArgumentParser(description="Mistral 4B CLI Interface")
    parser.add_argument("--model-dir", default=os.path.expanduser("~/.mistral-cli/models"), 
                       help="Directory to store models")
    parser.add_argument("--download", action="store_true", help="Download model")
    parser.add_argument("--chat", action="store_true", help="Start interactive chat")
    
    args = parser.parse_args()
    
    # Initialize CLI
    cli = MistralCLI(args.model_dir)
    
    # Show welcome message
    console.print(Panel.fit("🤖 Mistral CLI", style="bold magenta"))
    
    # Check if model exists
    model_exists = any(Path(args.model_dir).glob("**/pytorch_model*.bin")) or \
                  any(Path(args.model_dir).glob("**/model*.safetensors"))
    
    if args.download or not model_exists:
        if not cli.download_model():
            sys.exit(1)
    
    # Load model
    console.print("Loading model...")
    if not cli.load_model():
        sys.exit(1)
    
    console.print("✅ Model loaded successfully!")
    
    # Start interactive chat
    if args.chat or not any([args.download]):
        cli.interactive_chat()

if __name__ == "__main__":
    main()
EOF

    chmod +x "$PYTHON_SCRIPT"
}

# Function to create launch script
create_launch_script() {
    print_status "Creating launch script..."
    
    cat > "$INSTALL_DIR/launch.sh" << EOF
#!/bin/bash
source "$VENV_DIR/bin/activate"
python3 "$PYTHON_SCRIPT" --chat
EOF

    chmod +x "$INSTALL_DIR/launch.sh"
}

# Function to show menu
show_menu() {
    clear
    echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║                    Mistral 4B CLI Setup                      ║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${GREEN}1.${NC} Install all dependencies and setup environment"
    echo -e "${GREEN}2.${NC} Download Mistral model"
    echo -e "${GREEN}3.${NC} Launch interactive chat"
    echo -e "${GREEN}4.${NC} Check installation status"
    echo -e "${GREEN}5.${NC} Uninstall everything"
    echo -e "${GREEN}6.${NC} Exit"
    echo ""
}

# Function to check installation status
check_status() {
    clear
    echo -e "${BLUE}Installation Status:${NC}"
    echo ""
    
    if [ -d "$VENV_DIR" ]; then
        echo -e "${GREEN}✓${NC} Virtual environment created"
    else
        echo -e "${RED}✗${NC} Virtual environment missing"
    fi
    
    if [ -f "$PYTHON_SCRIPT" ]; then
        echo -e "${GREEN}✓${NC} Python CLI script created"
    else
        echo -e "${RED}✗${NC} Python CLI script missing"
    fi
    
    if command_exists python3; then
        echo -e "${GREEN}✓${NC} Python 3 installed"
    else
        echo -e "${RED}✗${NC} Python 3 missing"
    fi
    
    if [ -d "$MODEL_DIR" ] && [ "$(ls -A $MODEL_DIR)" ]; then
        echo -e "${GREEN}✓${NC} Model directory has content"
    else
        echo -e "${YELLOW}!${NC} Model not downloaded yet"
    fi
    
    echo ""
    read -p "Press Enter to continue..."
}

# Function to uninstall
uninstall() {
    echo -e "${YELLOW}Warning: This will remove all installed files and models!${NC}"
    read -p "Are you sure you want to uninstall? (yes/no): " confirm
    
    if [ "$confirm" = "yes" ]; then
        print_status "Removing installation directory..."
        rm -rf "$INSTALL_DIR"
        print_success "Uninstalled successfully!"
    else
        print_status "Uninstall cancelled"
    fi
    
    read -p "Press Enter to continue..."
}

# Function to run installation
install_all() {
    print_status "Starting installation process..."
    
    # Check for macOS
    if [[ "$OSTYPE" != "darwin"* ]]; then
        print_error "This script is designed for macOS only!"
        exit 1
    fi
    
    # Install dependencies
    install_homebrew
    install_python
    
    # Create environment
    create_directories
    create_venv
    install_dependencies
    create_python_script
    create_launch_script
    
    print_success "Installation completed successfully!"
    print_status "You can now:"
    print_status "1. Download the model (option 2)"
    print_status "2. Launch the chat interface (option 3)"
    print_status "Or run directly: $INSTALL_DIR/launch.sh"
    
    read -p "Press Enter to continue..."
}

# Function to download model
download_model() {
    if [ ! -f "$PYTHON_SCRIPT" ]; then
        print_error "Installation not complete! Please run option 1 first."
        read -p "Press Enter to continue..."
        return
    fi
    
    print_status "Starting model download..."
    source "$VENV_DIR/bin/activate"
    python3 "$PYTHON_SCRIPT" --download
    
    read -p "Press Enter to continue..."
}

# Function to launch chat
launch_chat() {
    if [ ! -f "$PYTHON_SCRIPT" ]; then
        print_error "Installation not complete! Please run option 1 first."
        read -p "Press Enter to continue..."
        return
    fi
    
    source "$VENV_DIR/bin/activate"
    python3 "$PYTHON_SCRIPT" --chat
}

# Main menu loop
main() {
    while true; do
        show_menu
        read -p "Please select an option (1-6): " choice
        
        case $choice in
            1)
                install_all
                ;;
            2)
                download_model
                ;;
            3)
                launch_chat
                ;;
            4)
                check_status
                ;;
            5)
                uninstall
                ;;
            6)
                print_success "Thank you for using Mistral CLI!"
                exit 0
                ;;
            *)
                print_error "Invalid option. Please select 1-6."
                sleep 2
                ;;
        esac
    done
}

# Run main function
main