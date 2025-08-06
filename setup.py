#!/usr/bin/env python3
"""
SEO Cannibalization Analyzer - Setup Script
Automatically creates the project structure and configuration files
"""

import os
import sys
import shutil
from pathlib import Path

def create_directory_structure():
    """Create the necessary directory structure"""
    
    directories = [
        ".streamlit",
        "test_data",
        "exports",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}/")
    
    return True

def create_config_files():
    """Create configuration files if they don't exist"""
    
    # Check if config.toml exists
    config_path = Path(".streamlit/config.toml")
    if not config_path.exists():
        print("📝 config.toml not found. Creating from template...")
        # You would copy the config.toml content here
        print("   Please copy the config.toml content from the artifact")
    else:
        print("✅ config.toml already exists")
    
    # Check if secrets.toml.example exists
    secrets_example_path = Path(".streamlit/secrets.toml.example")
    if not secrets_example_path.exists():
        print("📝 Creating secrets.toml.example...")
        # You would copy the secrets.toml.example content here
        print("   Please copy the secrets.toml.example content from the artifact")
    else:
        print("✅ secrets.toml.example already exists")
    
    # Create actual secrets.toml if it doesn't exist
    secrets_path = Path(".streamlit/secrets.toml")
    if not secrets_path.exists():
        print("\n⚠️  IMPORTANT: secrets.toml not found!")
        print("   1. Copy .streamlit/secrets.toml.example to .streamlit/secrets.toml")
        print("   2. Add your actual API keys")
        print("   3. Never commit secrets.toml to version control!")
    else:
        print("✅ secrets.toml exists (remember: never commit this file!)")
    
    return True

def check_requirements():
    """Check if all required files exist"""
    
    required_files = [
        "app.py",
        "utils.py",
        "test.py",
        "requirements.txt",
        "README.md",
        ".gitignore"
    ]
    
    missing_files = []
    
    print("\n📋 Checking required files:")
    for file in required_files:
        if Path(file).exists():
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} - MISSING")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n⚠️  Missing {len(missing_files)} required file(s)")
        print("   Please ensure all core files are present before running the app")
        return False
    
    return True

def install_dependencies():
    """Install Python dependencies"""
    
    print("\n📦 Installing dependencies...")
    
    try:
        import subprocess
        
        # Check if we're in a virtual environment
        if sys.prefix == sys.base_prefix:
            print("⚠️  WARNING: Not in a virtual environment!")
            print("   It's recommended to use a virtual environment:")
            print("   python -m venv venv")
            print("   source venv/bin/activate  # On Windows: venv\\Scripts\\activate")
            response = input("\n   Continue anyway? (y/N): ")
            if response.lower() != 'y':
                return False
        
        # Install requirements
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False
    except FileNotFoundError:
        print("❌ requirements.txt not found!")
        return False

def create_sample_env():
    """Create a sample .env file for local development"""
    
    env_content = """# Environment Variables for Local Development
# Copy to .env and fill in your actual keys

# Required
SERPER_API_KEY=your-serper-api-key-here

# Optional AI Providers
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
GOOGLE_API_KEY=your-google-ai-key-here

# App Configuration
MAX_SERP_QUERIES=50
SERP_BATCH_SIZE=10
DEBUG_MODE=false
"""
    
    if not Path(".env.example").exists():
        with open(".env.example", "w") as f:
            f.write(env_content)
        print("✅ Created .env.example")
    
    if not Path(".env").exists():
        print("📝 .env not found - create one from .env.example and add your API keys")
    
    return True

def display_next_steps():
    """Display next steps for the user"""
    
    print("\n" + "="*60)
    print("🚀 SETUP COMPLETE!")
    print("="*60)
    
    print("\n📋 Next Steps:")
    print("\n1. Configure API Keys:")
    print("   - Copy .streamlit/secrets.toml.example to .streamlit/secrets.toml")
    print("   - Add your Serper API key (required)")
    print("   - Add AI provider keys (optional)")
    
    print("\n2. Test the Setup:")
    print("   python test.py generate  # Generate test data")
    print("   python test.py test      # Run tests")
    
    print("\n3. Run the Application:")
    print("   streamlit run app.py")
    
    print("\n4. Access the App:")
    print("   Open http://localhost:8501 in your browser")
    
    print("\n📚 Documentation:")
    print("   - README.md for detailed instructions")
    print("   - test.py for testing utilities")
    
    print("\n🔑 Get API Keys:")
    print("   - Serper: https://serper.dev (2,500 free queries/month)")
    print("   - OpenAI: https://platform.openai.com/api-keys")
    print("   - Anthropic: https://console.anthropic.com/")
    print("   - Google AI: https://makersuite.google.com/app/apikey")
    
    print("\n⚠️  Security Reminder:")
    print("   - Never commit secrets.toml to version control")
    print("   - Keep your API keys secure")
    print("   - Use environment variables in production")
    
    print("\n" + "="*60)

def main():
    """Main setup function"""
    
    print("="*60)
    print("SEO Cannibalization Analyzer - Setup Script")
    print("="*60)
    
    # Create directory structure
    print("\n📁 Creating directory structure...")
    create_directory_structure()
    
    # Create config files
    print("\n⚙️ Setting up configuration files...")
    create_config_files()
    
    # Check required files
    if not check_requirements():
        print("\n❌ Setup incomplete - missing required files")
        print("   Please ensure all files from the artifacts are present")
        sys.exit(1)
    
    # Create sample .env
    create_sample_env()
    
    # Ask about dependency installation
    print("\n" + "="*60)
    response = input("Would you like to install Python dependencies now? (y/N): ")
    if response.lower() == 'y':
        if not install_dependencies():
            print("⚠️  Dependencies installation failed - please install manually:")
            print("   pip install -r requirements.txt")
    else:
        print("ℹ️  Skipping dependency installation")
        print("   Install manually with: pip install -r requirements.txt")
    
    # Display next steps
    display_next_steps()

if __name__ == "__main__":
    main()
