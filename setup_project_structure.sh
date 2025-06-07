#!/bin/bash

# Data Quality Tool - Project Structure Setup Script
# Run this from the root of your repository

echo "🚀 Setting up Data Quality Tool project structure..."

# Create main directories
echo "📁 Creating directory structure..."
mkdir -p core
mkdir -p checkers/biobank
mkdir -p reports
mkdir -p utils
mkdir -p web/templates
mkdir -p tests/sample_data

# Create core module files
echo "📄 Creating core module files..."
touch core/__init__.py
touch core/base_checker.py
touch core/scoring_engine.py
touch core/data_profiler.py
touch core/report_generator.py

# Create checker module files
echo "🔍 Creating checker module files..."
touch checkers/__init__.py
touch checkers/completeness.py
touch checkers/consistency.py
touch checkers/validity.py
touch checkers/uniqueness.py

# Create biobank specific checker files
echo "🧬 Creating biobank-specific checker files..."
touch checkers/biobank/__init__.py
touch checkers/biobank/participant_id.py
touch checkers/biobank/clinical_data.py
touch checkers/biobank/sample_tracking.py

# Create report module files
echo "📊 Creating report module files..."
touch reports/__init__.py
touch reports/quality_report.py
touch reports/comparative_report.py

# Create utility files
echo "🛠️ Creating utility files..."
touch utils/__init__.py
touch utils/data_loader.py

# Create web interface files
echo "🌐 Creating web interface files..."
touch web/app.py
touch web/templates/index.html
touch web/templates/report.html

# Create test files
echo "🧪 Creating test files..."
touch tests/__init__.py
touch tests/test_checkers.py
touch tests/test_core.py
touch tests/conftest.py

# Create sample data placeholders
echo "📋 Creating sample data placeholders..."
touch tests/sample_data/sample_biobank.csv
touch tests/sample_data/sample_dirty_data.csv
touch tests/sample_data/sample_clean_data.csv

# Create configuration files
echo "⚙️ Creating configuration files..."
touch pyproject.toml
touch .gitignore
touch requirements.txt

# Create documentation files
echo "📚 Creating documentation files..."
touch README.md
touch CONTRIBUTING.md
touch LICENSE

# Create example/demo files
echo "🎯 Creating demo files..."
mkdir -p examples
touch examples/demo_basic.py
touch examples/demo_biobank.py
touch examples/demo_comparison.py

echo "✅ Project structure created successfully!"
echo ""
echo "📁 Directory structure:"
echo "├── core/                    # Core framework"
echo "├── checkers/               # Quality checkers"
echo "│   └── biobank/            # Biobank-specific checkers"
echo "├── reports/                # Report generation"
echo "├── utils/                  # Utility functions"
echo "├── web/                    # Web interface"
echo "│   └── templates/          # HTML templates"
echo "├── tests/                  # Test files"
echo "│   └── sample_data/        # Test datasets"
echo "├── examples/               # Demo scripts"
echo "└── *.toml, *.md           # Config and docs"
echo ""
echo "🎯 Next steps:"
echo "1. Initialize uv project: uv init"
echo "2. Add dependencies: uv add pandas numpy streamlit plotly"
echo "3. Start implementing the base classes in core/"
echo "4. Assign team members to different modules"