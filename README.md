# DirtyDataScan

An automated tool for assessing data quality with specialized support for biobank datasets. Provides a "dirty score" (0-100) and actionable recommendations to improve data quality and determine "insight readiness".

## 🎯 Project Overview

This tool was developed during a 2-day hackathon to create a comprehensive data quality assessment framework that:

- **Calculates Quality Scores**: Overall dirty score (0-100) where higher = better quality
- **Identifies Issues**: Specific problems with actionable recommendations  
- **Supports Comparison**: Compare quality across multiple datasets
- **Biobank Specialized**: Domain-specific validation for biobank data
- **Insight Readiness**: Determines if data is ready for analysis

## 🏗️ Architecture

### Core Components

- **Core Framework** (`core/`): Base classes and scoring engine
- **Quality Checkers** (`checkers/`): Modular validation components
- **Biobank Extensions** (`checkers/biobank/`): Domain-specific validators
- **Reports** (`reports/`): Quality assessment and comparison reports
- **Web Interface** (`web/`): Streamlit-based user interface

### Quality Dimensions

1. **Completeness**: Missing data detection
2. **Consistency**: Data type and format validation  
3. **Validity**: Range and constraint checking
4. **Uniqueness**: Duplicate detection
5. **Biobank-Specific**: Participant IDs, clinical ranges, sample tracking

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- UV package manager

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd data-quality-tool

# Install dependencies with UV
uv sync

# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
```

### Basic Usage

```python
from core.scoring_engine import ScoringEngine
from checkers import CompletenessChecker, UniquenessChecker
import pandas as pd

# Load your data
df = pd.read_csv('your_data.csv')

# Initialize scoring engine
engine = ScoringEngine()

# Run quality checks
completeness = CompletenessChecker(weight=1.5)
uniqueness = UniquenessChecker(weight=1.0)

engine.add_result(completeness.check(df))
engine.add_result(uniqueness.check(df))

# Get overall score
overall_score = engine.calculate_overall_score()
print(f"Data Quality Score: {overall_score:.1f}/100")
```

### Web Interface

```bash
# Start the web application
uv run streamlit run web/app.py
```

## 📊 Demo Examples

See the `examples/` directory for demonstration scripts:

- `demo_basic.py`: Basic quality assessment
- `demo_biobank.py`: Biobank-specific validation
- `demo_comparison.py`: Compare multiple datasets

## 🧬 Biobank Features

### Specialized Validators

- **Participant ID Validation**: Format compliance and uniqueness
- **Clinical Data Ranges**: Age, weight, BMI, blood pressure validation
- **Sample Tracking**: Chain of custody and integrity checks

### Usage

```python
from checkers.biobank import ParticipantIDChecker, ClinicalDataChecker

# Validate participant IDs with custom pattern
id_checker = ParticipantIDChecker(id_pattern=r'^BB-\d{6}$')
result = id_checker.check(biobank_df, 'participant_id')

# Check clinical data ranges
clinical_checker = ClinicalDataChecker()
result = clinical_checker.check(biobank_df)
```

## 🛠️ Development

### Team Workflow (Hackathon Setup)

The project is designed for 4-person parallel development:

1. **Person 1**: Core framework (`core/` modules)
2. **Person 2**: Basic checkers (`checkers/*.py`)  
3. **Person 3**: Biobank extensions (`checkers/biobank/`)
4. **Person 4**: Reports & interface (`reports/`, `web/`)

### Adding Custom Checkers

```python
from core.base_checker import BaseQualityChecker, QualityResult

class CustomChecker(BaseQualityChecker):
    @property
    def name(self) -> str:
        return "Custom Validation"
    
    def check(self, df, column=None) -> QualityResult:
        # Your validation logic here
        return QualityResult(
            checker_name=self.name,
            score=95.0,
            weight=self.weight,
            issues_found=[],
            recommendations=[]
        )
```

### Testing

```bash
# Run all tests
uv run pytest

# Run specific test modules
uv run pytest tests/test_checkers.py
uv run pytest tests/test_core.py
```

## 📋 Quality Score Interpretation

| Score Range | Quality Level | Insight Readiness | Action Required |
|-------------|---------------|-------------------|-----------------|
| 80-100      | Excellent     | ✅ Ready          | Minor cleanup   |
| 60-79       | Good          | ✅ Ready          | Some fixes needed |
| 40-59       | Fair          | ⚠️ Caution        | Significant work |
| 0-39        | Poor          | ❌ Not Ready      | Major cleanup   |

## 🔧 Configuration

### Custom Validation Rules

```python
# Configure validity checker with custom rules
from checkers.validity import ValidityChecker

custom_rules = {
    'age': {'min_value': 0, 'max_value': 120},
    'income': {'min_value': 0, 'max_value': 1000000}
}

validator = ValidityChecker(custom_rules=custom_rules)
```

### Biobank ID Patterns

```python
# Custom participant ID format
from checkers.biobank import ParticipantIDChecker

checker = ParticipantIDChecker(id_pattern=r'^[A-Z]{3}-\d{8}$')
```

## 📈 Sample Output

```
DATA QUALITY ASSESSMENT REPORT
==================================================

Overall Quality Score: 87.3/100 (Excellent)
Insight Readiness: ✅ Ready

SCORE BREAKDOWN:
--------------------
Completeness: 95.2/100
Uniqueness: 88.7/100
Consistency: 82.1/100
Validity: 91.5/100
Participant ID Validation: 100.0/100

ISSUES FOUND: 12 total
--------------------
1. [Completeness] Column 'phone_number' has 45 missing values
2. [Uniqueness] Dataset has 3 duplicate rows
3. [Consistency] Column 'age' contains mixed data types

RECOMMENDATIONS:
--------------------
1. Investigate missing data in column 'phone_number'
2. Remove or consolidate duplicate entries
3. Standardize data entry formats
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests for new functionality
5. Submit a pull request

See `CONTRIBUTING.md` for detailed guidelines.

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## 🙋‍♀️ Support

For questions and support:
- Open an issue on GitHub
- Check the `examples/` directory for usage patterns
- Review the docstrings in the core modules

## 🎯 Roadmap

### Hackathon Goals (24 hours)
- [x] Core framework and base classes
- [x] Basic quality checkers (completeness, uniqueness, consistency)
- [x] Biobank-specific validators
- [x] Simple web interface
- [x] Demo with sample data

### Future Enhancements
- [ ] Advanced statistical outlier detection
- [ ] Machine learning-based quality prediction
- [ ] Integration with common data platforms
- [ ] Advanced visualization dashboards
- [ ] Export to PDF reports
- [ ] API endpoints for programmatic access

---

*Built with ❤️ during a 2-day hackathon for better biobank data quality*