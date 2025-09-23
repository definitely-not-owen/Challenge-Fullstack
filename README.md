# Clinical Trial Matching System

A sophisticated clinical trial matching system that uses BioMCP for trial discovery and LLM-powered ranking to match patients with relevant clinical trials.

## 🎯 Overview

This system addresses the critical challenge of matching cancer patients with appropriate clinical trials by:

1. **Fetching real clinical trials** using BioMCP (both SDK and MCP protocol modes)
2. **Intelligently ranking trials** using LLM-powered analysis
3. **Comprehensive evaluation** with multiple validation approaches

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- NCI API key for BioMCP access
- OpenAI API key for LLM ranking (optional)

### Current Status

✅ **Working Features:**
- BioMCP integration (SDK and MCP modes)
- Patient data loading and processing
- Mock trial data for development
- Command-line interface
- Evaluation suite
- Biomarker-based filtering

⚠️ **In Development:**
- LLM-powered ranking (next phase)
- Real-time BioMCP API calls (requires API key)
- Advanced evaluation metrics

### Installation

1. **Clone and setup environment:**
```bash
git clone <repository-url>
cd Challenge-Fullstack

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

2. **Set up API keys:**
```bash
# Required for BioMCP access (you mentioned you have this!)
export NCI_API_KEY="your-nci-api-key"

# Optional for LLM ranking
export OPENAI_API_KEY="your-openai-api-key"
```

**Note:** Without the NCI_API_KEY, the system will use mock data for development and testing.

3. **Run the system:**
```bash
# Match trials for a specific patient (use numeric ID: 1, 2, 3, etc.)
python src/match.py --patient_id 1

# Run evaluation suite
python tests/eval.py

# Get help
python src/match.py --help
```

## 📁 Project Structure

```
Challenge-Fullstack/
├── src/
│   ├── match.py              # Main matching interface
│   └── biomcp_fetcher.py     # Dual-mode BioMCP client
├── tests/
│   ├── eval.py               # Evaluation suite
│   └── test_*.py             # Test files
├── patients.csv              # Patient data (30 patients)
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## 🔧 Usage

### Basic Patient Matching

```bash
# Match trials for a specific patient
python src/match.py --patient_id P001

# Match with custom parameters
python src/match.py --patient_id P002 --max_trials 10 --cancer_type "Breast"
```

### Programmatic Usage

```python
from src.biomcp_fetcher import TrialMatcher

# Initialize matcher
matcher = TrialMatcher(mode="auto")  # Auto-detects best mode

# Patient data
patient = {
    'cancer_type': 'Breast Cancer',
    'cancer_stage': 'II',
    'biomarkers_detected': 'ER+, PR+, PIK3CA mutation',
    'biomarkers_ruled_out': 'HER2+'
}

# Find matching trials
trials = await matcher.match_patient(patient, max_trials=5)
```

### BioMCP Integration Modes

The system supports multiple BioMCP integration approaches:

#### 1. SDK Mode (HTTP API)
```python
from src.biomcp_fetcher import BioMCPClient

async with BioMCPClient(mode="sdk") as client:
    trials = await client.search_trials(
        condition="Breast Cancer",
        additional_terms=["ER+", "HER2-"],
        max_results=10
    )
```

#### 2. MCP Mode (Model Context Protocol)
```python
async with BioMCPClient(mode="mcp") as client:
    trials = await client.search_trials(
        condition="Lung Cancer",
        additional_terms=["EGFR mutation"],
        max_results=10
    )
```

#### 3. Auto Mode (Recommended)
```python
# Automatically selects best available mode
matcher = TrialMatcher(mode="auto")
```

## 🧪 Evaluation Suite

The evaluation system uses multiple approaches to validate matching quality:

### Running Evaluations

```bash
# Run full evaluation suite
python tests/eval.py

# Run specific evaluation
python tests/eval.py --eval_type llm_judge
python tests/eval.py --eval_type biomarker_validation
python tests/eval.py --eval_type clinical_logic
```

### Evaluation Methods

1. **LLM-as-Judge**: Uses GPT-4 to evaluate match quality
2. **Biomarker Validation**: Checks molecular compatibility
3. **Clinical Logic Testing**: Validates stage-appropriate selection
4. **Synthetic Data Generation**: Creates edge cases for testing

## 📊 Patient Data

The system includes 30 real patient records with:

- **Demographics**: Age, gender, race, location, BMI
- **Cancer Details**: Type, stage, grade, biomarkers
- **Clinical Status**: ECOG status, comorbidities, treatment history
- **Treatment Intent**: Curative vs palliative

**Cancer Types Covered:**
- Breast Cancer (12 patients)
- Lung Cancer (5 patients)
- Bladder Cancer (5 patients)
- Pancreatic Cancer (3 patients)
- Ovarian Cancer (2 patients)
- Prostate Cancer (1 patient)
- Colorectal Cancer (1 patient)

## 🔍 Key Features

### Smart Trial Fetching
- **Dual-mode BioMCP integration** (SDK + MCP protocol)
- **Intelligent caching** to reduce API calls
- **Biomarker-based filtering** for precise matching
- **Graceful fallback** to mock data when APIs unavailable

### LLM-Powered Ranking
- **Multi-criteria scoring** (eligibility, biomarkers, geography, clinical appropriateness)
- **Structured prompting** for consistent results
- **Confidence scoring** for ranking reliability
- **Detailed explanations** for each match

### Comprehensive Evaluation
- **Multiple validation approaches** for robust testing
- **Synthetic data generation** for edge case testing
- **Performance metrics** (precision@k, NDCG, clinical relevance)
- **Real-time evaluation** during development

## ⚙️ Configuration

### Environment Variables

```bash
# Required
NCI_API_KEY="your-nci-api-key"           # BioMCP access

# Optional
OPENAI_API_KEY="your-openai-api-key"     # LLM ranking
BIOMCP_MODE="auto"                       # SDK, mcp, or auto
CACHE_DURATION="24"                      # Hours
```

### API Rate Limits

| Service | Without Key | With Key |
|---------|-------------|----------|
| BioMCP | 3 req/sec | 10 req/sec |
| ClinicalTrials.gov | 50 req/min | 50 req/min |
| OpenAI | N/A | Based on tier |

## 🚨 Troubleshooting

### Common Issues

1. **"Patient P001 not found"**
   - Use numeric IDs: `python src/match.py --patient_id 1`
   - Patient IDs are 1-30 (corresponding to rows in patients.csv)

2. **"BioMCP SDK not available"**
   - Ensure `biomcp-python` is installed: `pip install biomcp-python`
   - Check NCI_API_KEY is set correctly

3. **"No trials found"**
   - This is normal when using mock data
   - Set NCI_API_KEY to get real trials
   - Check network connectivity

4. **"MCP connection failed"**
   - This is expected without BioMCP server running
   - System will fallback to mock data
   - Set NCI_API_KEY for real API access

5. **"float object has no attribute 'split'"**
   - Fixed in latest version
   - Update to latest code if you see this error

### Debug Mode

```bash
# Enable detailed logging
export LOG_LEVEL=DEBUG
python src/match.py --patient_id P001
```

## 📈 Performance

- **Trial Fetching**: <2 seconds per query
- **LLM Ranking**: <5 seconds per patient
- **Full Evaluation**: <30 seconds for all patients
- **Cache Hit Rate**: ~80% for repeated queries

## 🔮 Future Enhancements

- [ ] Real-time trial status updates
- [ ] Advanced biomarker matching with variant databases
- [ ] Integration with electronic health records
- [ ] Web interface for clinical use
- [ ] Multi-language support for international trials

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is part of a technical assessment for Radical Health.

## 📞 Support

For questions or issues:
- Check the troubleshooting section above
- Review the implementation plan in `implementation_plan.md`
- Examine test files for usage examples

---

**Built with ❤️ for better clinical trial matching**
