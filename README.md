# RightShip Deficiency Classification


## Overview

Here we have 2 scripts process PDF inspection reports and classify deficiencies based on their descriptions, root causes, and corrective actions. Two classification methods are implemented to compare traditional ML approaches with LLM capabilities.

## Classification Methods

### Method 1: OpenAI-Based Classification
Uses GPT models for classification with few-shot learning. Particularly effective with limited training data and provides explainable reasoning for each classification decision.

**Key Features:**
- Few-shot learning with 5 training examples
- Semantic understanding of maritime terminology
- Detailed reasoning for each classification
- Cost: approximately $0.01-0.02 per classification

### Method 2: Traditional ML Classification
Combines Random Forest machine learning with rule-based keyword patterns. Uses TF-IDF vectorization for text processing and includes a fallback system for low-confidence predictions.

**Key Features:**
- Random Forest classifier with TF-IDF features
- Rule-based pattern matching for different risk levels
- Hybrid approach combining ML and rule-based classification
- Local processing without external API dependencies
## Classification Criteria

**High Risk:** Safety-critical issues that could lead to immediate danger, environmental damage, or vessel detention (fire safety equipment failures, emergency system malfunctions, structural damage)

**Medium Risk:** Compliance issues, certification problems, or equipment deficiencies needing prompt attention (certificate discrepancies, documentation issues, non-critical equipment problems)

**Low Risk:** Administrative issues, minor record-keeping problems, or training deficiencies (logbook entries, routine documentation, minor procedural issues)

## Project Structure

```
deficiency_classifier.py      - Main module containing both classification classes
simple_llm_classifier.py      - OpenAI classification wrapper script
simple_classifier.py          - ML classification wrapper script
sample_inspection_report.pdf  - Training data (5 deficiencies)
risk_severity.xlsx            - Risk labels for training data
new_inspection_report.pdf     - Test data for classification
requirements.txt              - Python dependencies
```
## Technical Implementation

### Text Processing
Both methods use a shared PDF extraction function that:
1. Extracts text from PDF pages
2. Uses regex patterns to identify deficiency sections
3. Parses deficiency descriptions, root causes, and corrective actions

### OpenAI Method
1. Loads training examples as few-shot context
2. Creates structured prompts with classification guidelines
3. Uses GPT models to classify with reasoning
4. Handles API errors with fallback responses

### ML Method
1. Vectorizes text using TF-IDF with unigrams and bigrams
2. Trains Random Forest classifier on sample data
3. Applies rule-based classification as backup
4. Combines predictions based on confidence thresholds

## Installation and Setup

```bash
# Install dependencies
pip install -r requirements.txt

# For OpenAI method, set API key
export OPENAI_API_KEY='your-api-key-here'
```

## Testing

Run both classification methods on the provided test data:

```bash
# Test both approaches
python simple_llm_classifier.py  # Generates openai_classification_report.txt
python simple_classifier.py      # Generates classification_report.txt
```

Compare outputs to evaluate the effectiveness of each approach for the given dataset and use case requirements.