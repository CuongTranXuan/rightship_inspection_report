"""
Deficiency Risk Classification System for RightShip Inspection Reports

This script automatically classifies deficiencies in inspection reports as High, Medium, or Low risk
based on patterns learned from previously classified deficiencies.
"""

import re
import pandas as pd
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from typing import List, Dict
import openai


def extract_deficiencies_from_pdf(pdf_path: str) -> List[Dict]:
    """Shared utility function to extract deficiencies from PDF inspection report"""
    reader = PdfReader(pdf_path)
    all_text = ""
    for page in reader.pages:
        all_text += page.extract_text() + "\n"

    # Split text into deficiencies using regex
    deficiency_pattern = r'Deficiency\s*(\d+)\s*\n(.*?)(?=Deficiency\s*\d+|\Z)'
    matches = re.findall(deficiency_pattern, all_text, re.DOTALL | re.IGNORECASE)

    deficiencies = []
    for match in matches:
        deficiency_num = int(match[0])
        deficiency_text = match[1].strip()

        # Extract specific parts
        deficiency_desc = _extract_section(deficiency_text, "Deficiency:")
        root_cause = _extract_section(deficiency_text, "Root Cause:")
        corrective_action = _extract_section(deficiency_text, "Corrective action:")
        preventive_action = _extract_section(deficiency_text, "Preventive action:")

        deficiencies.append({
            'number': deficiency_num,
            'description': deficiency_desc,
            'root_cause': root_cause,
            'corrective_action': corrective_action,
            'preventive_action': preventive_action,
            'full_text': deficiency_text
        })

    return deficiencies


def _extract_section(text: str, section_name: str) -> str:
    """Shared utility function to extract a specific section from deficiency text"""
    pattern = rf'{re.escape(section_name)}\s*(.*?)(?=\n[A-Z][a-z]+\s*[a-z]*\s*:|\Z)'
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


class SimpleMLDeficiencyClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.risk_patterns = {
            'high': [
                'fire', 'extinguisher', 'safety equipment', 'rescue boat', 'emergency',
                'rusted seriously', 'seriously', 'critical', 'major defect',
                'weather protection', 'safety critical'
            ],
            'medium': [
                'certificate', 'documentation', 'compliance', 'loadline', 'approval',
                'company name', 'DOC', 'CSR', 'trading certificate', 'master',
                'cross check', 'ship type', 'IOPP', 'class surveyor'
            ],
            'low': [
                'logbook', 'record', 'drill record', 'nav logbook', 'training',
                'forgotten', 'record keeping', 'administrative', 'documentation record'
            ]
        }



    def train_classifier(self, sample_pdf_path, risk_severity_excel_path):
        """Train the classifier using sample data"""

        # Load sample deficiencies
        sample_deficiencies = extract_deficiencies_from_pdf(sample_pdf_path)

        # Load risk severity labels
        risk_df = pd.read_excel(risk_severity_excel_path)
        risk_mapping = dict(zip(risk_df['Deficiency'], risk_df['Risk']))

        # Prepare training data
        texts = []
        labels = []

        for deficiency in sample_deficiencies:
            # Combine all text for feature extraction
            combined_text = f"{deficiency['description']} {deficiency['root_cause']} {deficiency['corrective_action']}"
            texts.append(combined_text)

            # Get corresponding risk label
            risk_label = risk_mapping.get(deficiency['number'], 'Medium')  # Default to Medium
            labels.append(risk_label)

        # Vectorize the text
        X = self.vectorizer.fit_transform(texts)
        y = np.array(labels)

        # Train the classifier
        self.classifier.fit(X, y)

        print(f"Classifier trained on {len(texts)} deficiencies")

    def classify_deficiencies(self, deficiencies):
        """Classify a list of deficiencies"""

        predictions = []
        for def_item in deficiencies:
            # Combine text for classification
            combined_text = f"{def_item['description']} {def_item['root_cause']} {def_item['corrective_action']}"

            # Rule-based classification as backup
            rule_based_risk = self._rule_based_classification(combined_text)

            # ML-based classification
            try:
                X = self.vectorizer.transform([combined_text])
                ml_prediction = self.classifier.predict(X)[0]
                prediction_proba = self.classifier.predict_proba(X)[0]
                confidence = max(prediction_proba)

                # Use ML prediction if confidence is high, otherwise use rule-based
                if confidence > 0.6:
                    final_prediction = ml_prediction
                else:
                    final_prediction = rule_based_risk

            except Exception as e:
                print(f"ML classification failed: {e}")
                final_prediction = rule_based_risk
                confidence = 0.0

            prediction = {
                'deficiency_number': def_item['number'],
                'description': def_item['description'],
                'predicted_risk': final_prediction,
                'rule_based_risk': rule_based_risk,
                'confidence': confidence
            }
            predictions.append(prediction)

        return predictions

    def _rule_based_classification(self, text):
        """Rule-based classification using keyword patterns"""
        text_lower = text.lower()

        # Count pattern matches for each risk level
        high_score = sum(1 for pattern in self.risk_patterns['high'] if pattern in text_lower)
        medium_score = sum(1 for pattern in self.risk_patterns['medium'] if pattern in text_lower)
        low_score = sum(1 for pattern in self.risk_patterns['low'] if pattern in text_lower)

        # Determine risk based on highest score
        if high_score > medium_score and high_score > low_score:
            return 'High'
        elif medium_score > low_score:
            return 'Medium'
        else:
            return 'Low'

    def generate_report(self, predictions, output_file=None):
        """Generate a classification report"""
        report_lines = []
        report_lines.append("DEFICIENCY RISK CLASSIFICATION REPORT")

        for pred in predictions:
            report_lines.append(f"Deficiency {pred['deficiency_number']}: {pred['predicted_risk']}")
            report_lines.append(f"Description: {pred['description']}")
            report_lines.append(f"Rule-based prediction: {pred['rule_based_risk']}")
            report_lines.append(f"Confidence: {pred['confidence']:.2f}")
            report_lines.append("-" * 50)

        # Summary
        risk_counts = {}
        for pred in predictions:
            risk = pred['predicted_risk']
            risk_counts[risk] = risk_counts.get(risk, 0) + 1

        report_lines.append("SUMMARY:")
        for risk, count in risk_counts.items():
            report_lines.append(f"{risk} Risk: {count} deficiencies")

        report_text = "\n".join(report_lines)

        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"Report saved to {output_file}")


class SimpleLLMDeficiencyClassifier:
    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini"):
        """
        Initialize the OpenAI-based classifier

        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY environment variable)
            model: OpenAI model to use (default is gpt-4o-mini)
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.training_examples = []


    def load_training_examples(self, sample_pdf_path: str, risk_severity_excel_path: str):
        """Load training examples from sample data"""

        # Load sample deficiencies
        sample_deficiencies = extract_deficiencies_from_pdf(sample_pdf_path)

        # Load risk severity labels
        risk_df = pd.read_excel(risk_severity_excel_path)
        risk_mapping = dict(zip(risk_df['Deficiency'], risk_df['Risk']))

        # Prepare training examples
        self.training_examples = []
        for deficiency in sample_deficiencies:
            risk_label = risk_mapping.get(deficiency['number'], 'Medium')
            self.training_examples.append({
                'description': deficiency['description'],
                'root_cause': deficiency['root_cause'],
                'corrective_action': deficiency['corrective_action'],
                'risk_level': risk_label
            })

        print(f"Loaded {len(self.training_examples)} training examples")

    def _create_prompt(self, deficiency: Dict) -> str:
        """Create a prompt for OpenAI classification"""

        # Few-shot examples from training data
        examples_text = ""
        for i, example in enumerate(self.training_examples):
            examples_text += f"""
                Example {i+1}:
                Description: {example['description']}
                Root Cause: {example['root_cause']}
                Corrective Action: {example['corrective_action']}
                Classification: {example['risk_level']}
                """

        prompt = f"""You are an expert marine surveyor specializing in ship inspection risk assessment. Classify deficiencies as High, Medium, or Low risk.

            RISK CLASSIFICATION GUIDELINES:
            • HIGH RISK: Safety-critical issues that could lead to immediate danger, environmental damage, or vessel detention
            Examples: Fire safety equipment failures, emergency system malfunctions, structural damage

            • MEDIUM RISK: Compliance issues, certification problems, or equipment deficiencies needing prompt attention
            Examples: Certificate discrepancies, documentation issues, non-critical equipment problems

            • LOW RISK: Administrative issues, minor record-keeping problems, or training deficiencies
            Examples: Logbook entries, routine documentation, minor procedural issues

            TRAINING EXAMPLES:
            {examples_text}

            NEW DEFICIENCY TO CLASSIFY:
            Description: {deficiency['description']}
            Root Cause: {deficiency['root_cause']}
            Corrective Action: {deficiency['corrective_action']}

            Based on the examples and guidelines above, classify this deficiency.

            Respond in this exact format:
            CLASSIFICATION: [High/Medium/Low]
            REASONING: [Brief explanation of why this classification was chosen]"""

        return prompt

    def classify_deficiencies(self, deficiencies):
        """Classify a single deficiency or list of deficiencies using OpenAI"""

        predictions = []
        for i, def_item in enumerate(deficiencies, 1):
            print(f"Classifying deficiency {i}/{len(deficiencies)}...")

            prompt = self._create_prompt(def_item)

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert marine surveyor. Follow the format exactly."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=150,
                    temperature=0.1  # Low temperature for consistent classification
                )

                result = response.choices[0].message.content.strip()

                # Parse the structured response
                classification = "Medium"  # Default
                reasoning = "Could not parse response"

                lines = result.split('\n')
                for line in lines:
                    if line.startswith('CLASSIFICATION:'):
                        classification = line.split(':', 1)[1].strip()
                    elif line.startswith('REASONING:'):
                        reasoning = line.split(':', 1)[1].strip()

                prediction = {
                    'deficiency_number': def_item['number'],
                    'description': def_item['description'],
                    'predicted_risk': classification,
                    'reasoning': reasoning,
                    'raw_response': result
                }

            except Exception as e:
                print(f"OpenAI API error: {e}")

            predictions.append(prediction)

        return predictions

    def generate_report(self, predictions: List[Dict], output_file: str = None) -> str:
        """Generate a classification report"""
        report_lines = []
        report_lines.append("OPENAI-BASED DEFICIENCY RISK CLASSIFICATION REPORT")
        report_lines.append(f"Model: {self.model}")
        report_lines.append(f"Training Examples: {len(self.training_examples)}")

        for pred in predictions:
            report_lines.append(f"Deficiency {pred['deficiency_number']}: {pred['predicted_risk']} Risk")
            report_lines.append(f"Description: {pred['description']}")
            report_lines.append(f"Reasoning: {pred['reasoning']}")
            report_lines.append("-" * 50)

        # Summary
        risk_counts = {'High': 0, 'Medium': 0, 'Low': 0}
        for pred in predictions:
            risk = pred['predicted_risk']
            risk_counts[risk] = risk_counts.get(risk, 0) + 1

        report_lines.append("SUMMARY:")
        for risk, count in risk_counts.items():
            report_lines.append(f"  {risk} Risk: {count} deficiencies")

        total_cost = len(predictions) * 0.01  # Rough estimate
        report_lines.append(f"\nEstimated API Cost: ~${total_cost:.2f}")

        report_text = "\n".join(report_lines)

        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"Report saved to {output_file}")

        return report_text
