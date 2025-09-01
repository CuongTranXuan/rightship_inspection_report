#!/usr/bin/env python3
"""
Simple LLM Deficiency Classifier - Streamlined version using OpenAI for quick classification
"""

from deficiency_classifier import SimpleLLMDeficiencyClassifier, extract_deficiencies_from_pdf
import os

def main():
    """Simple main function for OpenAI-based classification"""
    print("RightShip OpenAI-Based Deficiency Classifier")

    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: Please set your OpenAI API key:")
        print("    export OPENAI_API_KEY='your-api-key-here'")
        return

    try:
        # Initialize OpenAI-based classifier
        classifier = SimpleLLMDeficiencyClassifier(api_key=api_key, model="gpt-4o-mini")
        print("Loading training examples...")
        classifier.load_training_examples('sample_inspection_report.pdf', 'risk_severity.xlsx')

        # Process new report
        print("Processing new inspection report...")
        deficiencies = extract_deficiencies_from_pdf('new_inspection_report.pdf')
        print(f"Found {len(deficiencies)} deficiencies")

        # Classify deficiencies
        print("Classifying deficiencies using OpenAI...")
        predictions = classifier.classify_deficiencies(deficiencies)
        print("Classification completed")

        # Display results
        print("\nClassification Results:")
        print("-" * 50)

        for pred in predictions:
            print(f"Deficiency {pred['deficiency_number']}: {pred['predicted_risk']} Risk")
            print(f"  Description: {pred['description']}")
            print(f"  Reasoning: {pred['reasoning']}")
            print()

        # Save detailed report
        classifier.generate_report(predictions, 'openai_classification_report.txt')
        print("Detailed report saved to openai_classification_report.txt")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
