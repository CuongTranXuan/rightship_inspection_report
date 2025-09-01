#!/usr/bin/env python3
"""
Simple Deficiency Classifier - Streamlined version for quick classification
"""

from deficiency_classifier import SimpleMLDeficiencyClassifier, extract_deficiencies_from_pdf

def main():
    """Simple main function for quick classification"""
    print("RightShip Deficiency Classifier Using ML")

    try:
        # Initialize and train classifier
        classifier = SimpleMLDeficiencyClassifier()
        print("Training classifier...")
        classifier.train_classifier('sample_inspection_report.pdf', 'risk_severity.xlsx')

        # Process new report
        print("Processing new inspection report...")
        deficiencies = extract_deficiencies_from_pdf('new_inspection_report.pdf')
        predictions = classifier.classify_deficiencies(deficiencies)

        # Display results
        print(f"\nFound {len(predictions)} deficiencies:")
        print("-" * 40)

        for pred in predictions:
            print(f"Deficiency {pred['deficiency_number']}: {pred['predicted_risk']} Risk")
            print(f"  Description: {pred['description']}")
            print(f"  Confidence: {pred['confidence']:.2f}")
            print()

        # Save detailed report
        classifier.generate_report(predictions, 'ml_classification_report.txt')
        print("Detailed report saved to ml_classification_report.txt")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
