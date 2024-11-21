import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, test_images, test_labels):
    """
    Evaluates the performance of the trained model on the test dataset.

    Args:
    - model: Trained model object.
    - test_images: NumPy array of test images.
    - test_labels: NumPy array of true labels corresponding to the test images.

    Returns:
    - Dictionary containing evaluation metrics (accuracy, precision, recall, F1-score).
    """
    # Predict the labels for the test images
    predicted_labels = model.predict(test_images)
    predicted_labels = np.round(predicted_labels).astype(int)

    # Compute evaluation metrics
    accuracy = accuracy_score(test_labels, predicted_labels)
    precision = precision_score(test_labels, predicted_labels)
    recall = recall_score(test_labels, predicted_labels)
    f1 = f1_score(test_labels, predicted_labels)

    # Create a dictionary to store the evaluation metrics
    evaluation_metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1
    }

    return evaluation_metrics

# Example usage
if __name__ == "__main__":
    # Load the trained model
    model = # Load your trained model here
    
    # Load the test dataset
    test_images = # Load your test images here
    test_labels = # Load your test labels here
    
    # Evaluate the model
    evaluation_results = evaluate_model(model, test_images, test_labels)
    
    # Print the evaluation results
    print("Evaluation Results:")
    for metric, value in evaluation_results.items():
        print(f"{metric}: {value}")
