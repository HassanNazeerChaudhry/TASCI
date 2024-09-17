import numpy as np
import torch
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score
from AspectSentimentIntentModel import AspectSentimentIntentModel
import pandas as pd

def load_semeval_datasets():
    # Replace these paths with the actual file paths to your CSV datasets
    restaurant_csv_path = "ds/restaurant_dataset.csv"
    laptop_csv_path = "ds/laptop_dataset.csv"
    twitter_csv_path = "ds/twitter_dataset.csv"

    # Helper function to load a dataset from CSV
    def load_dataset(csv_path):
        # Load the dataset from the CSV file
        df = pd.read_csv(csv_path)

        # Assuming 'input_s_t_minus_1' and 'input_s_t' are stored as comma-separated strings in each cell
        input_s_t_minus_1 = np.array([np.fromstring(seq, sep=',') for seq in df['input_s_t_minus_1']])
        input_s_t = np.array([np.fromstring(seq, sep=',') for seq in df['input_s_t']])

        # Load the labels
        labels = df['label'].values

        # Return the dataset as a dictionary
        return {
            "input_s_t_minus_1": input_s_t_minus_1,
            "input_s_t": input_s_t,
            "labels": labels
        }

    # Load each dataset
    restaurant_dataset = load_dataset(restaurant_csv_path)
    laptop_dataset = load_dataset(laptop_csv_path)
    twitter_dataset = load_dataset(twitter_csv_path)

    return restaurant_dataset, laptop_dataset, twitter_dataset


# Function to evaluate the model on a given dataset
def evaluate_model_on_dataset(model, dataset):
    # Extract the inputs and labels from the dataset
    input_s_t_minus_1 = dataset["input_s_t_minus_1"]
    input_s_t = dataset["input_s_t"]

    # Ensure the inputs are converted to tensors if using a PyTorch model
    input_s_t_minus_1_tensor = torch.tensor(input_s_t_minus_1, dtype=torch.float32)
    input_s_t_tensor = torch.tensor(input_s_t, dtype=torch.float32)

    # Put the model in evaluation mode (important for models with dropout or batch norm layers)
    model.eval()

    # Disable gradient calculation (saves memory and computation during inference)
    with torch.no_grad():
        # Run the model on the inputs
        outputs = model(input_s_t_minus_1_tensor, input_s_t_tensor)

        # Assuming the model outputs logits (raw scores), apply softmax to convert to probabilities
        probabilities = torch.softmax(outputs, dim=-1).numpy()

        # Get the class predictions (the index of the highest probability for each sample)
        predictions = np.argmax(probabilities, axis=-1)

    return predictions

# Function to calculate precision, recall, and F1 scores by class (Positive, Negative, Neutral)
def calculate_class_metrics(true_labels, predictions):
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average=None, labels=[0, 1, 2])
    return precision, recall, f1

# Function to print the detailed performance by sentiment class in LaTeX format
def print_detailed_table(results):
    print("\\begin{table*}[t]")
    print("\\centering")
    print("\\caption{Detailed Performance of TASCI by Sentiment Class}")
    print("\\label{tab:performance_by_class}")
    print("\\resizebox{\\textwidth}{!}{\\begin{tabular}{|l|c|c|c|c|c|c|c|c|c|}")
    print("\\hline")
    print("\\textbf{Dataset} & \\multicolumn{3}{c|}{\\textbf{Positive}} & \\multicolumn{3}{c|}{\\textbf{Negative}} & \\multicolumn{3}{c|}{\\textbf{Neutral}} \\\\")
    print("\\hline")
    print(" & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1} \\\\")
    print("\\hline")
    for dataset_name in ['restaurant', 'laptop', 'twitter']:
        res = results[dataset_name]
        print(f"{dataset_name.capitalize()} & "
              f"{res['precision'][0]:.2f} & {res['recall'][0]:.2f} & {res['f1'][0]:.2f} & "
              f"{res['precision'][1]:.2f} & {res['recall'][1]:.2f} & {res['f1'][1]:.2f} & "
              f"{res['precision'][2]:.2f} & {res['recall'][2]:.2f} & {res['f1'][2]:.2f} \\\\")
        print("\\hline")
    print("\\end{tabular}}")
    print("\\end{table*}")

def main():
    # Load the datasets from SemEval 2014 Task 4
    restaurant_dataset, laptop_dataset, twitter_dataset = load_semeval_datasets()

    # Initialize the models (replace with the actual initialization and pre-trained weights if available)
    tasci_model = AspectSentimentIntentModel(vocab_size=10000, embedding_dim=128, num_heads=8, num_blocks=6,
                                             gru_units=256, intent_units=128)

    # Evaluate on each dataset
    restaurant_predictions = evaluate_model_on_dataset(tasci_model, restaurant_dataset)
    laptop_predictions = evaluate_model_on_dataset(tasci_model, laptop_dataset)
    twitter_predictions = evaluate_model_on_dataset(tasci_model, twitter_dataset)

    # Calculate precision, recall, and F1 for each sentiment class
    restaurant_precision, restaurant_recall, restaurant_f1 = calculate_class_metrics(restaurant_dataset["labels"], restaurant_predictions)
    laptop_precision, laptop_recall, laptop_f1 = calculate_class_metrics(laptop_dataset["labels"], laptop_predictions)
    twitter_precision, twitter_recall, twitter_f1 = calculate_class_metrics(twitter_dataset["labels"], twitter_predictions)

    # Organize results (update with computed metrics)
    results = {
        "restaurant": {"precision": restaurant_precision, "recall": restaurant_recall, "f1": restaurant_f1},
        "laptop": {"precision": laptop_precision, "recall": laptop_recall, "f1": laptop_f1},
        "twitter": {"precision": twitter_precision, "recall": twitter_recall, "f1": twitter_f1}
    }

    # Generate the detailed LaTeX table by sentiment class
    print_detailed_table(results)

if __name__ == '__main__':
    main()
