import numpy as np
import torch
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score
from AspectSentimentIntentModel import AspectSentimentIntentModel
import pandas as pd



# Function to load SemEval datasets from CSV files
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


def calculate_metrics(true_labels, predictions):
    accuracy = accuracy_score(true_labels, predictions)
    macro_f1 = f1_score(true_labels, predictions, average='macro')
    return accuracy, macro_f1


def print_table(results):
    print("\\begin{table*}[t]")
    print("\\centering")
    print("\\caption{Performance Comparison of Various Models}")
    print("\\label{tab:performance_comparison}")
    print("\\resizebox{\\textwidth}{!}{\\begin{tabular}{|l|c|c|c|c|c|c|}")
    print("\\hline")
    print(
        "\\textbf{Model} and \\textbf{Year} & \\multicolumn{2}{c|}{\\textbf{Restaurant}} & \\multicolumn{2}{c|}{\\textbf{Laptop}} & \\multicolumn{2}{c|}{\\textbf{Twitter}} \\\\")
    print("\\hline")
    print(
        " & \\textbf{Accu.} & \\textbf{Macro-F1} & \\textbf{Accu.} & \\textbf{Macro-F1} & \\textbf{Accu.} & \\textbf{Macro-F1} \\\\")
    print("\\hline")
    print(
        f"\\textbf{{TASCI}} (2024) & \\textbf{{{results['restaurant']['accuracy']}}} & \\textbf{{{results['restaurant']['macro_f1']}}} & "
        f"\\textbf{{{results['laptop']['accuracy']}}} & \\textbf{{{results['laptop']['macro_f1']}}} & "
        f"\\textbf{{{results['twitter']['accuracy']}}} & \\textbf{{{results['twitter']['macro_f1']}}} \\\\")
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

    # Calculate metrics for each dataset
    restaurant_accuracy, restaurant_macro_f1 = calculate_metrics(restaurant_dataset["labels"], restaurant_predictions)
    laptop_accuracy, laptop_macro_f1 = calculate_metrics(laptop_dataset["labels"], laptop_predictions)
    twitter_accuracy, twitter_macro_f1 = calculate_metrics(twitter_dataset["labels"], twitter_predictions)

    # Organize results (update with computed metrics)
    results = {
        "restaurant": {"accuracy": round(restaurant_accuracy * 100, 2),
                       "macro_f1": round(restaurant_macro_f1 * 100, 2)},
        "laptop": {"accuracy": round(laptop_accuracy * 100, 2), "macro_f1": round(laptop_macro_f1 * 100, 2)},
        "twitter": {"accuracy": round(twitter_accuracy * 100, 2), "macro_f1": round(twitter_macro_f1 * 100, 2)}
    }

    # Generate the LaTeX table
    print_table(results)


if __name__ == '__main__':
    main()
