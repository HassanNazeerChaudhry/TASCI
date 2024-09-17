import numpy as np
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from AspectExtractionModel import AspectExtractionModel
from AspectSentimentIntentModel import AspectSentimentIntentModel
from GRUSeq2SeqModel import GRUSeq2SeqModel
from SelfAttention import SelfAttention


# Function to load SemEval datasets from CSV files
def load_semeval_datasets():
    restaurant_csv_path = "ds/restaurant_dataset.csv"
    laptop_csv_path = "ds/laptop_dataset.csv"
    twitter_csv_path = "ds/twitter_dataset.csv"

    def load_dataset(csv_path):
        df = pd.read_csv(csv_path)
        input_s_t_minus_1 = np.array([np.fromstring(seq, sep=',') for seq in df['input_s_t_minus_1']])
        input_s_t = np.array([np.fromstring(seq, sep=',') for seq in df['input_s_t']])
        labels = df['label'].values
        return {
            "input_s_t_minus_1": input_s_t_minus_1,
            "input_s_t": input_s_t,
            "labels": labels
        }

    restaurant_dataset = load_dataset(restaurant_csv_path)
    laptop_dataset = load_dataset(laptop_csv_path)
    twitter_dataset = load_dataset(twitter_csv_path)

    return {
        "restaurant": restaurant_dataset,
        "laptop": laptop_dataset,
        "twitter": twitter_dataset
    }


# Function to evaluate the model on a given dataset
def evaluate_model_on_dataset(model, dataset):
    input_s_t_minus_1 = dataset["input_s_t_minus_1"]
    input_s_t = dataset["input_s_t"]

    input_s_t_minus_1_tensor = torch.tensor(input_s_t_minus_1, dtype=torch.float32)
    input_s_t_tensor = torch.tensor(input_s_t, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        outputs = model(input_s_t_minus_1_tensor, input_s_t_tensor)
        probabilities = torch.softmax(outputs, dim=-1).numpy()
        predictions = np.argmax(probabilities, axis=-1)

    return predictions


# Function to calculate evaluation metrics
def calculate_metrics(true_labels, predictions):
    accuracy = accuracy_score(true_labels, predictions)
    macro_f1 = f1_score(true_labels, predictions, average='macro')
    return accuracy, macro_f1


# Function to print results table in LaTeX format
def print_table(results):
    print("\\begin{table*}[t]")
    print("\\centering")
    print("\\caption{Ablation Study of TASCI Components}")
    print("\\label{tab:ablation_study}")
    print("\\resizebox{\\textwidth}{!}{\\begin{tabular}{|l|c|c|c|c|c|c|}")
    print("\\hline")
    print(
        "\\textbf{Model Variation} & \\multicolumn{2}{c|}{\\textbf{Restaurant}} & \\multicolumn{2}{c|}{\\textbf{Laptop}} & \\multicolumn{2}{c|}{\\textbf{Twitter}} \\\\")
    print("\\hline")
    print(
        " & \\textbf{Accu.} & \\textbf{Macro-F1} & \\textbf{Accu.} & \\textbf{Macro-F1} & \\textbf{Accu.} & \\textbf{Macro-F1} \\\\")
    print("\\hline")

    for variation, result in results.items():
        print(f"{variation} & {result['restaurant']['accuracy']} & {result['restaurant']['macro_f1']} & "
              f"{result['laptop']['accuracy']} & {result['laptop']['macro_f1']} & "
              f"{result['twitter']['accuracy']} & {result['twitter']['macro_f1']} \\\\")
        print("\\hline")

    print("\\end{tabular}}")
    print("\\end{table*}")


# Main function to load datasets, evaluate models, and print results
def main():
    # Load the datasets from SemEval
    datasets = load_semeval_datasets()

    # Initialize the models
    aspect_extraction_model = AspectExtractionModel(vocab_size=10000, embedding_dim=128, num_heads=8, num_blocks=6)
    sentiment_intent_model = AspectSentimentIntentModel(vocab_size=10000, embedding_dim=128, num_heads=8, num_blocks=6,
                                                        gru_units=256, intent_units=128)
    gru_seq2seq_model = GRUSeq2SeqModel(input_dim=128, hidden_dim=256, output_dim=128)
    self_attention_model = SelfAttention(embedding_dim=128, num_heads=8)

    # Dictionary to store results for different variations of the model
    results = {}

    # List of variations for evaluation
    model_variations = {
        "TASCI with Aspect Extraction Model": aspect_extraction_model,
        "TASCI with Sentiment & Intent Model": sentiment_intent_model,
        "TASCI with GRUSeq2Seq Model": gru_seq2seq_model,
        "TASCI with Self-Attention Model": self_attention_model
    }

    # Loop through each model variation and evaluate on all datasets
    for variation_name, model in model_variations.items():
        variation_results = {}
        for dataset_name, dataset in datasets.items():
            predictions = evaluate_model_on_dataset(model, dataset)
            accuracy, macro_f1 = calculate_metrics(dataset["labels"], predictions)
            variation_results[dataset_name] = {"accuracy": round(accuracy * 100, 2),
                                               "macro_f1": round(macro_f1 * 100, 2)}

        results[variation_name] = variation_results

    # Print the results table in LaTeX format
    print_table(results)


if __name__ == '__main__':
    main()
