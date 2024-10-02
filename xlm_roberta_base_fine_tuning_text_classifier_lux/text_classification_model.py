import os
import pandas as pd
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset
import torch
import numpy as np

class TextClassificationModel:
    def __init__(self, model_name='FacebookAI/xlm-roberta-base', cache_dir=None):
        """
        Initializes the model, tokenizer, and device.
        :param model_name: Name of the pre-trained XLM-RoBERTa model (default: FacebookAI/xlm-roberta-base)
        :param cache_dir: Directory where the model cache is stored (default: None)
        """
        self.model_name = model_name
        self.cache_dir = cache_dir if cache_dir else os.getcwd()
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(model_name, cache_dir=self.cache_dir)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.df = None  # DataFrame to store the data
        self.class_mapping = {}  # Dictionary to map class labels to indices

    def load_data(self, file_path):
        """Loads the data from a CSV or Excel file."""
        if file_path.endswith('.csv'):
            self.df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            self.df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")
        print("Columns found in the dataset:", self.df.columns.tolist())
        return self.df.columns.tolist()

    def set_columns(self, label_column, text_columns, update_class_mapping=False):
        """
        Sets the label and text columns for training, validation, or testing.
        :param label_column: The column containing the target labels
        :param text_columns: The columns containing text data
        :param update_class_mapping: Boolean to indicate if the class mapping should be updated (only for training data)
        :return: Processed text and label data (or just text data if labels are unavailable)
        """
        # Check if the selected columns exist in the dataset
        if label_column not in self.df.columns or any(col not in self.df.columns for col in text_columns):
            raise ValueError("Invalid column selection. Ensure the columns exist in the dataset.")
        
        # Concatenate multiple text columns into one
        if len(text_columns) > 1:
            text_data = self.df[text_columns].apply(lambda row: ' '.join(row.values.astype(str)), axis=1).tolist()
        else:
            text_data = self.df[text_columns[0]].astype(str).tolist()

        # Handle the label column (only if label_column is provided)
        if label_column in self.df.columns and update_class_mapping:
            unique_labels = sorted(self.df[label_column].unique())
            print(f"Text labels found, converting them to numeric: {unique_labels}")
            self.class_mapping = {label: idx for idx, label in enumerate(unique_labels)}
            y_data = self.df[label_column].map(self.class_mapping).tolist()
        elif label_column in self.df.columns:
            y_data = self.df[label_column].map(self.class_mapping).tolist()
        else:
            y_data = None  # No labels for predictions

        return text_data, y_data

    def tokenize_data(self, texts):
        """Tokenizes the text data using the XLM-RoBERTa tokenizer."""
        return self.tokenizer(texts, truncation=True, padding=True, max_length=512)

    def compute_metrics(self, p):
        """Computes metrics (accuracy, precision, recall, f1-score) during evaluation."""
        labels = p.label_ids
        preds = p.predictions.argmax(-1)
        precision, recall, f1, support = precision_recall_fscore_support(labels, preds, average=None)
        acc = accuracy_score(labels, preds)

        # Convert numpy arrays to lists before returning
        return {
            'accuracy': float(acc),
            'precision_per_class': precision.tolist(),
            'recall_per_class': recall.tolist(),
            'f1_per_class': f1.tolist(),
            'support_per_class': support.tolist(),
            'precision': precision.mean(),  # Mean precision across classes
            'recall': recall.mean(),        # Mean recall across classes
            'f1': f1.mean()                 # Mean F1 across classes
        }


    def create_dataset(self, encodings, labels):
        """Creates a PyTorch dataset from encodings and labels."""
        class CustomDataset(Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
                return item

        return CustomDataset(encodings, labels)

    def train(self, text_train, y_train, text_test, y_test, save_model_path='./saved_model'):
        """Trains the model using the given training and testing data."""
        train_encodings = self.tokenize_data(text_train)
        test_encodings = self.tokenize_data(text_test)

        train_dataset = self.create_dataset(train_encodings, y_train)
        test_dataset = self.create_dataset(test_encodings, y_test)

        model = XLMRobertaForSequenceClassification.from_pretrained(self.model_name, num_labels=len(set(y_train)), cache_dir=self.cache_dir).to(self.device)

        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            evaluation_strategy="epoch",
            logging_dir='./logs',
            logging_steps=10,
            save_strategy="epoch",
            load_best_model_at_end=True
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=self.compute_metrics
        )

        trainer.train()
        model.save_pretrained(save_model_path)
        print(f"Model saved to {save_model_path}")

        eval_results = trainer.evaluate()
        self.print_class_wise_metrics(eval_results)

        return eval_results

    def load_and_evaluate(self, save_model_path, text_test, y_test):
        """Loads a pre-trained model and evaluates it on the given test data."""
        model = XLMRobertaForSequenceClassification.from_pretrained(save_model_path, num_labels=len(set(y_test))).to(self.device)

        test_encodings = self.tokenize_data(text_test)
        test_dataset = self.create_dataset(test_encodings, y_test)

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False)

        model.eval()

        predictions, true_labels = [], []
        for batch in test_loader:
            with torch.no_grad():
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                outputs = model(**inputs)
                logits = outputs.logits
                predictions.append(logits.argmax(-1).cpu().numpy())
                true_labels.append(batch['labels'].cpu().numpy())

        predictions = np.concatenate(predictions)
        true_labels = np.concatenate(true_labels)

        precision, recall, f1, support = precision_recall_fscore_support(true_labels, predictions, average=None)
        acc = accuracy_score(true_labels, predictions)

        eval_results = {
            'accuracy': float(acc),
            'precision_per_class': precision.tolist(),
            'recall_per_class': recall.tolist(),
            'f1_per_class': f1.tolist(),
            'support_per_class': support.tolist()
        }

        self.print_class_wise_metrics(eval_results)

        return eval_results

    def predict(self, text_test, y_test, save_predictions_path, save_model_path):
        """Generates predictions for the test dataset and saves them with true labels."""
        model = XLMRobertaForSequenceClassification.from_pretrained(save_model_path, num_labels=len(self.class_mapping)).to(self.device)

        test_encodings = self.tokenize_data(text_test)
        test_dataset = self.create_dataset(test_encodings, [0]*len(text_test))

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False)

        model.eval()

        predictions = []
        for batch in test_loader:
            with torch.no_grad():
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                outputs = model(**inputs)
                logits = outputs.logits
                predictions.append(logits.argmax(-1).cpu().numpy())

        predictions = np.concatenate(predictions)

        predicted_labels = [list(self.class_mapping.keys())[list(self.class_mapping.values()).index(pred)] for pred in predictions]
        true_labels = [list(self.class_mapping.keys())[list(self.class_mapping.values()).index(true)] for true in y_test]

        df_predictions = pd.DataFrame({
            'Text': [' '.join(text) if isinstance(text, list) else text for text in text_test],
            'True Label': true_labels,
            'Predicted Label': predicted_labels
        })

        try:
            df_predictions.to_excel(save_predictions_path, index=False)
            print(f"Predictions saved to (Excel) {save_predictions_path}")
        except:
            df_predictions.to_csv(save_predictions_path, index=False)
            print(f"Predictions saved to (CSV) {save_predictions_path}")

        return df_predictions

    def print_class_wise_metrics(self, eval_results):
        """
        Prints class-wise precision, recall, F1-score, and support.
        :param eval_results: The evaluation results containing metrics per class
        """
        precision, recall, f1, support = eval_results['eval_precision_per_class'], eval_results['eval_recall_per_class'], eval_results['eval_f1_per_class'], eval_results['eval_support_per_class']

        print("\nClass-wise Precision, Recall, F1-score, and Support:")
        print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-score':<10} {'Support':<10}")
        
        for idx, class_name in enumerate(self.class_mapping.keys()):
            print(f"{class_name:<15} {precision[idx]:<10.4f} {recall[idx]:<10.4f} {f1[idx]:<10.4f} {support[idx]:<10}")

            # Optional: Log each class-wise metric individually to Tensorboard or other logs
            # if you are using Tensorboard or another logging tool.
            # e.g.:
            # self.trainer.log({'precision_class_{}'.format(class_name): precision[idx]})
            # self.trainer.log({'recall_class_{}'.format(class_name): recall[idx]})
            # self.trainer.log({'f1_class_{}'.format(class_name): f1[idx]})


    def evaluate_predictions_from_file(self, predictions_file_path):
        """Loads predictions from a file and compares predicted labels with true labels to generate a classification report."""
        if predictions_file_path.endswith('.csv'):
            df_predictions = pd.read_csv(predictions_file_path)
        elif predictions_file_path.endswith('.xlsx'):
            df_predictions = pd.read_excel(predictions_file_path)
        else:
            raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")

        true_labels = df_predictions['True Label']
        predicted_labels = df_predictions['Predicted Label']

        precision, recall, f1, support = precision_recall_fscore_support(true_labels, predicted_labels, average=None, labels=list(self.class_mapping.keys()))
        accuracy = accuracy_score(true_labels, predicted_labels)

        print("\nClass-wise Precision, Recall, F1-score, and Support:")
        print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-score':<10} {'Support':<10}")
        for idx, class_name in enumerate(self.class_mapping.keys()):
            print(f"{class_name:<15} {precision[idx]:<10.4f} {recall[idx]:<10.4f} {f1[idx]:<10.4f} {support[idx]:<10}")

        print(f"\nOverall Accuracy: {accuracy:.4f}")

        return {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1_score': f1.tolist(),
            'support': support.tolist(),
            'accuracy': accuracy
        }
