# XLM-RoBERTa Fine-Tuning Text Classifier - Lux
[![Downloads](https://static.pepy.tech/badge/xlm-roberta-base-fine-tuning-text-classifier-lux)](https://pepy.tech/project/xlm-roberta-base-fine-tuning-text-classifier-lux)

**XLM-RoBERTa Fine-Tuning Text Classifier - Lux** is a high-performance library designed for fine-tuning pre-trained FacebookAI/xlm-roberta-base models on multilingual datasets (French, German, English, and Luxembourgish). XLM-RoBERTa is a robust, multilingual variant of RoBERTa, trained on a large-scale dataset covering 100 languages (refer [HERE](https://huggingface.co/FacebookAI/xlm-roberta-base)), making it highly suitable for tasks involving multilingual text. 
The library provides a streamlined interface for loading your dataset, fine-tuning the XLM-RoBERTa model, and evaluating it using key metrics like accuracy, precision, recall, and F1-score.
For more information, study our example on [Google Colab](https://colab.research.google.com/drive/11UJWhM_nZWLBJ3pkx0FpLnGA7xowG0KV?usp=sharing)

## Table of Contents
- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [Fine-tuning the Model](#fine-tuning-the-model)
- [Validation and Testing](#validation-and-testing)
- [Results](#results)
- [Authors](#authors)
- [License](#license)
- [Example Usage](#example-usage)


## Key Features

- **XLM-RoBERTa Fine-tuning**: Fine-tune a pre-trained XLM-RoBERTa model on your custom text classification task.
- **Multilingual Support**: Works seamlessly with datasets in French, German, English, and Luxembourgish.
- **Comprehensive Metrics**: Evaluates the model using precision, recall, F1-score, and accuracy per class.
- **GPU Acceleration**: Supports CUDA-enabled devices for faster training.

## Quick Start

The primary interface for interacting with this library is through the TextClassificationModel class, which allows you to load data, fine-tune the model, and evaluate it on test data.


## Fine-tuning the Model

To work with multiple languages, we will be using the XLM-RoBERTa model for fine-tuning.



### 1. Create an Instance of the Model:

```python
from text_classifier import TextClassificationModel

# Initialize the multilingual XLM-RoBERTa model
model = TextClassificationModel()
```
### 2. Load and Prepare Your Data:

You need to provide the paths to your training, validation, and test datasets. These files can be in CSV or Excel format. 
Ensure that the selected columns for training, validation, and testing do not contain null values.

```python
# Load the training data
train_file_path = "/path/to/your/training_file.csv"
train_columns = model.load_data(train_file_path)

# Specify label and text columns
label_column = 'label'
text_columns = ['text_column1', 'text_column2']

# Set columns for training
text_train, y_train = model.set_columns(label_column, text_columns, update_class_mapping=True)

```
### 3. Load Validation and Test Data:

Repeat the steps to load the validation and test datasets:

```python
# Load validation data
val_file_path = "/path/to/your/validation_file.csv"
text_val, y_val = model.load_data(val_file_path)

# Load test data
test_file_path = "/path/to/your/test_file.csv"
text_test, y_test = model.load_data(test_file_path)
```
### 4. Train and Fine-Tune the Model:

You can now fine-tune the model using your dataset:

```python
# Fine-tune the model
save_model_path = './saved_model'
eval_results = model.train(text_train, y_train, text_val, y_val, save_model_path)

print("Evaluation results after fine-tuning:", eval_results)

# Fine-tune the model
save_model_path = './saved_model'
eval_results = model.train(text_train, y_train, text_val, y_val, save_model_path)

print("Evaluation results after fine-tuning:", eval_results)
```
 ### 5. Generate Predictions:
 ```python
 # Get the path from the user to save the predictions
save_predictions_path = input("Please enter the path to save the predictions (default: ./predictions.csv): ").strip() or './predictions.csv'

# Generate predictions and save them along with the true labels
predictions_df = model.predict(text_test, y_test, save_predictions_path, save_model_path)

print("Predictions saved in the file:", predictions_df.head())
```

### Results

During fine-tuning, the following key metrics are displayed after each epoch:

- **Training Loss**: Tracks model performance on the training dataset.
- **Validation Loss**: Tracks performance on the validation dataset to avoid overfitting.
- **Accuracy**: The overall accuracy of the model.
- **Precision, Recall, and F1-score Per Class**: Class-specific performance metrics.
- **Support Per Class**: The number of samples for each class.

### Example output after fine-tuning:

```plaintext
Epoch    Training Loss    Validation Loss    Accuracy
1        0.329800         0.284320           0.917459
2        0.154100         0.158046           0.964482
3        0.050600         0.126580           0.965983

```

### Precision, Recall, F1-score Per Class:

| Class   | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| class_1 | 0.9831    | 0.9949 | 0.9890   | 586     |
| class_2 | 0.9351    | 0.9105 | 0.9227   | 380     |

### Authors

- Mehrdad ALMASI (email: mehrdad.al.2023@gmail.com)
- Demival VASQUES FILHO (email: demival.vasques@uni.lu)

### License

This project is licensed under the MIT License - see the LICENSE file for details.

## Example Usage 

This section provides a complete example of how to load a dataset, split it into training, validation, and test sets, fine-tune the XLM-RoBERTa model, and evaluate it.


### 1. Build and Split the Dataset

If you do not have a dataset, we will build it together using the BNL newspapers dataset. If you already have a dataset, you can skip to the section where you input the training dataset path.

The example below demonstrates how to use the dataset, split it into training, validation, and test sets using stratified sampling, and convert it into a Pandas DataFrame.

Make sure the selected columns for training, validation, and testing do not contain null values.

Our example is available on [Google Colab](https://colab.research.google.com/drive/11UJWhM_nZWLBJ3pkx0FpLnGA7xowG0KV?usp=sharing)


```python

import pandas as pd
from sklearn.model_selection import train_test_split
from xlm_roberta_base_fine_tuning_text_classifier_lux import TextClassificationModel
from datasets import load_dataset

# Ask the user if they have a dataset or want to build one
build_dataset = input("Do you want to build a new dataset? (yes/no): ").strip().lower()

if build_dataset == 'yes':
    # Step 1: Build and Split the Dataset
    # Load the dataset
    dataset = load_dataset("biglam/bnl_newspapers1841-1879")

    # Extract columns
    data = pd.DataFrame({
        "identifier": dataset["train"]["identifier"],
        "date": dataset["train"]["date"],
        "metsType": dataset["train"]["metsType"],
        "newspaperTitle": dataset["train"]["newpaperTitle"],
        "paperID": dataset["train"]["paperID"],
        "publisher": dataset["train"]["publisher"],
        "text": dataset["train"]["text"],
        "creator": dataset["train"]["creator"],
        "article_type": dataset["train"]["type"]  # Target for stratified sampling
    })

    # Reduce the size of the data for demonstration purposes (optional)
    data = data.sort_values(by="article_type", ascending=False)
    data = data.head(1000)

    # Step 2: Split the dataset into training, validation, and test sets
    train_data, temp_data = train_test_split(data, test_size=0.2, stratify=data['article_type'], random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, stratify=temp_data['article_type'], random_state=42)

    # Step 3: Save the datasets
    train_file_path = input("Please enter the path where the training dataset should be saved (default: ./train_data.xlsx): ").strip() or './train_data.xlsx'
    val_file_path = input("Please enter the path where the validation dataset should be saved (default: ./validation_data.xlsx): ").strip() or './validation_data.xlsx'
    test_file_path = input("Please enter the path where the test dataset should be saved (default: ./test_data.xlsx): ").strip() or './test_data.xlsx'

    train_data.to_excel(train_file_path, index=False)
    val_data.to_excel(val_file_path, index=False)
    test_data.to_excel(test_file_path, index=False)

    print(f"Training dataset saved to: {train_file_path}")
    print(f"Validation dataset saved to: {val_file_path}")
    print(f"Test dataset saved to: {test_file_path}")

else:
    # Step 1: Proceed with user inputs for dataset paths if they already have the datasets
    train_file_path = input("Please enter the path to the training file (CSV or Excel): ").strip()
    val_file_path = input("Please enter the path to the validation file (CSV or Excel): ").strip()
    test_file_path = input("Please enter the path to the test file (CSV or Excel): ").strip()

# Step 2: Load the datasets based on their file types
# Load the training dataset
if train_file_path.endswith('.csv'):
    train_data = pd.read_csv(train_file_path)
elif train_file_path.endswith('.xlsx'):
    train_data = pd.read_excel(train_file_path)
else:
    raise ValueError("Unsupported file format. Please provide a CSV or Excel file for the training data.")

# Load the validation dataset
if val_file_path.endswith('.csv'):
    val_data = pd.read_csv(val_file_path)
elif val_file_path.endswith('.xlsx'):
    val_data = pd.read_excel(val_file_path)
else:
    raise ValueError("Unsupported file format for validation dataset. Please provide a CSV or Excel file.")

```
## 2. Fine-tune and Evaluate the Model
Once you have your dataset split, you can use the following script to fine-tune and evaluate the model.

```python


# Step 3: Create the model instance
model = TextClassificationModel()

# Step 4: Load the dataset into the model (this populates self.df)
train_columns = model.load_data(train_file_path)

# Step 5: User selects the label column and text columns
print(f"Available columns: {train_columns}")
label_column = input(f"Please choose the label column from: {train_columns}: ").strip()
text_columns = input(f"Please choose the text columns (comma-separated) from: {train_columns}: ").split(',')

# Step 6: Process the selected columns for training
text_train, y_train = model.set_columns(label_column, [col.strip() for col in text_columns], update_class_mapping=True)

# Repeat the same process for validation and test datasets
val_columns = model.load_data(val_file_path)
text_val, y_val = model.set_columns(label_column, [col.strip() for col in text_columns], update_class_mapping=False)

test_columns = model.load_data(test_file_path)
text_test, y_test = model.set_columns(label_column, [col.strip() for col in text_columns], update_class_mapping=False)


# Step 7: User provides the model save path
save_model_path = input("Please enter the path where the model should be saved (default: ./saved_model): ").strip() or './saved_model'

# Step 8: Train the model and save it (using training and validation datasets)
eval_results = model.train(text_train, y_train, text_val, y_val, save_model_path)
print("Evaluation results after training:", eval_results)

```
## 3. Analyze the Results
You can now analyze the output results based on metrics such as accuracy, precision, recall, and F1-score, which are generated during the evaluation phase.
```python

# Step 9: Get the path from the user to save the predictions
save_predictions_path = input("Please enter the path to save the predictions (default: ./predictions.csv): ").strip() or './predictions.csv'

# Generate predictions and save them
predictions_df = model.predict(text_test, y_test, save_predictions_path, save_model_path)

# predictions_df = model.predict(text_test, save_predictions_path, save_model_path)
print("Predictions saved in the file:", predictions_df)

# Assuming predictions are saved in 'save_predictions_path'
predictions_file_path = save_predictions_path

# Evaluate the predictions from the file and print the metrics
model.evaluate_predictions_from_file(predictions_file_path)


```

