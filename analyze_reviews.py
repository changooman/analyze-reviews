from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy as np
import torch
from transformers import pipeline, set_seed
import logging
set_seed(27)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def retrieve_and_load_dataset(dataset_name: str):
    """
    Load dataset from Hugging Face Datasets
    :param dataset_name: Dataset to be loaded
    :return: DatasetDict
    """
    try:
        dataset = load_dataset(dataset_name)
        logging.info(f"Dataset {dataset_name} loaded.")
        return dataset
    except Exception as e:
        logging.warning(e)
        raise


def preprocess_dataset(dataset, model_name: str):
    """
    Tokenize raw text and remove original column 'text'
    :param dataset: Dataset to be preprocessed
    :param model_name: Model whose tokenizer to use
    :return: DatasetDict
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256, )

    preprocessed_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"], )
    return preprocessed_dataset


def build_model(model_name: str):
    """
    Create a model based on model_name
    :param model_name: Name of the model
    :return: Model
    """
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        logging.info(f"Model {model_name} built.")
        return model
    except Exception as e:
        logging.warning(e)
        raise


def build_trainer(processed_dataset, model):
    """
    Create a trainer to train the dataset based on model
    :param processed_dataset: Processed dataset
    :param model: Model to be trained
    :return: Trainer
    """
    # Flag to check if GPU with cuda is available
    cuda_available = torch.cuda.is_available()
    # Train split
    train_dataset = processed_dataset["train"]
    # Test split
    test_dataset = processed_dataset["validation"]
    settings = TrainingArguments("savepts", evaluation_strategy="epoch", save_strategy="epoch", num_train_epochs=5,
                                 per_device_train_batch_size=8, per_device_eval_batch_size=8, fp16=cuda_available,
                                 load_best_model_at_end=True, metric_for_best_model="accuracy")
    logging.info(f"Trainer built.")
    def compute_metrics(p):
        pred = np.argmax(p.predictions, axis=-1)
        return metric.compute(predictions=pred, references=p.label_ids)
    return Trainer(model=model, args=settings, train_dataset=train_dataset, eval_dataset=test_dataset,
                   compute_metrics=compute_metrics)


def build_pipeline(use_model_name: str, model_name_backup: str):
    """
    :param use_model_name: Model to be used
    :param model_name_backup: Default backup model
    :return: Pipeline
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(use_model_name)
        logging.info(f"Tokenizer {use_model_name} found.")
    except Exception:
        logging.warning(f"{use_model_name} was not found. Using default tokenizer {model_name_backup}.")
        tokenizer = AutoTokenizer.from_pretrained(model_name_backup)
        tokenizer.save_pretrained(use_model_name)
    logging.info(f"Pipeline built.")
    return pipeline("sentiment-analysis", model=use_model_name, tokenizer=tokenizer)


if __name__ == '__main__':
    dataset_name = "cornell-movie-review-data/rotten_tomatoes"
    save_model_name = "trained"
    model_name = "distilbert-base-uncased"
    metric = evaluate.load("accuracy")
    raw_dataset = retrieve_and_load_dataset(dataset_name=dataset_name)
    processed_dataset = preprocess_dataset(raw_dataset, model_name)
    model = build_model(model_name)
    trainer = build_trainer(processed_dataset, model)
    trainer.train()
    trainer.save_model(save_model_name)
    test_results = trainer.evaluate(processed_dataset["test"])
    print("Test accuracy:", round(test_results["eval_accuracy"] * 100, 2), "%")
    analyze = build_pipeline(save_model_name, model_name)
    test_cases = ["That movie was horrible!",
                  "I loved that movie."]
    for case in test_cases:
        print(f"{case} --> {analyze(case)[0]}")
