import datasets
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import torch.cuda
import transformers
import wandb
from datasets import DatasetDict
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, DataCollatorWithPadding, \
    TrainingArguments, Trainer
import evaluate

def dynamicBatchSize():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0).lower()
        vramGiB = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # Convert to GB

        if "a100" in gpu_name:   # A100 (40GB+ VRAM)
            batch_size = 128
        elif "v100" in gpu_name:  # V100 (16GB/32GB VRAM)
            batch_size = 64 if vramGiB >= 32 else 32
        elif "p100" in gpu_name:  # P100 (16GB VRAM)
            batch_size = 32
        elif "t4" in gpu_name:    # Tesla T4 (16GB VRAM, common in Colab/Kaggle)
            batch_size = 32  # Maybe try 64 if no OOM
        elif "rtx 3090" in gpu_name or vramGiB >= 24:  # RTX 3090 (24GB VRAM)
            batch_size = 64
        elif vramGiB >= 16:   # Any other 16GB+ VRAM GPUs
            batch_size = 32
        elif vramGiB >= 8:    # 8GB VRAM GPUs (e.g., RTX 2080, 3060, etc.)
            batch_size = 16
        elif vramGiB >= 6:    # 6GB VRAM GPUs (e.g., RTX 2060)
            batch_size = 8
        else:
            batch_size = 4  # Safe fallback for smaller GPUs
    else:
        batch_size = 4  # CPU mode, keep it small

    return batch_size

if __name__ == '__main__':
    # constants / parameters
    MODEL_NAME = "distilbert-base-uncased"
    DATASET_NAME = "imdb"
    # dynamic batch size (kaggle vs my laptop)
    BATCH_SIZE = dynamicBatchSize() # 8 in my laptop, 32 in kaggle

    # enable some logs to debug properly, change wandb to offline mode for now
    transformers.logging.set_verbosity_debug()  # Set to 'INFO' for fewer logs
    wandb.init(mode="offline")  # Logs only locally

    # load imdb data
    imdb_datasets_dict = datasets.load_dataset(name=DATASET_NAME)

    # Drop unnecessary columns to speed up the process
    imdb_datasets_dict = DatasetDict({
        "train": imdb_datasets_dict["train"],
        "test": imdb_datasets_dict["test"]
    })

    # check gpu availability
    isGpuAvailable = torch.cuda.is_available()

    # load tokenizer
    distilBertTokenizer = DistilBertTokenizer.from_pretrained(pretrained_model_name_or_path=MODEL_NAME)

    # preprocess / map data to tokenized_data
    def tokenization_function(entry):
        try:
            value = entry["text"]
            tokenized_value = tokenized_value = distilBertTokenizer(text=value, padding="max_length", truncation=True)
            return tokenized_value
        except Exception as x:
            print(f"Tokenization function error: {x = }")
            return None

    tokenized_dataset_dict = imdb_datasets_dict.map(function=tokenization_function, batched=True)

    # drop unnecessary table from tokenized dataset
    print("creating tokenized_dataset")
    tokenized_dataset_dict = tokenized_dataset_dict.remove_columns(["text"])  # we don't need text column
    tokenized_dataset_dict = tokenized_dataset_dict.rename_column("label", "labels")  # cz huggingface wants y = labels
    tokenized_dataset_dict.set_format("torch") # convert to pytorch objects

    print("creating DataCollatorWithPadding")
    # Question: What is data collector? / what does it do?
    # Data collator for padding batches dynamically
    data_collator = DataCollatorWithPadding(tokenizer=distilBertTokenizer)

    # load the bert model
    bert_model = (DistilBertForSequenceClassification
                  .from_pretrained(pretrained_model_name_or_path=MODEL_NAME, num_labels=2))
    if isGpuAvailable:
        bert_model = bert_model.to("cuda")


    # init the training_args
    print("init training_args")
    training_args = TrainingArguments(
        run_name="exp-bert-2",
        output_dir="./bert-imdb",
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs"
    )
    # create trainer object

    # Load desired metrics
    # Load metrics
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    roc_auc_metric = evaluate.load("roc_auc")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=1)  # Get predicted class

        accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
        f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")["f1"]
        roc_auc = roc_auc_metric.compute(prediction_scores=logits, references=labels)["roc_auc"]

        return {
            "accuracy": accuracy,
            "f1": f1,
            "roc_auc": roc_auc
        }

    print("create trainer")
    trainer = Trainer(
        model=bert_model,
        training_args=training_args,
        train_dataset=tokenized_dataset_dict["train"], # train
        eval_dataset=tokenized_dataset_dict["test"],   # validate
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )



    # trainer.start, trainer.end
    # the training!
    print("trainer.train()!")
    trainer.train() # this will fine tune the dataset for 3 epochs!
    print("trainer.evaluate()!")
    trainer.evaluate() # evaluate

    # test_results = trainer.evaluate(test_dataset) # <-- this is the actual test

    def predict_sentiment(text):
        tokenized_text = DistilBertTokenizer(text, return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            outputs = bert_model(**tokenized_text)

        logits = outputs.logits
        probabilities = F.softmax(logits, dim=1)  # Convert logits to probabilities
        predicted_class = torch.argmax(probabilities, dim=1).item()  # Get class with max probability

        return f"Prediction: {'Positive' if predicted_class == 1 else 'Negative'}, Probabilities: {probabilities.tolist()}"

    print("predict_statement")
    print(predict_sentiment("I really loved this movie! It was fantastic."))
    print(predict_sentiment("This was the worst movie I have ever seen."))
    # make some predictions

    # new topic: explain the bert model, ie why it works / does not work
    pass
