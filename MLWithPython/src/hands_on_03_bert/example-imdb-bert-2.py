import datasets
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import torch.cuda
import transformers
import wandb
from datasets import DatasetDict
import torch.nn.functional as F
from torch import nn
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, DataCollatorWithPadding, \
    TrainingArguments, Trainer, BertModel
import evaluate

MODEL_NAME = "distilbert-base-uncased"


def getCorrectDevice():
    if torch.cuda.is_available():
        return torch.device("cuda")  # For NVIDIA GPUs
    elif torch.backends.mps.is_available():
        return torch.device("mps")  # For Apple Silicon Macs
    else:
        return torch.device("cpu")   # Fallback to CPU

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

def getGpuName():
    gpu_name = torch.cuda.get_device_name(0).lower()
    return gpu_name


class IMDBClassifier(nn.Module):

    def __init__(self, n_classes):
        super(IMDBClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.h0 = torch.randn(1, 16, 100)
        self.c0 = torch.randn(1, 16, 100)
        self.LSTM = nn.LSTM(768, 100, 1, batch_first=True)
        self.conv1 = torch.nn.Conv1d(1, 1, 3, stride=2)
        self.conv2 = torch.nn.Conv1d(1, 1, 3, stride=2)

        self.Linear = nn.Linear(24, 2)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

        output = self.drop(output['pooler_output'])

        output = output.view(output.shape[0], 1, output.shape[1])
        output, (hn, cn) = self.LSTM(output, (self.h0, self.c0))
        output = self.conv1(output)
        output = self.conv2(output)

        flatten = output.view(16, -1)
        dense1 = self.Linear(flatten)
        return dense1

if __name__ == '__main__':
    print(getGpuName())

    # constants / parameters
    DATASET_NAME = "imdb"
    # dynamic batch size (kaggle vs my laptop)
    BATCH_SIZE = dynamicBatchSize()  # kaggle supports batch size = 64 for T4 gpu

    # enable some logs to debug properly, change wandb to offline mode for now
    transformers.logging.set_verbosity_debug()  # Set to 'INFO' for fewer logs
    wandb.init(mode="offline")  # Logs only locally

    # load imdb data
    imdb_datasets_dict = datasets.load_dataset(DATASET_NAME)

    # Drop unnecessary columns to speed up the process
    isMyLaptop = "nvidia geforce rtx 2060" in getGpuName()

    if isMyLaptop:
        # my laptop is not meant to do actual bert training. just some quick runs to makesure my code is ok.
        # else I need to debug the code in kaggle which will be a hassle
        imdb_datasets_dict = DatasetDict({
            "train": imdb_datasets_dict["train"].select(range(25)),
            # Select the first 25 entries from the train dataset
            "test": imdb_datasets_dict["test"].select(range(25))  # Select the first 25 entries from the test dataset
        })
    else:
        imdb_datasets_dict = DatasetDict({
            "train": imdb_datasets_dict["train"],
            "test": imdb_datasets_dict["test"]
        })

    # check gpu availability
    isGpuAvailable = torch.cuda.is_available()

    # load tokenizer
    distilBertTokenizer = DistilBertTokenizer.from_pretrained(pretrained_model_name_or_path=MODEL_NAME)

    distilBertTokenizer.encode_plus("some review",max_length = 100, add_special_tokens= True,
                                                  pad_to_max_length= True,return_attention_mask=True,
                                                  return_token_type_ids=False,return_tensors='pt')

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
    tokenized_dataset_dict.set_format("torch")  # convert to pytorch objects

    print("creating DataCollatorWithPadding")
    # Question: What is data collector? / what does it do?
    # Data collator for padding batches dynamically
    data_collator = DataCollatorWithPadding(tokenizer=distilBertTokenizer)

    # load the bert model
    # tutorial 1
    isCustomModel = True
    if isCustomModel:
        bert_model = (DistilBertForSequenceClassification
                      .from_pretrained(pretrained_model_name_or_path=MODEL_NAME, num_labels=2))
    else:
        # tutorial 2
        bert_model = IMDBClassifier(n_classes=2)

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

        positive_logits = logits[:,
                          1]  # convert 2d array into 1d array like this: logits[0][1], logits[1][1], logits[2][1], ... ..., logits[n][1]
        print("----- debug start ----")
        print(f"{logits = }")  # a 2d array.
        print(f"{labels = }")  # 1d array
        print(f"{predictions = }")  # 1d array
        print(f"{positive_logits = }")
        print("----- debug end ----")

        accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
        f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")["f1"]
        # roc_auc = roc_auc_metric.compute(prediction_scores=logits, references=labels)["roc_auc"] # 2d array vs 1d array matrix dim mismatch
        roc_auc = roc_auc_metric.compute(prediction_scores=positive_logits, references=labels)[
            "roc_auc"]  # using positive_logits repairs the error

        return {
            "accuracy": accuracy,
            "f1": f1,
            "roc_auc": roc_auc
        }


    print("create trainer")
    trainer = Trainer(
        model=bert_model,
        args=training_args,
        train_dataset=tokenized_dataset_dict["train"],  # train
        eval_dataset=tokenized_dataset_dict["test"],  # validate
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # trainer.start, trainer.end
    # the training!
    print("trainer.train()!")
    trainer.train()  # this will fine tune the dataset for 3 epochs!
    print("trainer.evaluate()!")
    trainer.evaluate()  # evaluate


    # test_results = trainer.evaluate(test_dataset) # <-- this is the actual test

    def predict_sentiment(text):
        device = getCorrectDevice()
        tokenized_text = distilBertTokenizer(text, return_tensors="pt", padding=True, truncation=True)
        tokenized_text = {key: value.to(device) for key, value in tokenized_text.items()}

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

