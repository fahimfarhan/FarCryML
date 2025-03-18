"""

Coding Task:

    Fine-tune BERT on a small dataset (e.g., IMDb movie reviews for sentiment classification).
    Use transformers and Trainer from Hugging Face.

Hint: Use the datasets library to load IMDb:
"""
import torch
from datasets import load_dataset
from transformers import BertTokenizer, DataCollatorWithPadding, BertForSequenceClassification, TrainingArguments, \
    Trainer

MODEL_NAME = "distilbert-base-uncased" # "bert-base-uncased"

if __name__ == '__main__':
    imdb_dataset = load_dataset("imdb")
    print(f"{imdb_dataset = }")
    # 2 columns, text, and label

    # load tokenizer
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    # we need a tokenizer function. for now follow tutorial. java styled code organization later
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_dataset = imdb_dataset.map(tokenize_function, batched=True)

    print(f"{tokenized_dataset = }")
    """
    tokenized_dataset = DatasetDict({
        train: Dataset({
            features: ['text', 'label', 'input_ids', 'token_type_ids', 'attention_mask'],
            num_rows: 25000
        })
        test: Dataset({
            features: ['text', 'label', 'input_ids', 'token_type_ids', 'attention_mask'],
            num_rows: 25000
        })
        unsupervised: Dataset({
            features: ['text', 'label', 'input_ids', 'token_type_ids', 'attention_mask'],
            num_rows: 50000
        })
    })    
    """

    # convert data to pytorch format
    tokenized_dataset = tokenized_dataset.remove_columns(["text"]) # we don't need text column
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels") # huggingface library wants y = labels
    tokenized_dataset.set_format("torch") # convert matrices into pytorch

    # Data collator for padding batches dynamically
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)  # Question: What is data collector? / what does it do?

    # load the model
    model = (BertForSequenceClassification
             .from_pretrained(MODEL_NAME, num_labels = 2)
             .to("cuda"))  # <-- Ensure Trainer runs on GPU

    # define training arguments, and trainer
    BATCH_SIZE = 8

    train_args = TrainingArguments(
        output_dir="./bert-imdb",
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        # device="cuda" # looks like different version had this param
    )

    # question: what about optimizer, loss function?

    print(torch.cuda.is_available())  # Should print: True
    print(torch.cuda.device_count())  # Should print: 1 (if you have one GPU)
    print(torch.cuda.get_device_name(0))  # Should print: NVIDIA GeForce RTX 2060 Max-Q

    # initialize the trainer
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        # device="cuda" # looks like different version had this param
    )

    # the training!
    trainer.train() # this will fine tune the dataset for 3 epochs!

    trainer.evaluate() # evaluate

    def predict_sentiment(text):
        tokenized_text = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad:
            outputs = model(**tokenized_text)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
        return "Positive" if prediction == 1 else "Negative"


    print(predict_sentiment("I really loved this movie! It was fantastic."))
    print(predict_sentiment("This was the worst movie I have ever seen."))

    pass
