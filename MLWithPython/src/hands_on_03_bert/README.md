* constants / parameters
MODEL_NAME = "distilbert-base-uncased"
DATASET_NAME = "imdb"
* dynamic batch size (kaggle vs my laptop)
BATCH_SIZE = dynamicBatchSize() # 8 in my laptop, 32 in kaggle

* enable some logs to debug properly, change wandb to offline mode for now

* load dataset
* drop unnecessary columns
* check gpu availability
* load tokenizer
* preprocess / map data to tokenized_data
* drop unnecessary table from tokenized dataset
* convert data to pytorch format
* create data_collector object
* load the bert model,
* create custom metrics function
* init training args
* init trainer object
* train, eval
* predictions

