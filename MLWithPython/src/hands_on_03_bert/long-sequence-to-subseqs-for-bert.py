# Specify the file path
from collections import UserDict

import torch

file_path = 'really-long-text.txt'

# Open the file and read its contents
with open(file_path, 'r') as file:
    file_contents = file.read()

# Now file_contents contains all the text from the file
# print(file_contents)



from transformers import BertForSequenceClassification, BertTokenizer, BatchEncoding

MODEL_NAME = "ProsusAI/finbert"

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME)

txt = file_contents

tokensv1 = tokenizer.encode_plus(
    txt,
    add_special_tokens=True, # adds [cls], [sep], [pad]
    max_length=512,
    truncation=True,
    padding="max_length"
)

print(f"{tokensv1 = }")
# this would have worked for "normal cases" . But we are dealing with abnormal cases.
# so the default encode_plus is simply incompatible with our long seqs.

# need to modify the encode plus

# 1. no default truncate / padding
# cls, sep are expected at the start, and end of seqs. need to manually add these tokens

# The new encode_plus method looks like this:

tokens = tokenizer.encode_plus(
    txt,
    add_special_tokens=False,
    return_attention_mask=True,
    return_tensors="pt"
)

print(f"{ len(tokens['input_ids'][0]) = }")
print(f"{tokens = }")

# Preparing The Chunks
input_ids_chunks = list(tokens["input_ids"][0].split(510))
musks_chunks = list(tokens["attention_mask"][0].split(510))

for tensor in input_ids_chunks:
    print(f"{len(tensor) = }")

# CLS and SEP
# manually add these
# pad --> 0, unk --> 100, cls = 101, sep = 102, mask = 103

# it looks like there's a for loop missing!

# We also need to perform padding manually



for i in range(0, len(input_ids_chunks)):
    input_ids_chunks[i] = torch.cat([
        torch.Tensor([101]),
        input_ids_chunks[i],
        torch.Tensor([102])
    ])

    musks_chunks[i] = torch.cat([
        torch.Tensor([1]),
        musks_chunks[i],
        torch.Tensor([1])
    ])

    pad_length = 512 - input_ids_chunks[i].shape[0]
    print(f"{i}th_{pad_length = }")

    if pad_length > 0:
        input_ids_chunks[i] = torch.cat([
            input_ids_chunks[i],
            torch.Tensor([0] * pad_length)
        ])
        musks_chunks[i] = torch.cat([
            musks_chunks[i],
            torch.Tensor([0] * pad_length)
        ])


# concat / stack the chunks
input_ids = torch.stack(input_ids_chunks)
attention_mask = torch.stack(musks_chunks)

input_dict = {
    "input_ids": input_ids.long(),
    "attention_mask": attention_mask.int()
}


print(f"{input_dict = }")

# making predictions!
output = model(**input_dict)
print(f"{output = }")

def longSequenceEncodePlus(tokenizer: BertTokenizer, sequence: str) -> BatchEncoding:
    max_size = 512

    map: BatchEncoding = tokenizer.encode_plus(
        sequence,
        add_special_tokens=False,
        return_attention_mask=True,
        return_tensors="pt"
    )

    someInputIds1xN = map["input_ids"]  # shape = 1xN , N = sequence length
    someMasks1xN = map["attention_mask"]
    inputIdsList = list(someInputIds1xN[0].split(510))
    masksList = list(someMasks1xN[0].split(510))

    tmpLength: int = len(inputIdsList)

    for i in range(0, tmpLength):
        cls: torch.Tensor = torch.Tensor([101])
        sep: torch.Tensor = torch.Tensor([102])

        isTokenUnitTensor = torch.Tensor([1])

        inputIdsList[i]: torch.Tensor = torch.cat([
            cls,
            inputIdsList[i],
            sep
        ])

        masksList[i] = torch.cat([
            isTokenUnitTensor,
            masksList[i],
            isTokenUnitTensor
        ])


        pad_len: int = 512 - inputIdsList[i].shape[0]
        if pad_len > 0:
            pad: torch.Tensor = torch.Tensor([0] * pad_len)

            inputIdsList[i]: torch.Tensor = torch.cat([
                inputIdsList[i],
                pad
            ])

            masksList[i]: torch.Tensor = torch.cat([
                masksList[i],
                pad
            ])


    # so each item len = 512, and the last one may have some padding
    input_ids: torch.Tensor = torch.stack(inputIdsList)
    attention_mask: torch.Tensor = torch.stack(masksList)

    encoded_map: dict = {
        "input_ids": input_ids.long(),
        "attention_mask": attention_mask.int()
    }

    batchEncodingDict: BatchEncoding = BatchEncoding(encoded_map)
    return batchEncodingDict

testObject: BatchEncoding = longSequenceEncodePlus(tokenizer, txt)
print(f"{testObject = }")

output2 = model(**testObject)
print(f"{output2 = }")



