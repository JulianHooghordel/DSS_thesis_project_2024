# Import necessary modules
from collections import Counter
import torch

# Load model directly
from transformers import AutoTokenizer, AutoModelForTokenClassification

tokenizer = AutoTokenizer.from_pretrained("pdelobelle/robbert-v2-dutch-ner")
model = AutoModelForTokenClassification.from_pretrained("pdelobelle/robbert-v2-dutch-ner")


def get_named_entities(text):
    """Returns a list of all tokenized entities in the text, and their predicted entity-classes. The list contains (token, entity_class) tuples"""
    tokens = tokenizer(text, return_tensors="pt")
    token_list = tokens['input_ids'].tolist()[0]
    decoded_token_list = [tokenizer.decode(token) for token in token_list]

    #Let's create the predictions (the logits)
    with torch.no_grad():
        logits = model(**tokens).logits

    predicted_token_class_ids = logits.argmax(-1)

    predicted_tokens_classes = [model.config.id2label[t.item()] for t in predicted_token_class_ids[0]]

    entity_token_list = list(zip(predicted_tokens_classes, decoded_token_list))

    return entity_token_list


def get_named_entities_counts(text):
    tokens = tokenizer(text, return_tensors="pt")
    token_list = tokens['input_ids'].tolist()[0]
    decoded_token_list = [tokenizer.decode(token) for token in token_list]

    #Let's create the predictions (the logits)
    with torch.no_grad():
        logits = model(**tokens).logits

    predicted_token_class_ids = logits.argmax(-1)

    predicted_tokens_classes = [model.config.id2label[t.item()] for t in predicted_token_class_ids[0]]

    entity_token_list = list(zip(predicted_tokens_classes, decoded_token_list))

    entity_list = [substring for (entity, substring) in entity_token_list if (entity != "O") and ("PER" not in entity)]

    standardized_output_list = [entity.lower().strip() for entity in entity_list]

    return Counter(standardized_output_list)


def get_named_entities_counts_V2(text):
    """"Version 2 of get_named_entities_counts function. This version concatenates the named entities into a single entity, in the hope to reduce noise"""
    
    tokens = tokenizer(text, return_tensors="pt")
    token_list = tokens['input_ids'].tolist()[0]
    decoded_token_list = [tokenizer.decode(token) for token in token_list]

    #Let's create the predictions (the logits)
    with torch.no_grad():
        logits = model(**tokens).logits

    predicted_token_class_ids = logits.argmax(-1)

    predicted_tokens_classes = [model.config.id2label[t.item()] for t in predicted_token_class_ids[0]]

    entity_token_list = list(zip(predicted_tokens_classes, decoded_token_list))

    ##START
    entity_strings = []
    current_entity = ""

    for (entity_token, substring) in entity_token_list:
        if entity_token.startswith("B-") and "PER" not in entity_token:
            # Start a new entity
            if current_entity:
                entity_strings.append(current_entity.lower().strip())
            current_entity = substring
        elif entity_token.startswith("I-") and "PER" not in entity_token:
            # Continue the current entity
            current_entity += substring
        elif entity_token == "O":
            # If "O" is encountered, end the current entity
            if current_entity:
                entity_strings.append(current_entity.lower().strip())
            current_entity = ""
    
    return Counter(entity_strings)

