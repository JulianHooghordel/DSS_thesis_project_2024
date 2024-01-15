# Perform necessary imports
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import string
import torch

# Load the model directly
from transformers import AutoTokenizer, AutoModelForTokenClassification

NER_model_token = "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"

tokenizer = AutoTokenizer.from_pretrained("romjansen/robbert-base-v2-NER-NL-legislation-refs")
model = AutoModelForTokenClassification.from_pretrained("romjansen/robbert-base-v2-NER-NL-legislation-refs", token = NER_model_token)


def get_article_counts(text):
    #Tokenize the input text
    tokens = tokenizer(text, return_tensors="pt")
    token_list = tokens['input_ids'].tolist()[0]

    #Get the predicted classes (i.e., the entities)
    with torch.no_grad():
        logits = model(**tokens).logits

    predicted_token_class_ids = logits.argmax(-1)

    predicted_tokens_classes = [model.config.id2label[t.item()] for t in predicted_token_class_ids[0]]

    #Put the entities, and corresponding tokens in a list
    #The list contains (entity, token) tuples
    #Extract the entities with "REF" in the name (these correspond to legal references)
    entity_token_list = list(zip(predicted_tokens_classes, token_list))

    token_list_output = []
    for (entity, token) in entity_token_list:
        if entity.__contains__("REF"):
            token_list_output.append(token)

    #Decode the tokens corresponding to the legal reference entities
    decoded_text = tokenizer.decode(token_list_output)

    #Perfrom some preprocessing of the entities
    stop_words = stopwords.words("dutch")

    ouput_list = []
    for word in word_tokenize(decoded_text, language="Dutch"):
        if word not in stop_words and word not in string.punctuation:
            ouput_list.append(word)
    
    standardized_output_list = [entity.lower().strip() for entity in ouput_list]
       
    #Finally, return a dicationary with entities and their counts
    return Counter(standardized_output_list)