# FEATURE EXTRACTION MODEL 3

# SIMPLE CREATES ROBBERT-2023 BASED EMBEDDINGS OUT OF THE INPUT TEXTS, USING A STRATGIC FORM OF MEAN POOLING
# NO FINE-TUNING IS APPLIED HERE


from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import torch
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("DTAI-KULeuven/robbert-2023-dutch-large")
model = AutoModel.from_pretrained("DTAI-KULeuven/robbert-2023-dutch-large")


#Load the testing data
#load the unstratified data
df_exp = pd.read_csv("data/court_case_dataset_sampled_not_stratified.csv")
print("Shape of unstratified sample: ", df_exp.shape)

#Set the ecli number to be the index column, which is easier for later analysis
df_exp.set_index("ecli", inplace=True)


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def document_encoder(text):
    # Tokenize the document into words using NLTK
    words = word_tokenize(text)

    # Split the text into chunks of 350 words
    words_per_chunk = 350
    chunks = [' '.join(words[i:i+words_per_chunk]) for i in range(0, len(words), words_per_chunk)]

    model_input = tokenizer(chunks, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        model_output = model(**model_input)

    sentence_embeddings = mean_pooling(model_output, model_input["attention_mask"])

    document_embedding = np.mean(sentence_embeddings.detach().numpy(), axis=0)

    return pd.Series(document_embedding)


tqdm.pandas(desc="Creating embedding matrix: ")
embedding_df = df_exp["full_text"].progress_apply(document_encoder)

#Export bag-of-words model
embedding_df.to_csv("output_data/GPU_RobBERT_model_output_V2.csv", index=True)