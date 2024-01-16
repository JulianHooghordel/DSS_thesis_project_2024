# BUILDING OF FEATURE EXTRACTION MODEL 3.2

# note: THIS MODEL DID NOT MAKE THE FINAL THESIS. DUE TO UNSOLVED ISSUES WITH SAVING/LOADING THE FINE-TUNED MODEL, NO CLUSTER IMPROVEMENTS COULD BE MADE WITH RESPECT TO MODEL 3.1 (MODEL 3 IN THESIS)

# THIS SCRIPT CONTAINS THE CODE TO TRAIN THE ROBBERT-2023 MODEL ON A CLASSIFICATION TASK. THE IDEA IS THAT BY TRAINING THE PRE-TRAINED MODEL TO CLASSIFY THE (MAIN) SUBJECT OF THE TEXTS, THE MODEL WOULD ALSO LEARN TO CREATE BETTER EMBEDDINGS FOR THE TEXTS
# THE CLASSIFICATION TASK RESULTED IN A VALIDATION-ACCURACY OF 80%, AND A F1-SCORE OF 0.79, A SIGNIFICANT IMPROVEMENT COMPARED TO EARLIER EXPERIMENTS, WHERE VALIDATION-ACCURACY SCORES LOWER THAN 60% WERE ACHIEVED. 
# TRAINING WAS DONE WITH THE FOLLOWING HYPERPARAMETERS:
#   - num_epochs = 3
#   - batch_size = 16
#   - learning_rate = 5e-5
#  THE TRAINING DATASET CONTAINED AROUND 8000 SAMPLES, AND VALIDATION WAS DONE ON A SET OF 2000 SAMPLES. 

# AGAIN, THIS MODEL DID (UNFORTUNATELY) NOT MAKE THE FINAL THESIS. HOWEVER, WHEN LOOKING AT THE RESULTS FROM TRAINING AND VALIDATION, I'M CONFIDENT IT WOULD HAVE MADE A NICE IMPROVEMENT.


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AdamW, PreTrainedModel, RobertaConfig
from nltk import word_tokenize
import numpy as np
import pandas as pd

seed = 495

# Load the dataset
# Note, the data is expected to contain a "full_text" column, containing the actual texts
path_to_data = ""
df = pd.read_csv(path_to_data)
print("Shape of fine-tuning dataset: ", df.shape)


# Assuming 'text' is the column containing texts and 'subject' is the column containing labels
texts = df['full_text'].values
labels = df['main_subject'].values

# Split the dataset into train and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=seed)


# Create DataLoader for training and validation sets
class MyDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {'text': self.texts[idx], 'label': self.labels[idx]}
    
    def get_labels(self):
        return self.config.id2label


train_dataset = MyDataset(train_texts, train_labels)
val_dataset = MyDataset(val_texts, val_labels)

# Adjust the batch_size according to your resources
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


# Define the model architecture
class MyDocumentClassifier(PreTrainedModel):

    def __init__(self, pretrained_model_name, input_size, num_classes):
        config = RobertaConfig.from_pretrained(pretrained_model_name)
        super(MyDocumentClassifier, self).__init__(config)

        # Load pre-trained RoBERTa model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.roberta = AutoModel.from_pretrained(pretrained_model_name)

        # Linear classifier
        self.linear = nn.Linear(input_size, num_classes)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


    def forward(self, texts):
        #Initiaze a list to stroe individual text outputs
        individual_outputs = []

        for t in texts:
            # Tokenize the document into words using NLTK
            words = word_tokenize(t)

            # Split the text into chunks of 350 words
            words_per_chunk = 350
            chunks = [' '.join(words[i:i+words_per_chunk]) for i in range(0, len(words), words_per_chunk)]

            # Tokenize and encode the chunks
            model_input = self.tokenizer(chunks, return_tensors="pt", padding=True, truncation=True)

            # Forward pass through RoBERTa
            with torch.no_grad():
                model_output = self.roberta(**model_input)

            # Perform mean pooling on the RoBERTa output
            sentence_embeddings = self.mean_pooling(model_output, model_input["attention_mask"])

            # Calculate the document embedding
            document_embedding = np.mean(sentence_embeddings.detach().numpy(), axis=0)

            # Forward pass through the linear classifier
            output = self.linear(torch.from_numpy(document_embedding).float())

            individual_outputs.append(output)

        outputs = torch.stack(individual_outputs)

        return outputs




# Define hyperparameters
pretrained_model_name = "DTAI-KULeuven/robbert-2023-dutch-large"
input_size = 1024  # RobBERT hidden state size
num_classes = 4  # Number of classes for multi-class classification
learning_rate = 3e-5
num_epochs = 4



# Initialize the model, loss function, and optimizer
model = MyDocumentClassifier(pretrained_model_name, input_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)


print("Let's start training!")
# Training loop
for epoch in range(num_epochs):
    model.train()
    train_predictions = []
    train_true_labels = []

    for batch in tqdm(train_loader, desc=f'Training. Epoch {epoch + 1}/{num_epochs}', leave=False):
        texts, labels = batch['text'], batch['label']
        
        optimizer.zero_grad()

        # Forward pass
        outputs = model(texts)
        loss = criterion(outputs, torch.tensor(labels))

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Collect predictions and true labels for accuracy and F1-score
        train_predictions.extend(outputs.argmax(dim=1).cpu().numpy())
        train_true_labels.extend(labels.cpu().numpy())

    # Calculate accuracy and F1-score for training set
    train_accuracy = accuracy_score(train_true_labels, train_predictions)
    train_f1 = f1_score(train_true_labels, train_predictions, average='weighted')

    print(f'Training Accuracy: {train_accuracy:.4f}, Training F1-score: {train_f1:.4f}')


# Validation loop
model.eval()
val_predictions = []
val_true_labels = []

for batch in tqdm(val_loader, desc='Validation', leave=False):
    texts, labels = batch['text'], batch['label']

    # Forward pass
    output = model(texts)

    # Collect predictions and true labels for accuracy and F1-score
    val_predictions.extend(output.argmax(dim=1).cpu().numpy())
    val_true_labels.extend(labels.cpu().numpy())


# Calculate accuracy and F1-score for validation set
val_accuracy = accuracy_score(val_true_labels, val_predictions)
val_f1 = f1_score(val_true_labels, val_predictions, average='weighted')


print("Fine_tuning is finished! Did it work?")
print(f'Validation Accuracy: {val_accuracy:.4f}, Validation F1-score: {val_f1:.4f}')


# Save the fine-tuned model
model.save_pretrained("models/Amateur_legal_RobBERT_2023_V2")
model.tokenizer.save_pretrained("models/Amateur_legal_RobBERT_2023_V2")



