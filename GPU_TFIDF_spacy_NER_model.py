# Perform necessary imports
import pandas as pd
import spacy
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import DutchStemmer
import string
from collections import Counter
import re
from tqdm import tqdm

# Load stopwords
file_path_1 = "data/dutch_stopwords.txt"
with open(file_path_1, 'r') as file:
    stop_words = file.readlines()

stop_words = [word.strip() for word in stop_words]


#read the actual dataset
file_path_2 = "data/court_case_dataset_sampled_not_stratified.csv"
df_exp = pd.read_csv(file_path_2)

df_exp.set_index("ecli", inplace=True)

case_to_drop = 'ECLI:NL:GHSHE:2017:2650'
df_exp = df_exp.drop(case_to_drop)


#import spacy models for NER and pre-processing
nlp_medium = spacy.load("nl_core_news_md")


#define the named entity extraction models
def get_articles(text):
    article_pattern = re.compile(r"\bartikel\b \d\S*", re.IGNORECASE)
    legal_articles = article_pattern.findall(text)
    legal_articles_adj = [i.rstrip(".,") for i in legal_articles]

    article_dict = {}

    for article in set(legal_articles_adj):
        current_count = article_dict.get(article.lower(), 0)
        count = text.count(article)
        article_dict[article.lower()] = current_count + count

    return list(article_dict.items())


def get_judicial_institutes(text):
    judicial_institutes = df_exp["creator"].unique()

    institute_dict = {}

    for institute in judicial_institutes:
        institute_lower = institute.lower()
        if institute_lower in text.lower():
            count = text.lower().count(institute_lower)
            institute_dict[institute] = count
    
    return list(institute_dict.items())


def get_ecli_nums(text):
    ecli_pattern = re.compile(r'ECLI:[^:]+:[^:]+:[^:]+:\w+', re.IGNORECASE)
    ecli_numbers = ecli_pattern.findall(text)

    ecli_dict = {}

    for ecli in set(ecli_numbers):
        current_count = ecli_dict.get(ecli.lower(), 0)
        count = text.lower().count(ecli.lower())
        ecli_dict[ecli.lower()] = current_count + count
    
    return list(ecli_dict.items())


def named_entity_extraction(text):
    doc = nlp_medium(text)
    named_entities = [(ent.text, ent.label_) for ent in doc.ents]

    #remove certain unwanted entities from the list
    unwanted_entities = ["CARDINAL", "MONEY", "TIME", "WORK_OF_ART", "PERSON", "QUANTITY", "PERCENT", "LANGUAGE"]
    named_entities_adj = [entity for entity in named_entities if entity[1] not in unwanted_entities]

    #extract only years from dates
    named_entities_adj_years_EUR = []

    #additional filtering
    for entity in named_entities_adj:
        #we are only interested in the years, not the specific dates
        if entity[1] == "DATE":
            last_item = entity[0].split()[-1]
            if len(last_item) == 4 and last_item not in ["jaar", "uren", "jaren"]:
                named_entities_adj_years_EUR.append((last_item, "DATE"))
        #remove all entities with EUR currency tokens
        elif entity[0].__contains__("€"):
            pass
        elif len(entity[0]) == 1:
            pass
        else:
            named_entities_adj_years_EUR.append(entity)
    
    return named_entities_adj_years_EUR


def NER_counter(named_entities):
    counts = Counter(item[0] for item in named_entities)
    result = [(key, counts[key]) for key in counts]
    return result


#function ro remove extracted entities from the texts
def remove_entities_from_text(text_to_process, entities):
    for ent in entities:
        ent_to_remove = ent[0].lower()
        text_to_process = text_to_process.lower().replace(ent_to_remove, "")

    return text_to_process


#preprocessing functions
def additional_pre_processing(remaining_text):
    ### Perform the actual pre-processing of the remaining text###
    remaining_text = remaining_text.replace("/", " ")   #specifically for the combination "en/of"

    # Remove punctuation and numbers
    translator = str.maketrans('', '', string.punctuation + '0123456789')
    text_no_punct = remaining_text.translate(translator)

    # Tokenize the text
    tokens = word_tokenize(text_no_punct, language='dutch')

    # Remove stopwords
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

    # Stemming
    stemmer = DutchStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]

    return stemmed_tokens

def final_standardization(complete_entity_list):
    ### Final standardization of the complete entity list
    standardized_counts = {}

    for entity, count in complete_entity_list:
        if entity.__contains__("rtikel"):
            standardized_counts[entity] = standardized_counts.get(entity, 0) + count
        else:
            standardized_entity = "".join(char.lower() for char in entity if char.isalnum() or char.isspace())
        
            # Update the dictionary with the standardized string and its count
            standardized_counts[standardized_entity] = standardized_counts.get(standardized_entity, 0) + count

    # Convert the dictionary back to a list of tuples
    result_list = [(string_val, count) for string_val, count in standardized_counts.items()]

    return result_list


# All sub functions combined into a single function
def full_pre_processing(text):
    """Creates a NER-based bag of words for a single text"""

    #apparently, sometimes "\t" symbols are found. We remove these
    text_new = text.replace("\t", "").replace("ˈ", "")

    ### Extract entitites, and prepare text for traditional pre-processing on remaining text ###
    articles = get_articles(text_new)
    named_entities = named_entity_extraction(text_new)
    ecli_numbers = get_ecli_nums(text_new)
    judicial_institutes = get_judicial_institutes(text_new)
    named_entities_counted = NER_counter(named_entities)

    combined_entities = articles + named_entities_counted + ecli_numbers + judicial_institutes

    remaining_text = remove_entities_from_text(text_new, combined_entities)


    ### Perform the actual pre-processing of the remaining text###
    processed_tokens = additional_pre_processing(remaining_text)


    ### CREATE A BAG OF WORDS FROM THE PRE-PROCESSED TEXT
    # Create a bag of words using Counter
    bag_of_words = Counter(processed_tokens)

    # Convert the bag of words to a list of tuples
    processed_text_tuples = list(bag_of_words.items())

    final_bag_of_words_1 = combined_entities + processed_text_tuples

    #finally, remove all values that have a lenght of one (that is, individual characters)
    final_bag_of_words_2 = [(string, count) for string, count in final_bag_of_words_1 if len(string) > 2 and count > 1]

    ### Perform final standardization on the total bag of words
    final_bag_of_words_3 = final_standardization(final_bag_of_words_2)

    return pd.Series(dict(final_bag_of_words_3))



tqdm.pandas(desc="Processing texts: ")
count_vector_df = df_exp["full_text"].progress_apply(full_pre_processing)

print("Shape of the BoW-model:  ", count_vector_df.shape)


#Export bag-of-words model
count_vector_df.to_csv("output_data/tfidf_spacy_NER_count_vector.csv", index=True)



