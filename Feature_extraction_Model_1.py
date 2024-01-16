# FEATURE EXTRACTION MODEL 1

# CREATES A SIMPLE BOW USING STANDARD PREPROCESSING
# THIS BOW CAN EASILY BE TRANSFORMED INTO A TF-IDF VECTOR USING LIBRARIES LIKE SCIKIT-LEARN


# Perform necessary imports
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import DutchStemmer
import string
from collections import Counter
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


def traditional_pre_processing(remaining_text):
    # Remove punctuation and numbers
    translator = str.maketrans('', '', string.punctuation + '0123456789')
    text_no_punct = remaining_text.translate(translator)

    # Tokenize the text
    tokens = word_tokenize(text_no_punct, language='dutch')

    # Remove stopwords
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

    # Stemming
    stemmer = DutchStemmer()
    stemmed_tokens = [stemmer.stem(word).lower() for word in filtered_tokens]

    return stemmed_tokens


# All sub functions combined into a single function
def full_pre_processing(text):
    """Creates a traditional bag of words for a single text"""

    #apparently, sometimes "\t" symbols are found. We remove these
    #we also want to replace the "/" symbol specifically for the "en/of" tokens
    text_new = text.replace("\t", "").replace("/", " ")

    ### Perform the actual pre-processing of the remaining text###
    processed_tokens = traditional_pre_processing(text_new)


    ### CREATE A BAG OF WORDS FROM THE PRE-PROCESSED TEXT
    # Create a bag of words using Counter
    bag_of_words = Counter(processed_tokens)

    # Convert the bag of words to a list of tuples
    processed_text_tuples = list(bag_of_words.items())

    #finally, remove all values that have a lenght of 2 or less, and tokens with a count of 1
    final_bag_of_words = [(string, count) for string, count in processed_text_tuples if len(string) > 2 and count > 1]

    return pd.Series(dict(final_bag_of_words))


tqdm.pandas(desc="Processing texts: ")
count_vector_df = df_exp["full_text"].progress_apply(full_pre_processing)

print("Shape of the BoW-model:  ", count_vector_df.shape)


#Export bag-of-words model
count_vector_df.to_csv("output_data/tfidf_count_vector.csv", index=True)