##SCRIPT MEANT TO BE RUN ON EXTERNAL GPU MACHINES USING AUROMETALSAURUS

#import necessary modules and models
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import string
from tqdm import tqdm

from functions.ner_model_dutch import get_named_entities_counts, get_named_entities, get_named_entities_counts_V2
from functions.get_article_counts import get_article_counts
from functions.sentence_tokenizer import sentence_tokenizer

file_path = "data/court_case_dataset_sampled_not_stratified.csv"
df_exp = pd.read_csv(file_path)

df_exp.set_index("ecli", inplace=True)

case_to_drop = 'ECLI:NL:GHSHE:2017:2650'
df_exp = df_exp.drop(case_to_drop)


def full_entity_counter(text):
    #Instantiate two Counter() objects
    document_counter_articles = Counter()
    document_counter_other = Counter()

    #Transform the text into sentences
    sentences = sentence_tokenizer(text)

    for sent in sentences:
        try:
            sentence_counter_articles = get_article_counts(sent)
            document_counter_articles += sentence_counter_articles

            sentence_counter_other = get_named_entities_counts_V2(sent)
            document_counter_other += sentence_counter_other
        except RuntimeError:
            split_point = len(sent) // 2
            try:
                s1 = sent[:split_point]
                s2 = sent[split_point:]

                sentence_counter_articles_s1 = get_article_counts(s1)
                document_counter_articles += sentence_counter_articles_s1
                sentence_counter_other_s1 = get_named_entities_counts_V2(s1)
                document_counter_other += sentence_counter_other_s1

                sentence_counter_articles_s2 = get_article_counts(s2)
                document_counter_articles += sentence_counter_articles_s2
                sentence_counter_other_s2 = get_named_entities_counts_V2(s2)
                document_counter_other += sentence_counter_other_s2

                print("RuntimeError mitigated")
            except RuntimeError:
                pass

    result_counter = document_counter_articles + document_counter_other

    return pd.Series(dict(result_counter))


tqdm.pandas(desc="Processing texts: ")
count_vector_df = df_exp["full_text"].progress_apply(full_entity_counter)

print("Shape of the BoW-model:  ", count_vector_df.shape)


#Export bag-of-words model
count_vector_df.to_csv("output_data/count_vector.csv", index=True)

