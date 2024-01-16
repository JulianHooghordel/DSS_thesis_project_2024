from nltk.tokenize import word_tokenize, sent_tokenize

def sentence_tokenizer(text):
    sentences = sent_tokenize(text)

    short_sentences = []
    
    for sent in sentences:
        words = word_tokenize(sent)
        if len(words) < 300:
            short_sentences.append(sent)
        elif len(words) < 600:
            sent1 = " ".join(words[:300])
            sent2 = " ".join(words[300:])
            short_sentences.append(sent1)
            short_sentences.append(sent2)
        else:
            pass    #ignore sentences longer than 600 words due to too much noise

    return short_sentences