from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords 
import spacy
import matplotlib.pyplot as plt
from collections import Counter

def get_embedding_matrix(keys, embedding_dic):
    result=[]
    for k in keys:
        result.append(embedding_dic[k])
    return np.array(result)

def closest_to_mean(doc_embedding):
    #doc_embedding=list(dic_embedding.values())
    mean_embed=np.array(doc_embedding).mean(axis=0)
    distances=[]
    for x in doc_embedding:
        distances.append(cosine_similarity(mean_embed.reshape(1,-1), x.reshape(1,-1))[0,0])
    return np.argmax(distances)


def preprocess_text(document):
    nlp = spacy.load("en_core_web_sm")
    
    stop_words = set(stopwords.words('english'))
    
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(document))

    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)

    # Converting to Lowercase
    document = nlp(document)
    tokens = [x.lemma_.lower() for x in document]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [word for word in tokens if len(word)  > 1]
    return tokens


def query_dates(dates, start, end):
    start_=pd.to_datetime(start)
    end_=pd.to_datetime(end)
    return (dates>=start_)&(dates<=end_)


def documents_to_entities(series_entities, dict_entities, key_string, k=10):
    result=[]
    series_entities.apply(lambda x: result.extend(x))
    return {dict_entities[key][key_string]:value for key,value in Counter(result).most_common(k)}

def describe_entities(data, entities, peak_data, k=7):
    fig, axs = plt.subplots(1,3,figsize=(26,14))
    width = 0.3
    for i, (entity_name, (entity_map, key_name))in enumerate(entities.items()):
        most_common_data=pd.Series(documents_to_entities(data[entity_name], entity_map, key_name, k=len(entity_map)))
        most_common_peak=pd.Series(documents_to_entities(peak_data[entity_name], entity_map, key_name, k=k))
        (most_common_peak/len(peak_data)*100).plot(kind="bar", ax=axs.flat[i], color="blue", width=width, label="Peak", position=0)
        (most_common_data[most_common_peak.index]/len(data)*100).plot(kind="bar", ax=axs.flat[i], color="red",width=width, label="Entire Dataset", position=1)
        axs.flat[i].legend()
        axs.flat[i].set_xlabel("Frequency (%)")
        axs.flat[i].set_title(entity_name)
    plt.legend()
    plt.show()