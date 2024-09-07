import nltk
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import re
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import googlemaps
import os
from dotenv import load_dotenv
load_dotenv()
_key = os.getenv('GOOGLE_MAPS_KEY')
gmaps = googlemaps.Client(key=_key)
current_directory = os.getcwd()
current_directory = current_directory.replace('\\','/')
dfpath = f"{current_directory}/jobdf.csv"
print(dfpath)
dummy = pd.read_csv(dfpath)

def gettingProcessedJoblist():
    # dummy.drop(["Ratings","Date"],axis=1,inplace=True)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    dummy['Summary'] = dummy['Summary'].apply(lambda x: re.sub("[,.]"," ", x))
    dummy['Summary'] = dummy['Summary'].apply(lambda x: re.sub("  "," ", x))
    jobdoc = []
    for i in range(len(dummy)):
        desc = dummy.iloc[i]['Summary']
        desc = desc.split(' ')
        filtered = [w for w in desc if not w.lower() in stop_words]
        final = [lemmatizer.lemmatize(w) for w in filtered]
        jobdoc.append(filtered)
    jbdc = []
    for i in jobdoc:
        jbdc.append(' '.join(i))
    with open('joblist.txt', 'w',encoding='utf-8') as f:
        for listitem in jbdc:
            f.write('%s\n' % listitem)


def recc(skills,location):
    jblist = []
    if os.path.isfile("joblist.txt"):
        gettingProcessedJoblist()
    with open('joblist.txt', 'r',encoding='utf-8') as filehandle:
        for line in filehandle:
            current = line[:-1]
            jblist.append(current)
    cv = CountVectorizer()
    wc_vector = cv.fit_transform(jblist)
    tfidf_transformer=TfidfTransformer(norm="l2") 
    tfidf_transformer.fit(wc_vector)
    tfidf_matrix = tfidf_transformer.transform(wc_vector)
    query = [f"{skills}"]
    q_vec = cv.transform(query)
    q1 = tfidf_transformer.transform(q_vec)
    results = cosine_similarity(q1,tfidf_matrix)
    results = results.reshape(-1)
    sorted_res = np.argsort(-results)
    #printing jobs
    # recjblist = [jblist[i] for i in sorted_res[:10]]
    usr_loc = f"{location}"
    distance_scores = []
    for i in sorted_res[:10]:
        comp_dist = gmaps.distance_matrix(usr_loc,str(dummy.iloc[i]['City']))['rows'][0]['elements'][0]['distance']['value']
        distance_score = 1 / (1 + comp_dist)  # The closer the job, the higher the score
        distance_scores.append(distance_score)
    distance_scores = np.array(distance_scores)
    distance_scores = distance_scores / np.max(distance_scores)
    combined_scores = [[i,0.7 * results[i]] for i in sorted_res[:10]]
    for i in range(10):
        combined_scores[i][1]+= 0.3*distance_scores[i]
    sorted(combined_scores,key = lambda l:l[1],reverse=True)
    jb_inds = [i for i,_ in combined_scores]
    # recommen_joblist = [jblist[i] for i,_ in combined_scores]
    # print(recommen_joblist)
    # print(jb_inds)
    return jb_inds
