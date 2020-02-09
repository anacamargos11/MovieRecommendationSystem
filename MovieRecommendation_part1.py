# ECE9063: Project
# Content Based Movie Recommendation Engine
# Part 1: data pre-processing
#
# Ana Carolina Camargos Couto
# Dept. of Applied Math. Western University

import numpy as np
import pandas as pd
import pickle
import ast
from numpy import asarray, save
from datetime import datetime
from scipy import linalg
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn import preprocessing
from tempfile import TemporaryFile

print("Loading the data files... \n")
RatingsDataFrame = pd.read_csv('movies-dataset/ratings.csv')
Metadata = pd.read_csv('movies-dataset/movies_metadata.csv')
Nmovies = Metadata['id'].shape[0]
Nusers = RatingsDataFrame['userId'].nunique()
MovieFeatures = pd.DataFrame()

# treat or drop mal-formatted data rows
Metadata['id'][35587] = '22'
Metadata['id'][29503] = '12'
Metadata['id'][19730] = '1'
Metadata['budget'][35587] = '0'
Metadata['budget'][29503] = '0'
Metadata['budget'][19730] = '0'
Metadata['popularity'][35587] = '2.185485'
Metadata['popularity'][29503] = '1.931659'
Metadata['popularity'][19730] = '0.065736'
Metadata['revenue'][35587] = '0'
Metadata['revenue'][29503] = '0'
Metadata['revenue'][19730] = '0'
Metadata['runtime'][35587] = '0'
Metadata['runtime'][29503] = '0'
Metadata['runtime'][19730] = '0'
Metadata['adult'][35587] = 'False'
Metadata['adult'][29503] = 'False'
Metadata['adult'][19730] = 'False'
Metadata['original_language'][35587] = 'en'
Metadata['original_language'][29503] = 'ja'
Metadata['original_language'][19730] = 'en'
Metadata['genres'][35587] = float('nan')
Metadata['genres'][29503] = float('nan')
Metadata['genres'][19730] = float('nan')
Metadata['production_companies'][35587] = "[{'name': 'Odyssey Media', 'id': 17161}, {'name': 'Pulser Productions', 'id': 18012}, {'name': 'Rogue State', 'id': 18013}, {'name': 'The Cartel', 'id': 23822}]"
Metadata['production_companies'][29503] = "[{'name': 'Aniplex', 'id': 2883}, {'name': 'GoHands', 'id': 7759}, {'name': 'BROSTA TV', 'id': 7760}, {'name': 'Mardock Scramble Production Committee', 'id': 7761}, {'name': 'Sentai Filmworks', 'id': 33751}]"
Metadata['production_companies'][19730] = "[{'name': 'Carousel Productions', 'id': 11176}, {'name': 'Vision View Entertainment', 'id': 11602}, {'name': 'Telescene Film Group Productions', 'id': 29812}]"
Metadata['production_countries'][35587] = "[{'iso_3166_1': 'CA', 'name': 'Canada'}]"
Metadata['production_countries'][29503] = "[{'iso_3166_1': 'US', 'name': 'United States of America'}, {'iso_3166_1': 'JP', 'name': 'Japan'}]"
Metadata['production_countries'][19730] = "[{'iso_3166_1': 'CA', 'name': 'Canada'}, {'iso_3166_1': 'LU', 'name': 'Luxembourg'}, {'iso_3166_1': 'GB', 'name': 'United Kingdom'}, {'iso_3166_1': 'US', 'name': 'United States of America'}]"
Metadata['spoken_languages'][35587] = "[{'iso_639_1': 'en', 'name': 'English'}]"
Metadata['spoken_languages'][29503] = "[{'iso_639_1': 'ja', 'name': 'æ—¥æœ¬èªž'}]"
Metadata['spoken_languages'][19730] = "[{'iso_639_1': 'en', 'name': 'English'}]"
Metadata['release_date'][35587] = '2014-01-01'
Metadata['release_date'][29503] = '2012-09-29'
Metadata['release_date'][19730] = '1997-08-20'
Metadata['original_title'][35587] = 'Avalanche Sharks'
Metadata['original_title'][29503] = 'Mardock Scramble: The Third Exhaust'
Metadata['original_title'][19730] = 'Midnight Man'
Metadata['overview'][35587] = ' Avalanche Sharks tells the story of a bikini contest that turns into a horrifying affair when it is hit by a shark avalanche.'
Metadata['overview'][29503] = ' Rune Balot goes to a casino connected to the October corporation to try to wrap up her case once and for all.'
Metadata['overview'][19730] = ' - Written by Ã˜rnÃ¥s'
Metadata['tagline'][35587] = 'Beware Of Frost Bites'
Metadata['tagline'][29503] = float('nan')
Metadata['tagline'][19730] = float('nan')

print("Copying the numerical attributes... \n")
# keep the numerical columns the same as they are
MovieFeatures[0] = Metadata['id']
MovieFeatures[1] = Metadata['budget']
MovieFeatures[2] = Metadata['popularity']
MovieFeatures[3] = Metadata['revenue']
MovieFeatures[4] = Metadata['runtime']
MovieFeatures[5] = Metadata['vote_average']
MovieFeatures[6] = Metadata['vote_count']

# extracting the year from the release date feature
release_date = np.array(Metadata['release_date'])
for i in range(0,Nmovies):
    if (type(release_date[i])==str and \
        len(release_date[i]) == 10):
        year = datetime.strptime(release_date[i], '%Y-%m-%d')
        year = year.year
        release_date[i] = year
    else:
        release_date[i] = float('nan')

for i in range(1,31): 
    if (type(release_date[-i])==str and \
        len(release_date[-i]) == 10): 
        year = datetime.strptime(release_date[-i], '%Y-%m-%d') 
        year = year.year 
        release_date[-i] = year 
    else: 
        release_date[-i] = float('nan') 

MovieFeatures[7] = release_date

print("Vectorizing the other attributes... \n")
# one-hot encoding binary features
le = LabelEncoder()
lb = LabelBinarizer()
original_language = le.fit_transform(Metadata['original_language'].fillna('0'))
original_language = lb.fit_transform(original_language)
for i in range(8,8+original_language.shape[1]):
    MovieFeatures[i] = original_language[:,i-8]

# this function extracts words out of dictionaries and formats them
def word_extractor(row_of_words):
    words_joined = []
    if (type(row_of_words)!=str or type(ast.literal_eval(row_of_words))!=list):
        words_joined = ['']
    else:
    # extract words from the dictionaries
        word_list = ast.literal_eval(row_of_words)
        for w in range(0,len(word_list)):
            word_list[w] = word_list[w]['name']
            word_list[w] = word_list[w].replace(" ","")
        words_joined.append(' '.join(word_list))    
    return words_joined

# applying the word_extractor function to dict. features
genres = []
for m in range(0,Nmovies):
    genres.append(word_extractor(Metadata['genres'][m]))

genres = np.array(genres)
genres = genres.ravel()

# finally, vectorizing the features.
count_vectorizer = CountVectorizer()
tfid = TfidfVectorizer(stop_words={'english','french','spanish','german'},\
                  max_features=200)

genres = count_vectorizer.fit_transform(genres)
original_title = tfid.fit_transform(Metadata['original_title'])
overview = tfid.fit_transform(Metadata['overview'].values.astype('U'))
tagline = tfid.fit_transform(Metadata['tagline'].values.astype('U'))

# this function records the processed data in the MovieFeatures DataFrame
def record_new_data(new_data):
  size = MovieFeatures.shape[1]
  for i in range(size,size+new_data.shape[1]):
    MovieFeatures[i] = new_data.toarray()[:,i-size]

record_new_data(genres)
record_new_data(original_title)
record_new_data(overview)
record_new_data(tagline)

print("Getting rid of NaN values... \n")
Metadata_mean = Metadata.mean(skipna=True,numeric_only=True)
MovieFeatures[3] = MovieFeatures[3].fillna(Metadata_mean['revenue'])
MovieFeatures[4] = MovieFeatures[4].fillna(Metadata_mean['runtime'])
MovieFeatures[5] = MovieFeatures[5].fillna(Metadata_mean['vote_average'])
MovieFeatures[6] = MovieFeatures[6].fillna(Metadata_mean['vote_count'])
MovieFeatures = MovieFeatures.fillna('0')

print("Running PCA on Movie Features... \n")
features = np.array(MovieFeatures)
scaler = MinMaxScaler(feature_range=[0, 1])
features[:,3:7] = scaler.fit_transform(features[:, 3:7])
ncomp = 2
pca = PCA(n_components=ncomp)
pca_features = pca.fit_transform(features[:,1:-1]) 
PCAfeatures = np.zeros((pca_features.shape[0],pca_features.shape[1]+1))
PCAfeatures[:,1:pca_features.shape[1]+1] = pca_features
PCAfeatures[:,0] = MovieFeatures[0]
PCA_df = pd.DataFrame(PCAfeatures)
pca_variance = pca.explained_variance_ratio_.sum()

print("Saving the clean data frames and arrays... \n")
Metadata.to_pickle("./Metadata.pkl")
RatingsDataFrame.to_pickle("./RatingsDataFrame.pkl")
PCA_df.to_pickle("./PCA_df.pkl")
save('pca_features.npy',pca_features)
genres_save = genres.toarray()
save('genressave.npy',genres_save)
