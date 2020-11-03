import pandas as pd
import numpy as np
import csv
import requests
import json
import os.path
import random
import seaborn as sns
import matplotlib.pyplot as plt
import re
import plotly.express as px
import streamlit as st
import datetime

from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

@st.cache(allow_output_mutation=True)
def getData(author, max_df, min_df, ngram_max=4, true_k = 5):

    # get all the publication pubmed IDs for an author from pubmed
    id_url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?'
    database = 'pubmed'

    id_response = requests.post(id_url+'db='+database+'&term='+author+'&retmax=300&retmode=json')
    id_response = id_response.json()['esearchresult']['idlist']

    # create string with all id numbers
    id_num = ','.join(id_response)

    # new API address for pulling pubmed metadata from IDs 
    metadata_url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?'

    # get all pubmed summary data
    metadata_response = requests.post(metadata_url + 'db='+database+'&id='+id_num+'&retmode=json')
    metadata_response = metadata_response.json()['result']

    # new API address for pulling mesh data from IDs 
    mesh_url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?'

    # get mesh keyword data
    mesh_response = requests.post(mesh_url + 'db='+database+'&id='+id_num+"&retmode=xml")
    mesh_response = mesh_response.text.split('<PubmedArticle>')

    # organize pubmed metadata into dataframe
    frames = pd.DataFrame()

    for id_num in id_response:
        if id_num in metadata_response: 
            curr_metadata = metadata_response[id_num]
            curr_metadata_list = [id_num, curr_metadata['sortpubdate'], curr_metadata['title']]
            curr_metadata_df = pd.DataFrame(np.array(curr_metadata_list))
            frames = frames.append(curr_metadata_df.T)

    frames.columns = ['id', 'Date', 'Title']
    frames = frames.set_index('id', drop=True)

    # sort by date
    frames['Date'] = pd.to_datetime(frames['Date'], errors='ignore')
    frames = frames.sort_values(by='Date', ascending=True)

    # create dictionaries for mesh/keyword/doi extractions
    mesh_dict = {}
    all_mesh_dict = {}
    doi_dict = {}

    # create lists for cleaning up mesh keywords
    singular_list = []

    # organize mesh terms into dataframe
    for mesh_data in mesh_response:
        article_id = mesh_data.split('</PMID>')[0].split('<PMID Version="1">')[-1]
        doi_id = 'not available'
        
        if 'ArticleId IdType="doi"' in mesh_data:
            doi_id = mesh_data.split('<ArticleId IdType="doi">')[-1].split('</ArticleId>')[0]
        
        keywords_list = []
        
        if 'MeshHeadingList' in mesh_data:
            keywords_list = mesh_data.split('</MeshHeadingList>')[0].split('<MeshHeadingList>')[-1].split('<MeshHeading>')
            keywords_list = [item.split('</DescriptorName>')[0].split('">')[-1] for item in keywords_list if '</QualifierName>' in item] 
        
        elif 'KeywordList' in mesh_data:
            keywords_list = mesh_data.split('</KeywordList>')[0].split('<KeywordList ')[-1].split('\n')
            keywords_list = [item.split('</Keyword>')[0].split('">')[-1] for item in keywords_list if '</Keyword>' in item]
            keywords_list = [item.title() for item in keywords_list]

        keywords_list = [item.strip() for item in keywords_list]
        keywords_list = [item.replace(' ', '-') for item in keywords_list]
        keywords_list = [item.replace('--', '-') for item in keywords_list]
        keywords_list = [item.replace(', ', ',') for item in keywords_list]
        keywords_list = [item.split(',')[-1]+'-'+item.split(',')[0] if ',' in item else item for item in keywords_list]
                    
        keywords_list = list(set(keywords_list))
        
        for keyword in keywords_list:
            if keyword[-1] != 's': 
                if keyword not in singular_list: singular_list.append(keyword) 

        if 'xml' not in article_id: 
            mesh_dict[article_id] = ' '.join(keywords_list)
            doi_dict[article_id] = doi_id

    mesh_df = pd.DataFrame.from_dict(mesh_dict, orient='index')
    mesh_df['doi'] = pd.DataFrame.from_dict(doi_dict, orient='index')[0]

    # merge metadata and mesh data
    frames = frames.merge(mesh_df, left_index=True, right_index=True, how='outer')
    frames.columns = ['Date', 'Title','mesh', 'doi']
    frames = frames.dropna()

    for ind in frames.index:
        if len(frames['mesh'][ind]) < 3:
            curr_title = frames['Title'][ind].split(' ')
            curr_sortTitle = [word for word in curr_title if word not in text.ENGLISH_STOP_WORDS]
            frames['mesh'][ind] = ' '.join(curr_sortTitle)
            
    frames = frames[~(frames['mesh'] == '')]
    
    common_words = ['Animal', 'Animals','Adult','Children','Mouse','Mice', 'Child', 'Disease', 'Diseases',
                    'Human','Humans','Male','Female','Cell', 'Cells', 'Gene', 'Genes','Protein', 'Proteins', 
                   'Receptor','Receptors', 'DNA','RNA']

    my_stop_words = text.ENGLISH_STOP_WORDS.union(common_words)
    
    vectorizer = TfidfVectorizer(stop_words=my_stop_words,max_df = max_df,min_df=min_df,ngram_range=(1,ngram_max), 
                                 lowercase=False,token_pattern="(?u)(\\b[\\w-]+\\b)") 
    
    X = vectorizer.fit_transform(frames['mesh'].to_list())

    model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=5, algorithm='full')
    model.fit(X)
    
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    feature_dict = {}
    
    for i in range(true_k):
        for ind in order_centroids[i, :1]: 
            feature_dict[i] = terms[ind]
    
    frames['label'] = [model.predict(vectorizer.transform([item])) for item in frames['mesh'] ]
    frames['label'] = [feature_dict[item[0]] for item in frames['label'] ]
    print(min_df, max_df, model.score(X), feature_dict.values())
 
    return frames
    
def plotTimeline(author, author_df, selected_year):

    author_df = author_df[author_df['year'] > selected_year]

    #plot
    colors = ["#FF6787", "#FFB68C", "#FACBC1", "#A6DBD7","#63CECE"]

    fig = px.scatter(author_df, y="label", x="Date", hover_name="Title", color='label',title=author,
     color_discrete_sequence=colors, labels = { 'Date': 'Publication Date', 'doi': 'doi'},
     width=900, height=500)

    fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)','paper_bgcolor': 'rgba(0, 0, 0, 0)',})
    fig.update_traces(marker_size=10)

    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True, tickfont={'size':16}, title_font={'size':16})
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True, showgrid=True,  gridwidth=1, gridcolor='LightGrey',tickfont={'size':16}, title_font={'size':1})

    fig.update_layout(showlegend=False)
    
    st.write(fig)

def streamlitPlotting():

    # add in author input
    author = st.sidebar.text_input('Author Name', value='Charles Darwin')

    # get author data
    frames = getData(author, 0.99, 0.01)

    author_df = frames.copy(deep=True)

    # get dates of frames
    author_df['year'] = pd.DatetimeIndex(author_df['Date']).year
    years = list(author_df['year'].unique())

    #add streamlit widget for date
    selected_year = st.sidebar.slider('Show papers after year: ', min_value=int(min(years)), max_value=int(max(years)), value=int(min(years)))

    # graphing timeline
    plotTimeline(author, author_df, selected_year)


streamlitPlotting()


