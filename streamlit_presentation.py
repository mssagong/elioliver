from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

import requests
from bs4 import BeautifulSoup

import streamlit as st

st.title("ConTinder")
st.header("Tinder, but for your taste in *any* contents in the world")
st.caption("created by Minseung Sagong as a part of personal project portfolio")
st.caption("No need to let algorithm learn you; no need to watch or listen anything beforehand; introducing a simple indicator if your pick will worth your time!")
st.divider()

query1 = st.text_input("Which film/tv series/artist describes you the most?", placeholder="e.g. Call Me By Your Name (film)")
url = "https://en.wikipedia.org/wiki/" + query1.replace(' ','_')

headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36'}
web = requests.get(url, headers=headers).content
source = BeautifulSoup(web, 'html.parser')
text1 = source.find('div', {'class':'mw-content-ltr mw-parser-output'}).get_text()
text1 = text1.replace('\n','')
text1 = text1.replace('\r','')
text1 = text1.replace('\'','')
text1 = text1.replace('^','')
text1 = text1.strip()

query2 = st.text_input("See if it matches", placeholder="e.g. La La Land")
url = "https://en.wikipedia.org/wiki/" + query2.replace(' ','_')

headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36'}
web = requests.get(url, headers=headers).content
source = BeautifulSoup(web, 'html.parser')
text2 = source.find('div', {'class':'mw-content-ltr mw-parser-output'}).get_text()
text2 = text2.replace('\n','')
text2 = text2.replace('\r','')
text2 = text2.replace('\'','')
text2 = text2.replace('^','')
text2 = text2.strip()

query3 = st.text_input("and", placeholder="e.g. NewJeans")
url = "https://en.wikipedia.org/wiki/" + query3.replace(' ','_')

headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36'}
web = requests.get(url, headers=headers).content
source = BeautifulSoup(web, 'html.parser')
text3 = source.find('div', {'class':'mw-content-ltr mw-parser-output'}).get_text()
text3 = text3.replace('\n','')
text3 = text3.replace('\r','')
text3 = text3.replace('\'','')
text3 = text3.replace('^','')
text3 = text3.strip()

corpus = [text1, text2, text3]
vectorizer = TfidfVectorizer()
result = vectorizer.fit_transform(corpus).todense()

st.divider()

df = pd.DataFrame(cosine_similarity(np.asarray(result), np.asarray(result)))
df.columns = [query1, query2, query3]
df.index = [query1, query2, query3]
# st.write(df)
st.metric("Similarity with " + query2, str(round(df.loc[query1, query2] * 100, 1)) + "%", str(round((df.loc[query1, query2]-df.loc[query1, query3]) * 100, 1)) + "%")
st.metric("Similarity with " + query3, str(round(df.loc[query1, query3] * 100, 1)) + "%", str(round((df.loc[query1, query3]-df.loc[query1, query2]) * 100, 1)) + "%")

if df.loc[query1, query2] >= 0.6 and df.loc[query1, query3] >= 0.6:
  st.write("We recommend both " + query2 + " & " + query3 + "!")
elif df.loc[query1, query2] >= 0.6 and df.loc[query1, query3] < 0.6:
  st.write("Highly recommend " + query2 + "!")
elif df.loc[query1, query2] < 0.6 and df.loc[query1, query3] >= 0.6:
  st.write("Highly recommend " + query3 + "!")
else:
  st.write("You know, these are not the only ones the entertainment industry has prepared for you. Try another one.")


