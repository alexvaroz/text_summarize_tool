import streamlit as st
from utils import cleanup_sentences, summarize, nlkt_word_tokenize, read_article
from gensim.models import Word2Vec

st.title("Minha primeira aplicação")

st.markdown("#### Minha primera aplicação realizará um resumo de um texto inserido.")
url = st.text_input("")

words = []
if len(url) > 1:
    text = read_article(url)
    clean_sentences = cleanup_sentences(text)
    for sent in clean_sentences:
        words.append(nlkt_word_tokenize(sent))
    model = Word2Vec(words, min_count=1, sg=1)
    result = summarize(text, model)
    st.write(result)

st.sidebar.header("About")
st.sidebar.subheader("Essa aplicação foi desenvolvida por Alexandre Vaz")
