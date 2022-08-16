import streamlit as st
from utils import cleanup_sentences, summarize, nlkt_word_tokenize, read_article
from gensim.models import Word2Vec

st.title("Sumarização de notícias")

st.write(" Esta aplicação apresentará as sentenças mais importantes do texto presente na URL inserida.")
valor_url = ""
url = st.text_input("Insira a url", valor_url)

words = []
if len(url) > 1:
    text = read_article(url)
    clean_sentences = cleanup_sentences(text)
    for sent in clean_sentences:
        words.append(nlkt_word_tokenize(sent))
    model = Word2Vec(words, min_count=1, sg=1)
    result = summarize(text, model)
    st.subheader("Resumo:")
    st.write(result)

st.sidebar.header("About")
if st.sidebar.checkbox('Mostrar texto completo:'):
    st.subheader("Texto completo:")
    st.write(text)
st.sidebar.subheader("Essa aplicação foi desenvolvida por Alexandre Vaz")
