import streamlit as st
import streamlit.components.v1 as components

#Apps
st.set_page_config(page_title="App Análise de Índice de Suicídios", page_icon= ":bar_chart:")
st.title("🎗Dashboard Analytics Suicide📊")

# Style
with open('style.css')as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)

st.markdown("<h2 style='text-align: center; color: red;'>Contatos</h2>", unsafe_allow_html=True)


st.markdown("Para desenvolvimento de novos projetos - Dashboard utilizando Inteligência Articial: Machine Learning")
st.divider()
st.markdown("")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.image("icons/whatsapp.png", caption="28 99918-3961", width=90)

with col2:
    st.image("icons/gmail.png", caption="viniciusmeireles@gmail.com", width=100)

with col3:
    st.image("icons/location.png", caption="Vitória/ES", width=90)    

with col4:
    st.image("icons/linkedin.png",caption= "/pviniciusmeireles", width=90)


#### Logo sidebar######
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.image("img/logo.png", caption="Create by", width=250)