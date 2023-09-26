import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
from plotly import graph_objs as go
import matplotlib.pyplot as plt

#Apps
st.set_page_config(page_title="App Análise de Índice de Suicídios", page_icon= ":bar_chart:")
st.title("🎗Dashboard Analytics Suicide📊")

# Style
with open('style.css')as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)


st.image("img/setembro_amarelo.png", width=280)

st.markdown("""<h6 style='text-align: justify;'> Essa aplicação demonstra uma análise exploratória dos dados de Suicidios ocorridos pelo mundo. Todos os anos, mais de 12 mil pessoas tiram suas próprias vidas no Brasil. 
Em um cenário mundial, esse número ultrapassa 1 milhão de pessoas, levando o suicídio a ser considerado um problema de saúde pública. Para você ter noção da dimensão desses números, saiba que [o suicídio tem uma taxa maior do que vítimas de AIDS e da maioria dos tipos de câncer](https://pt.wikipedia.org/wiki/Setembro_Amarelo). </h6""", unsafe_allow_html=True) 

st.markdown("""<h6 style='text-align: justify;'> Segundo a Organização Mundial da Saúde (OMS), o Brasil ocupa o oitavo lugar no número de suicídios no mundo: São 32 brasileiros por dia.Animações como a mostrada acima serão criadas e interpretadas. 
**Setembro Amarelo** é uma iniciativa da Associação Brasileira de Psiquiatria (ABP), em parceria com o Conselho Federal de Medicina (CFM), para divulgar e alertar a população sobre o problema. Oficialmente, o Dia Mundial de Prevenção ao Suicídio ocorre no dia 10 de setembro, porém durante o mês inteiro são promovidos debates, campanhas e ações para a conscientização sobre o suicídio.
O objetivo dessa aplicação é realizar consultas no DataFrame mundial, mostrar índices dos país com maiores casos, gráficos estatisticos por sexo, faixa etária entre outros.</h6""", unsafe_allow_html=True)

st.markdown("""<h6 style='text-align: justify;'> Cerca de 800 mil pessoas sucumbem ao agravamento da saúde mental e escolhem o suicídio como resgate. Ironicamente, apesar de números tão alarmantes, a maioria dos países ainda não tem uma estratégia nacional para prevenir suicídios. Esta é uma catástrofe totalmente nova em si mesma, pois os fatos afirmam que "para cada uma pessoa que morre devido ao suicídio, 000 ou mais pessoas estão tentando".</h6""", unsafe_allow_html=True)
st.markdown("""<h6 style='text-align: justify;'> Dada a situação desastrosa existente, os maiores órgãos administrativos de saúde do mundo estão recomendando que a prevenção do suicídio deve ser alcançada pela consideração sistemática de "fatores de risco e proteção e intervenções relacionadas". Em um mundo onde a tecnologia espia sua face em cada segundo aspecto das vidas humanas, seria uma pena se a mesma tecnologia não oferecesse uma maneira de trazer uma visão indicando potenciais vítimas de suicídio.</h6""", unsafe_allow_html=True)
st.markdown("""<h6 style='text-align: justify;'> Assim, tendo em mente os fatos acima, este blog usa algoritmos de Machine Learning para prever as taxas de suicídio ao analisar e encontrar sinais correlacionados ao aumento das taxas de suicídio entre diferentes coortes globalmente, em todo o espectro socioeconômico. O conjunto de dados usado é fornecido pela https://www.kaggle.com/ e https://ourworldindata.org/suicide. </h6""", unsafe_allow_html=True)
st.markdown("""<h6 style='text-align: justify;'> Fonte: https://sigmoidal.ai/setembro-amarelo-analise-do-suicidio-no-brasil-com-data-science/ </h6""", unsafe_allow_html=True)


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