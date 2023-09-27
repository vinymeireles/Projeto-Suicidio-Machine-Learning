import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
from plotly import graph_objs as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
import warnings
warnings.filterwarnings("ignore")

#Apps
st.set_page_config(page_title="App Análise de Índice de Suicídios", page_icon= ":bar_chart:")
st.title("🎗Dashboard Analytics Suicide📊")

# Style
with open('style.css')as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)

#load data
@st.cache_data
def load_data():
    data = pd.read_csv("data/suicide_rates.csv")
    return data

df = load_data()

st.subheader("Análise de Índice de Suicídios no Brasil")
st.markdown("""<h6 style='text-align: justify;'>O suicídio é uma ocorrência complexa, influenciada por fatores psicológicos, biológicos, sociais e culturais. Segundo dados da Organização Mundial da Saúde, mais de 700 mil pessoas morrem por ano devido ao suicídio, o que representa uma a cada 100 mortes registradas. Ainda de acordo com a OMS, as taxas mundiais de suicídio estão diminuindo, mas na região das Américas os números vêm crescendo. Entre 2000 e 2019, a taxa global diminuiu 36%. No mesmo período, nas Américas, as taxas aumentaram 17%. Entre os jovens de 15 a 29 anos, o suicídio aparece como a quarta causa de morte mais recorrente, atrás de acidentes no trânsito, tuberculose e violência interpessoal.</h6""", unsafe_allow_html=True)
st.markdown("""<h6 style='text-align: justify;'>Fonte: https://www.gov.br/saude/pt-br/assuntos/noticias/2022/setembro/anualmente-mais-de-700-mil-pessoas-cometem-suicidio-segundo-oms.</h6""", unsafe_allow_html=True)

st.divider()
#Dataset Global exceto o Brasil
df_brasil = df[df.country == "Brazil"].copy()

#Visualizar DataFrame original
if st.sidebar.checkbox("🧾**Mostrar Dados**", False, key=0):
    st.markdown(":red[**Dados no Brasil sobre Suicídios**]")
    st.markdown("Fonte Dataset: https://www.kaggle.com")
    
    with st.expander("🗓 **Visualizar DataFrame Original**"):
        st.dataframe(df_brasil, use_container_width=True)


#### Gráficos da Tendência de taxa de suicídio no Brasil
if st.sidebar.checkbox(" 📊 **Mostrar Gráfico**", False, key=1):

    # obter a media mundial e do Brasil em suicidios
    years = df_brasil.year.unique()    # pegar os anos para o eixo x
    suicides_brasil_mean = df_brasil.groupby('year')['suicides/100k pop'].mean()
    suicides_world_mean = df.groupby('year')['suicides/100k pop'].mean()

    # como o Brasil nao tem 2016, vou eliminar do dataframe mundial essa entrada
    suicides_world_mean.drop(2016, inplace=True)

    # plotar lineplot comparativo entre Brasil e Mundo
    st.write('📈 Gráfico comparativo taxa de suicídio entre Brasil x Mundo ')
    if not st.checkbox('Ocultar gráfico 1', False, key=2):
        fig = plt.figure(figsize=(12,6), tight_layout=True)
        fig.patch.set_facecolor('white')
        ax = sns.lineplot(x=years, y=suicides_brasil_mean, label='Brasil')
        ax = sns.lineplot(x=years, y=suicides_world_mean, label='Mundo')
        ax.set_facecolor('#fff')
        plt.legend(title="Taxa de suicídio")
        plt.title('Gráfico')
        plt.tight_layout()
        st.pyplot(fig)
        st.info('O gráfico acima demonstra é que apesar da taxa de suicídios no Brasil ser menor que a média mundial, ela vem crescendo constantemente ao longo de 30 anos.')
        st.divider()


    # plotar Gráfico Faixa etária com maior índice de suícidio no Brasil
    st.write('📊 Gráfico Faixa etária com maior índice de suícidio no Brasil por ano')
    if not st.checkbox('Ocultar gráfico 2', False, key=3): 
        # criar uma tabela dinâmica
        table = pd.pivot_table(df_brasil, values='suicides_no', index=['year'], columns=['age'])

        # reordenar as tabelas para deixar em ordem crescente
        column_order = ['5-14 years', '15-24 years', '25-34 years', '35-54 years', '55-74 years', '75+ years']
        table = table.reindex(column_order, axis=1)

        #Gráfico table das faixas etárias no Brasil
        st.pyplot(table.plot.bar(stacked=True, figsize=(16,8)).figure)
        st.info('O gráfico acima demonstra o grupo de pessoas que mais cometem suicídio está entre 35-54 anos. Em 2º lugar, estão pessoas entre 25-34 anos de idade.')
        st.divider()

    #Gráfico de taxa de suicídio entre homens e mulheres no Brasil
    st.write('⭕ Gráfico taxa de suicídio entre homens e mulheres no Brasil')
    if not st.checkbox('Ocultar gráfico 3', False, key=4): 
        # extrair valores entre homens e mulheres
        homens_mulheres = df_brasil.groupby('sex').suicides_no.sum() / df_brasil.groupby('sex').suicides_no.sum().sum()

        # plotar o gráfico de pizza
        fig2 = plt.figure(figsize=(8,6))
        explode = [0.0, 0.2]
        plt.pie(homens_mulheres, labels=['mulheres', 'homens'], autopct='%1.1f%%', shadow=False, explode=explode)
        plt.legend(title="sex")
        
        st.pyplot(fig2)
        st.info('O gráfico acima demonstra que analisando-se todo o período, o *dataset* utilizado mostrou que aproximadamente 78% dos casos foram cometidos por homens e 22% deles por mulheres. Optou-se por pegar a média dos 30 anos, pois não houve mudança significativa desse comportamento durante o período.')
        st.divider()


    # TENDÊNCIA DE SUICÍDIO POR GERAÇÃO DA POPULAÇÃO
    st.write('📊 Gráfico tendência de suicídio por geração no Brasil')
    if not st.checkbox('Ocultar gráfico 4', False, key=5): 
        fig, ax = plt.subplots(figsize = (10,5))
        sns.barplot(x = 'sex', y = 'suicides_no', hue = 'generation', data = df_brasil)
        plt.tight_layout()
        st.pyplot(fig)
        st.info('O gráfico acima mostra os números de suicídios para cada geração divididas por sexo. Nos dois casos, tanto entre os homens quanto mulheres, a geração que apresenta maior número de casos de suicídio é a Boomers. Este grupo da população corresponde àquelas pessoas que nasceram logo após a Segunda Guerra Mundial, até a metade de 1960. Enquanto a Geração Z, nascidos entre 2001 e 2010, apresentam o menor número de casos.')
        st.divider()

    # TENDÊNCIA DE SUICÍDIO POR GERAÇÃO DA POPULAÇÃO
    st.write('📊 Gráfico Nº de suicídio por faixa etária no Brasil')
    if not st.checkbox('Ocultar gráfico 5', False, key=6): 
        idade = df_brasil.groupby('age')['suicides_no'].sum().sort_values(ascending = False)
        eixox = idade.index
        fig = plt.figure(figsize = (10,6))
        sns.barplot(x = eixox, y = idade)
        plt.title('Índice de Suicídio')
        plt.xlabel('Faixa Etária')
        plt.ylabel('Número de suicídios')
        plt.tight_layout()
        st.pyplot(fig)
        st.info('O gráfico acima demonstra acima, podemos facilmente identificar que o grupo de pessoas que mais cometem suicídio é composto por pessoas cuja idade varia entre 35 e 54 anos. Em seguida, aparecem as pessoas com idades entre 25 e 34 anos. Juntos, esses dois grupos correspondem a quase 60% de todos os casos de suicídio no país..')
        st.divider()

#### Estatística de taxa de suicídio no Brasil ############################################################
if st.sidebar.checkbox("🧾**Mostrar Estatística**", False, key=7):
    # criar uma tabela dinâmica
    table = pd.pivot_table(df_brasil, values='suicides_no', index=['year'], columns=['age'])

    # reordenar as tabelas para deixar em ordem crescente
    column_order = ['5-14 years', '15-24 years', '25-34 years', '35-54 years', '55-74 years', '75+ years']
    table = table.reindex(column_order, axis=1)

    st.write('📌 Estatística de suicídio por faixa etária no Brasil (%)')
    ano = st.slider('Selecione o ano:', 1987, 2016, 2015)
    estat = round((table.loc[ano] / table.loc[ano].sum()) * 100,2)
    #st.text(f'Faixa etária para o {round(estat*100,2)} %')
    st.table(estat)
    st.info('O maior índice de taxa de suicídio no Brasil está entre 35-54 e depois 25-34 anos.')
    st.divider()

###### Dados Previsões futuras usando Machine Learning: 2000 - 2019 ######################################
if st.sidebar.checkbox(" ⏰ **Previsões futuras**", False, key=8):
    st.markdown("📊 **Previsões de novos casos no Brasil**:")        

    @st.cache_data
    def load_data1():
        data = pd.read_csv("data/suicides_cases.csv")
        return data

    dataset = load_data1()

    
    #Machine Learning 
    df_brasil = dataset[dataset.country == "Brazil"].copy()
    
    anos = df_brasil[['year']]
    casos = df_brasil[['cases']]

    #### PREPARE DATA ####
    X = np.array(anos).reshape(-1, 1)
    y = np.array(casos).reshape(-1, 1)
     
 
    #### TRAINING DATA ####
    model = linear_model.LinearRegression()
    model.fit(X=anos, y=casos)

    #### PREDICTION ####
    ano_futuro = st.slider('Selecione o ano :', 2020, 2030, 2023)
    novos_casos = model.predict(np.array(ano_futuro).reshape(-1, 1))
    novos_casos = int(novos_casos)

    st.markdown('🎯Previsões futuras de novos casos')  
    col1, col2= st.columns(2)
    with col1:
        st.info(f'📆 Ano da Previsão: **{ano_futuro}**.')
    with col2:    
        st.info(f'👨‍👩‍👧‍👧 **{novos_casos}** casos por 100k população no Brasil.')
    st.divider()

    #### Medição do modelo
    accuracy = model.score(anos,casos)
    st.markdown('🎯Accurácia de acerto do modelo de Machine Learning - IA')
    st.info(f'Accuracy: {round(accuracy*100,2)} %')
    st.divider()
    
    st.info('✅ Conclusão: A tendência é que, com o decorrer dos anos, os índices no Brasil venham a aumentar com relação ao restante do mundo de acordo com os índices apresentados pela Inteligência Artificial. O **Setembro Amarelo** é uma campanha de conscientização do governo brasileiro para combater essa doença psíquica.')



      
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
st.sidebar.image("img/logo.png", caption="Create by", width=250)