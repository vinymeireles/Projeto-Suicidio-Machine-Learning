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

#load data #### DataFrame 
@st.cache_data
def load_data():
    data = pd.read_csv("data/suicide_rates.csv")
    return data

df = load_data()


#Layout da página
st.subheader("Análise global de índice de suicídios por país")
st.markdown("""<h6 style='text-align: justify;'> Taxas de suicídio variam ao redor do mundo. As taxas de suicídio variam muito entre os países. Para alguns países da África Austral e da Europa Oriental, as taxas estimadas de suicídio são elevadas, com mais de 15 mortes anuais por 100.000 pessoas.
Enquanto isso, para outros países da Europa, América do Sul e Ásia, as taxas estimadas de suicídio são menores, com menos de 10 mortes anuais por 100.000 pessoas. A ampla variação nas taxas de suicídio em todo o mundo é provavelmente o resultado de muitos fatores. 
Isso inclui diferenças na saúde mental subjacente e tratamento, estresse pessoal e financeiro, restrições aos meios de suicídio, reconhecimento e conscientização do suicídio e outros fatores.</h6""", unsafe_allow_html=True)
st.markdown("""<h6 style='text-align: justify;'>Fonte: https://obeabadosertao.com.br/portal/2023/07/13/taxas-de-suicidio-caem-ao-redor-do-mundo-brasil-e-eua-vao-a-contramao/</h6""", unsafe_allow_html=True)

st.divider()
########################## DATAFRAME ATUALIZADO WORLD 2010-2019 ####################################################################
#load data World Atualizada
@st.cache_data
def load_data():
    data = pd.read_csv("data/suicides_global.csv")
    return data

dfw = load_data()

#Performance data cleaning and grouping
data_w = dfw.loc[(dfw["country"] != 'World') & (dfw["country"] != 'Africa') & 
    (dfw["country"] != 'Asia') & (dfw["country"] != 'Europe') &
    (dfw["country"] != 'Americas') & (dfw["country"] != 'Eastern Mediterranean') & 
    (dfw["country"] != 'High-income') & (dfw["country"] != 'Low-income') & (dfw["country"] != 'Lower-middle-income') & 
    (dfw["country"] != 'South-East Asia') & (dfw["country"] != 'Upper-middle-income') & (dfw["country"] != 'Western Pacific')]


#Visualizar DataFrame Atualizado
if st.sidebar.checkbox("🧾**Mostrar Dados Global**", False, key=0):
    st.markdown(":red[**Dados Mundial sobre Suicídios 2010-2019**]")
   
    with st.expander("🗓 **Visualizar DataFrame World Atualizado**"):
        st.dataframe(data_w, use_container_width=True)

    st.divider()

#Visualizar DataFrame original
    st.markdown(":red[**Dados detalhados:**]:")
    with st.expander("🗓 **Visualizar DataFrame detalhado**"):
        st.dataframe(df, use_container_width=True)

    st.markdown("")    
    st.markdown("Fonte Dataset: https://ourworldindata.org/suicide")


#Dataset Global exceto o Brasil
data = df[df.country != "Brazil"].copy()

#10 Países com o maior índice de casos suicidios
if st.sidebar.checkbox(" 🔟 **Mostrar índices**", False, key=1):
    st.write('🔟**Países com o maior índice de casos suicídios**:')
    df_group10 = df.groupby("country")[["suicides_no"]].sum()
    top_10_population = pd.DataFrame(df_group10).sort_values(by="suicides_no", ascending=False)[:10]
    st.dataframe(top_10_population, use_container_width=True)
    st.divider()
#Gráfico top 10
    if not st.checkbox('Ocultar gráfico 1', False, key=2):
        st.write('📊 **Gráfico dos 10 Países com o maior índice de casos**: ')
        data2 = df.groupby("country")[["country", "suicides_no"]].max().sort_values(by="suicides_no", ascending=False)[:10]
        fig1 = px.bar(data2, x="country" , y="suicides_no", color="suicides_no")
        st.plotly_chart(fig1)

###################################################################################################################

#agrupar os dados


df2 = data.groupby('country')[['suicides_no', 'suicides/100k pop']].sum().reset_index()
df3 = data.groupby('country')[['population', 'suicides_no', 'suicides/100k pop']].mean().reset_index()
df4 = data.groupby(['country', 'year'])[['suicides_no', 'suicides/100k pop', 'gdp_per_capita ($)']].max().reset_index()

#st.table(df4)

#### Análise de dados por país##########################################
if st.sidebar.checkbox("📝 **Mostrar análise por País**", False, key=3):
    st.markdown("📊 **Análise de casos de suicídios por país**:")
    
    #Colunas
    col1 , col2 = st.columns(2)
    with col1:
    #Selecionar por país
        country_select = st.selectbox('🔎 Selecionar o país:', df2['country'])
        select_country = df2[df2['country'] == country_select]
        select_country2 = df3[df3['country'] == country_select]
        select_country3 = df4[df4['country'] == country_select]
    
    st.markdown("")
    #Função para visualizar os resultados filtrados por país (Total de suicidios e suicidios /100k pop)
    def get_cases_analysis(dataresult):
        total_res = pd.DataFrame({'Status': ['Num Suicidio', 'Total Suicidio/100K pop'],
                                'Figure':(dataresult.iloc[0]['suicides_no'], dataresult.iloc[0]['suicides/100k pop'])
                                })
        
        return total_res

    total_country = get_cases_analysis(select_country)  

    # Data Visualization
    country_no_suicides = total_country.Figure[0]
    country_suicides_100k_pop = total_country.Figure[1]

    st.markdown(" 🗂 Informações sobre total de nº suicídios por país:")
    # 1ª linha dos resultados dos dados
    col1, col2, col3 = st.columns(3)
    col1.text("📍País:")
    col1.info(country_select)
    col2.text("✔ Total Suicidios:")
    col2.info(f"{country_no_suicides:,.0f}")
    col3.text("👨‍👨‍👧‍👧Total Suicídios/100Kpop:")
    col3.info(f"{country_suicides_100k_pop :,.0f}")
    st.divider()   

    #Função para visualizar os resultados filtrados por país (Média Populacional, média de suicidios e suicidios /100k pop)
    def get_mean_analysis(dataresult2):
        mean_res = pd.DataFrame({'Status': ['population','Num Suicidio', 'Total Suicidio/100K pop'],
                                'Figure':(dataresult2.iloc[0]['population'], dataresult2.iloc[0]['suicides_no'], dataresult2.iloc[0]['suicides/100k pop'])
                                })
        
        return mean_res

    mean_country = get_mean_analysis(select_country2)  

    # Data Visualization
    mean_population = mean_country.Figure[0]
    mean_no_suicides = mean_country.Figure[1]
    mean_suicides_100k_pop = mean_country.Figure[2]

    
    st.markdown("💹 Médias dos índices por país:")
    # 1ª linha dos resultados dos dados
    col1, col2, col3 = st.columns(3)
    col1.text("📍Média Populacional:")
    col1.info(f"{mean_population :,.2f}")
    col2.text("✔ Média de Suicidios:")
    col2.info(f"{mean_no_suicides:,.2f}")
    col3.text("👨‍👨‍👧‍👧Média Suicídios/100Kpop:")
    col3.info(f"{mean_suicides_100k_pop :,.2f}")
    st.divider()   

    #Função para visualizar os resultados filtrados por país (Maiores indice de nº suicidios e suicidios /100k pop)
    def get_max_analysis(dataresult3):
        max_res = pd.DataFrame({'Status': ['Num Suicidio', 'Total Suicidio/100K pop', 'gdp_per_capita ($)'],
                                'Figure':(dataresult3.iloc[0]['suicides_no'], dataresult3.iloc[0]['suicides/100k pop'], dataresult3.iloc[0]['gdp_per_capita ($)'],)
                                })
        
        return max_res

    max_country = get_max_analysis(select_country3)  

    # Data Visualization
    max_no_suicides = max_country.Figure[0]
    max_suicides_100k_pop = max_country.Figure[1]
    max_gdp_per_capita = max_country.Figure[2]

    st.markdown("📊 Maiores índices por país e ano:")
    # 1ª linha dos resultados dos dados
    col1, col2, col3 = st.columns(3)
    col1.text("✔ Nº de Suicídios:")
    col1.info(f"{max_no_suicides:,.2f}")
    col2.text("👨‍👨‍👧‍👧Suicídios/100Kpop:")
    col2.info(f"{max_suicides_100k_pop :,.2f}")
    col3.text("📍PIB per Capita ($):")
    col3.info(f"{max_gdp_per_capita :,.2f}")
    st.divider()   


########## Dados para exibição do grafíco global ####################
data_Brasil = data_w[data_w.country == "Brazil"].copy()
years = data_w.year.unique()    # pegar os anos para o eixo x
pais = data_w.country.unique()
suicides_world_suicides_brasil = data_Brasil.groupby('year')['suicides/100k pop'].mean()
suicides_world_suicides_world = data_w.groupby('year')['suicides/100k pop'].mean()   

#########Gráficos total por país######################################

if st.sidebar.checkbox(" 📊 **Mostrar Gráficos**", False, key=4):
    st.markdown("📊 **Gráfico de casos de suicídios por país**:")
    
       
    if not st.checkbox('Ocultar gráfico 1', False, key=5):
        fig, ax = plt.subplots(figsize = (12,6))
        ax = sns.lineplot(x=years, y=suicides_world_suicides_brasil, label='Brasil')
        ax = sns.lineplot(x=years, y=suicides_world_suicides_world, label='Mundo')
        plt.legend(title="Taxa média de suicídio")
        plt.title('Gráfico Comparativo Brasil x Mundo')
        plt.xlim(2000, 2019)
        plt.tight_layout()
        st.pyplot(fig)
        st.info('O gráfico acima demonstra é que apesar da taxa de suicídios no mundo vem diminuindo, no Brasil ela está crescendo desde de 2010.')
        st.divider()

    ## Gráfico 2
    if not st.checkbox('Ocultar gráfico 2', False, key=6):
        st.write("🌎 Gráfico global anual de nº casos suicídios por 100K população por país")
        fig2 = px.choropleth(data_w,
                locations='code',
                color="suicides/100k pop",
                hover_name="country",
                color_continuous_scale="Viridis",
                animation_frame="year" )
        st.plotly_chart(fig2, theme='streamlit', use_container_width=True)

    st.divider()

###### Dados Previsões futuras usando Machine Learning: 2000 - 2019
if st.sidebar.checkbox(" ⏰ **Previsões futuras**", False, key=7):
    st.markdown("📊 **Previsões de novos casos no mundo**:")        

    @st.cache_data
    def load_data1():
        data = pd.read_csv("data/suicides_cases.csv")
        return data

    dataset = load_data1()

    
    #Machine Learning 
    df_brasil = dataset[dataset.country == "Brazil"].copy()
    df_world = dataset[dataset.country == "World"].copy()

    anos = df_world[['year']]
    casos = df_world[['cases']]

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
        st.info(f'👨‍👩‍👧‍👧 **{novos_casos}** casos por 100k população mundial.')
    st.divider()

    #### Medição do modelo
    accuracy = model.score(anos,casos)
    st.markdown('🎯Accurácia de acerto do modelo de Machine Learning - IA')
    st.info(f'Accuracy: {round(accuracy*100,2)} %')
    st.divider()

    st.info('✅ Conclusão: A tendência que ao passar dos anos a taxa de suicídio mundial diminua, devido as campanhas realizadas pelo governo  sobre a importância da conscientização e do cuidado com a saúde mental. Informação é o primeiro passo de qualquer tratamento. No Brasil é realizado o Setembro Amarelo.')
       



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