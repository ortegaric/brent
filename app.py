# Bibliotecas

import pandas as pd 

import plotly.express as px

import streamlit as st 

from sklearn.ensemble import RandomForestRegressor


# Configurações

st.set_page_config(page_title='Brent', layout='wide')


# Funções

def baixar_dados(url, pular_linhas=0):
  try:
    return pd.read_html(url, skiprows=pular_linhas )
  except Exception as e:
    print(f'Erro ao tentar extrair dados: {e}')
    return None


# Dados: Brent
  
url_brent = 'http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view'

dados_brent = baixar_dados(url_brent, 1)

if dados_brent is None:
  brent = pd.read_csv('./dados/brent_raw.csv', sep=';')
else:
  brent = dados_brent[0].rename(columns={0:'data', 1:'preco'})
  brent.to_csv('./dados/brent_raw.csv', sep=';', index=False)

brent.data = pd.to_datetime(brent.data, format='%d/%m/%Y')

brent.preco = brent.preco / 100

brent['crescimento'] = brent.preco.pct_change().fillna(0) * 100

brent['volatilidade'] = brent.crescimento.rolling(window=7).std()

brent.dropna(inplace=True)

brent.reset_index(drop=True, inplace=True)

brent = brent.round(2)

brent = brent.sort_values(by='data', ascending=True).reset_index(drop=True)


# Dados PIB Mundial (GDP)

url_pib = 'https://data.worldbank.org/indicator/NY.GDP.MKTP.CD'

dados_pib = pd.read_csv('./dados/gdp_raw.csv', sep=',', decimal='.')

pib = pd.melt(dados_pib, var_name='ano', value_name='total')

pib = pib.iloc[1:-1]

pib.ano = pib.ano.astype('int')

pib.total = pib.total.astype('float')

pib['crescimento'] = (pib.total.pct_change() * 100).round(2)


# Conectando dados dos dois indicadores

df = brent[['data', 'preco']].copy()

df.insert(loc=1, column='ano', value=df.data.dt.year)

df.drop('data', axis=1, inplace=True)

df = df.groupby(by='ano', as_index=False).mean().round(2)

df.rename(columns={'preco': 'brent_preco_medio'}, inplace=True)

df['brent_crescimento'] = (df.brent_preco_medio.pct_change() * 100).round(2)

df = pd.merge(df, pib, on='ano', how='inner')

df.rename(columns={'total': 'pib_total', 'crescimento': 'pib_crescimento'}, inplace=True)

df.dropna(inplace=True)


# Modelo: Random Forest

ts_brent = brent[['data', 'preco']].rename(columns={'data':'ds', 'preco':'y'}).set_index('ds', drop=True).copy()

ts_brent = ts_brent.reindex(index=pd.date_range(start=ts_brent.index.min(), end=ts_brent.index.max(), normalize=True, freq='D'), method='ffill').rename_axis('ds')

ts_brent['lag'] = ts_brent.y.shift(1)

ts_brent.dropna(inplace=True)

train_size = int(len(ts_brent) * 0.8)

train, test = ts_brent[:train_size], ts_brent[train_size:]

X_train, y_train = train['lag'].values.reshape(-1, 1), train['y']

X_test, y_test = test['lag'].values.reshape(-1, 1), test['y']

model = RandomForestRegressor(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

x = ts_brent[['lag']]

y = ts_brent.y

future_dates = pd.date_range(start=ts_brent.index[-1] + pd.Timedelta(days=1), periods=30)

future_lag = ts_brent['y'].tail(30).values

future_df = pd.DataFrame({'lag': future_lag}, index=future_dates)

future_predictions = model.predict(future_df[['lag']])

future_df['predictions'] = future_predictions


# Gráficos

fig_brent_tempo = px.line(brent, 
                          y='preco', x='data',
                          labels={'preco': 'Preço (US$)', 'data': 'Data'}, 
                          title='Distribuição do preço do Brent no tempo')

fig_brent_volatilidade = px.line(brent,
                                 y='volatilidade', x='data',
                                labels={'volatilidade': 'Volatilidade (%)', 'data': 'Data'},
                                title='Volatilidade do preço do Brent em uma semana')

fig_pib_tempo = px.line(pib, y='total', x='ano',
                        labels={'total': 'Total (US$)', 'ano': 'Data'},
                        title='PIB Mundial')

fig_brent_pib = px.bar(df, y=['brent_crescimento', 'pib_crescimento'], x='ano',
                        labels={
                            'value': 'Crescimento (%)',
                            'ano': 'Data',
                            'variable': 'Indicadores'},
                        title='Comparação de crescimento anual dos indicadores')

fig_correlacao = px.imshow(df[['brent_preco_medio', 'pib_total']].corr(numeric_only=True).round(2), text_auto=True)


fig_brent_boxplot = px.box(brent, x='preco',
                           labels={'preco': 'Preço (US$)'},
                           title='Distribuição do preço do Brent')

fig_brent_treino = px.line(pd.DataFrame(data={'real':test.y, 'previsto':predictions}),
                            labels={'value': 'Preço (US$)', 'ds': 'Data'},
                            title='Análise de Previsão: Random Forest')

fig_brent_previsto = px.line(future_df[['predictions']], y='predictions',
                            labels={'value': 'Preço (US$)', 'index': 'Data'},
                            title='Previsão para os próximos 30 dias')


# Header

st.title('Preço do petróleo Brent', help='Título')


# Layout

tab1, tab2, tab3, tab4, tab5 = st.tabs(['Dashboard', 'Relação com PIB', 'Outros indicadores', 'Previsões', 'Dados'])

with tab1:

    st.header('Dashboard')

    st.metric(label='Ultima atualização', value=str(brent.data.iloc[-1])[:10])

    with st.container():

        col1, col2, col3 = st.columns(3)
    
        with col1:

            st.metric(label='Preço (US$)', value=brent.preco.iloc[-1], delta=(f'{brent.crescimento.iloc[-1]} %'))

        with col2:
           
           st.metric(label='Mínimo (US$)', value=brent.preco.min(), delta=(brent.preco.iloc[-1]-brent.preco.min()).round(2))

        with col3:
           
           st.metric(label='Máximo (US$)', value=brent.preco.max(), delta=(brent.preco.iloc[-1]-brent.preco.max()).round(2))
           
    st.plotly_chart(fig_brent_tempo, use_container_width=True)

    st.plotly_chart(fig_brent_volatilidade, use_container_width=True)

    with st.expander('Insights'):
        
        st.markdown('Cinco insights relevantes sobre os preços do petróleo do tipo Brent ao longo do tempo:')
        st.markdown('**Sensibilidade a Eventos Geopolíticos:**')
        st.markdown('Os preços do petróleo do tipo Brent frequentemente mostram sensibilidade a eventos geopolíticos significativos, como conflitos no Oriente Médio ou tensões entre grandes produtores. Esses eventos podem resultar em picos de volatilidade nos preços devido a preocupações com a oferta global.')
        st.markdown('**Influência da Produção da OPEP:**')
        st.markdown('A Organização dos Países Exportadores de Petróleo (OPEP) desempenha um papel crucial na determinação dos preços do Brent. Decisões da OPEP relacionadas à produção têm impacto direto nos níveis de oferta e, portanto, nos preços do petróleo.')
        st.markdown('**Reação a Mudanças na Demanda Global:**')
        st.markdown('Os preços do Brent são influenciados por mudanças na demanda global por petróleo. Eventos econômicos, como recessões ou períodos de crescimento econômico robusto, podem afetar significativamente a demanda, impactando os preços do petróleo.')
        st.markdown('**Correlação com Indicadores Econômicos:**')
        st.markdown('Há uma correlação observável entre os preços do Brent e indicadores econômicos globais, como o crescimento do Produto Interno Bruto (PIB). Um crescimento econômico saudável muitas vezes está associado a uma maior demanda por energia, influenciando positivamente os preços do petróleo.')
        st.markdown('**Adaptação a Mudanças no Mix Energético:**')
        st.markdown('A evolução do mix energético global, com um aumento no investimento em energias renováveis e a busca por alternativas mais sustentáveis, pode impactar os preços do petróleo. Mudanças nas percepções sobre a transição para fontes de energia mais limpas podem influenciar a demanda por petróleo e, consequentemente, os preços do Brent.')

    with st.expander('Eventos recentes'):

        st.markdown('**2020: A Tempestade Perfeita da Pandemia**')
        st.markdown('A pandemia de COVID-19 causou uma queda massiva na demanda por petróleo, resultando em uma queda drástica nos preços do Brent. Os produtores, então, enfrentaram o desafio de se ajustar a uma nova realidade de oferta e demanda.')
        st.markdown('**2021-2022: A Recuperação Gradual**')
        st.markdown('Com a implementação de vacinas contra a COVID-19 e a retomada das atividades econômicas, os preços do Brent começaram a se recuperar gradualmente, refletindo a esperança de uma recuperação econômica global.')
        st.markdown('**2023: Novos conflitos**')
        st.markdown('Desde o início do último conflito no Oriente Médio, o preço do Brent aumentou 6% devido à incerteza sobre o impacto do conflito na oferta e é provável que enfrente uma volatilidade contínua.')

with tab2:
   
    st.header('Relação entre Brent e PIB')

    st.metric(label='Ultima atualização', value=str(pib.ano.iloc[-1]))

    st.metric(label='Total Mundial (US$)', value=f'{pib.total.iloc[-1]:,.0f}', delta=f'{pib.crescimento.iloc[-1]} %')

    st.plotly_chart(fig_pib_tempo, use_container_width=True)

    st.plotly_chart(fig_brent_pib, use_container_width=True)

    with st.expander('Correlação'):
       st.markdown('Há uma correlação observável entre os preços do Brent e indicadores econômicos globais, como o crescimento do Produto Interno Bruto (PIB). Um crescimento econômico saudável muitas vezes está associado a uma maior demanda por energia, influenciando positivamente os preços do petróleo.')
       st.plotly_chart(fig_correlacao, use_container_width=True)

with tab3:

    st.header('Outros indicadores')

    st.metric(label='Ultima atualização', value='2022')

    st.header('Consumo mundial de energia')

    st.markdown('Variação do consumo mundial de energia de acordo com o U.S. Energy Information Administration.')

    st.markdown('<iframe src="https://ourworldindata.org/grapher/change-energy-consumption" loading="lazy" style="width: 100%; height: 600px; border: 0px none;"></iframe>',
                unsafe_allow_html=True)
    
    st.header('Produção mundial de petróleo')

    st.markdown('Produção de petróleo nas principais regiõees mundiais de acordo com o U.S. Energy Intitute')
    
    st.markdown('<iframe src="https://ourworldindata.org/grapher/oil-production-by-country" loading="lazy" style="width: 100%; height: 600px; border: 0px none;"></iframe>',
                unsafe_allow_html=True)

with tab4:
   
   st.header('Previsões')

   with st.container():

        col1, col2 = st.columns(2)
    
        with col1:

            st.metric(label='Próxima data prevista', value=str(future_df.index[0])[:10])

        with col2:
           
           st.metric(label='Próximo preço previsto (US$)', value=(future_df.predictions.iloc[0]).round(2), delta=(future_df.predictions.iloc[0]-ts_brent.y.iloc[-1]).round(2))

   st.plotly_chart(fig_brent_previsto, use_container_width=True)

   with st.expander('Modelo: Randon Forest'):
      
      st.markdown('O modelo de previsão dos preços escolhidos foi o Randon Forest, com um lag de um (1) dia, apresentando uma margem de erro de aproximadamente 2 dólares para cima ou para baixo.')

      st.plotly_chart(fig_brent_treino, use_container_width=True)

with tab5:

    st.header('Dados: Brent')
    
    st.dataframe(brent)

    st.markdown('[Fonte: ipeadata | Brent](http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view)')

    st.header('Dados: PIB Mundial')

    st.dataframe(pib)

    st.markdown('[Fonte: Banco Mundial | GDP](https://data.worldbank.org/indicator/NY.GDP.MKTP.CD)')

    
# Ricardo Ortega - FIAP - PosTech Data Analytics
