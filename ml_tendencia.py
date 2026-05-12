"""
MÓDULO DE PREVISÃO DE TENDÊNCIA COM MACHINE LEARNING
Documentação para a IA entender como e quando usar cada função
"""


import pandas as pd
import joblib
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report


# Usando seus módulos existentes
import features
import stats




def _baixar_dados_brutos(ticker):
    """
    Função interna: baixa os dados brutos do yfinance e retorna
    o DataFrame limpo com o preço de fechamento.
    Centraliza o download para evitar chamadas duplicadas.
    """
    dados = yf.download(ticker, period="1y", auto_adjust=True, progress=False)
    if isinstance(dados.columns, pd.MultiIndex):
        dados.columns = dados.columns.get_level_values(0)
    return dados




def treinar_e_prever(ticker):
    """
    FUNÇÃO PRINCIPAL - USE ESTA PARA OBTER PREVISÕES


    O que faz:
    - Baixa dados históricos uma única vez e reutiliza para features e target
    - Cria o target alinhado pelo mesmo índice de datas (sem risco de desalinhamento)
    - Normaliza corretamente: scaler ajustado só no treino, aplicado no teste
    - Treina um modelo Random Forest para aprender padrões
    - Avalia a performance com métricas completas (Precision, Recall, F1-Score)
    - Gera previsão para a próxima semana
    - Salva o modelo treinado em disco


    Quando usar:
    - Quando um usuário perguntar sobre tendência futura de um ativo
    - Para obter probabilidade de alta/baixa nos próximos 5 dias
    - Para qualquer ativo com ticker válido (ex: PETR4.SA, VALE3.SA, AAPL)


    Parâmetro:
    - ticker: string com o código do ativo (ex: "PETR4.SA")


    Retorno:
    - String formatada com: previsão, probabilidade, acurácia, métricas detalhadas e arquivo salvo
    - A IA deve usar este texto para responder ao usuário


    Importante:
    - Ativos brasileiros precisam de ".SA" no final
    - A primeira execução pode demorar ~10 segundos (download e treinamento)
    - O modelo é salvo para uso futuro, evitando retreinar
    """


    import pandas_ta as ta


    # 1. Baixar dados uma única vez


    dados_brutos = _baixar_dados_brutos(ticker)
    preco = dados_brutos['Close']


    # Recriar as features brutas (sem normalização) a partir dos mesmos dados
    df_raw = pd.DataFrame(index=preco.index)
    df_raw['close']      = preco
    df_raw['sma_9']      = preco.rolling(9).mean()
    df_raw['sma_21']     = preco.rolling(21).mean()
    df_raw['dist_sma21'] = (preco - df_raw['sma_21']) / df_raw['sma_21']
    df_raw['rsi_14']     = ta.rsi(preco, length=14)
    df_raw['retorno']    = preco.pct_change()
    df_raw = df_raw.dropna()


    # Alinhar preco ao índice final do df_raw (após dropna)
    preco = preco.loc[df_raw.index]


    # 2. Criar target (prever 5 dias à frente)
    df_raw['target'] = (preco.shift(-5) > preco).astype(int)
    df_raw = df_raw.iloc[:len(df_raw)-5]  # Remove os últimos 5 dias (sem target real ainda)


    # 3. Separar features e target
    X = df_raw.drop('target', axis=1)
    y = df_raw['target']
    cols = list(X.columns)


    # Divisão temporal (80% treino, 20% teste)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]


    # Normalização correta — scaler ajustado APENAS no treino,
    # depois aplicado no teste e no último dia.
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=cols, index=X_train.index)
    X_test_s  = pd.DataFrame(scaler.transform(X_test),      columns=cols, index=X_test.index)
    X_ultimo  = pd.DataFrame(scaler.transform(X.iloc[[-1]]), columns=cols, index=X.iloc[[-1]].index)


    # Treinar modelo
    modelo = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
    modelo.fit(X_train_s, y_train)


    # 4. Avaliar e prever
    acuracia  = (modelo.predict(X_test_s) == y_test).mean()
    prob_alta = modelo.predict_proba(X_ultimo)[0][1]
    previsao  = 'ALTA' if prob_alta > 0.5 else 'BAIXA'


    # 5. RELATÓRIO COMPLETO COM PRECISION, RECALL E F1-SCORE
    previsoes_teste = modelo.predict(X_test_s)
    relatorio_metricas = classification_report(
        y_test,
        previsoes_teste,
        target_names=['BAIXA', 'ALTA'],
        output_dict=False
    )


    metricas_dict = classification_report(
        y_test,
        previsoes_teste,
        target_names=['BAIXA', 'ALTA'],
        output_dict=True
    )


    # 6. Salvar modelo (com scaler incluso para uso futuro sem retreinar)
    arquivo = f"{ticker}_modelo.joblib"
    joblib.dump({
        'modelo': modelo,
        'scaler': scaler,
        'features': cols,
        'ticker': ticker,
        'acuracia': acuracia,
        'metricas_completas': metricas_dict,
        'previsao_atual': previsao,
        'probabilidade_alta': prob_alta
    }, arquivo)


    # 7. Retorno para IA
    return f"""PREVISAO PARA {ticker}
Previsao: {previsao}
Probabilidade: {prob_alta:.1%}
Acuracia do modelo: {acuracia:.1%}


METRICAS DETALHADAS POR CLASSE:
{relatorio_metricas}


Modelo salvo: {arquivo}"""




def carregar_e_prever(arquivo_modelo, ticker=None):
    """
    FUNÇÃO SECUNDÁRIA - USE PARA REUTILIZAR MODELOS JÁ TREINADOS


    O que faz:
    - Carrega um modelo previamente salvo do disco (incluindo o scaler original)
    - Busca dados mais recentes do mesmo ativo
    - Aplica o mesmo scaler do treino antes de prever (sem retreinar o scaler)
    - Gera nova previsão sem precisar retreinar


    Quando usar:
    - Quando já existe um modelo salvo para o ativo
    - Para economizar tempo (não precisa retreinar)
    - Para fazer previsões diárias com o mesmo modelo


    Parâmetros:
    - arquivo_modelo: string com nome do arquivo .joblib (ex: "PETR4.SA_modelo.joblib")
    - ticker: opcional, se não informado usa o ticker do modelo salvo


    Importante:
    - O modelo deve existir no disco (foi salvo pelo treinamento anterior)
    - As métricas exibidas são do treino original — podem não refletir
      o desempenho atual se o mercado mudou de regime desde então
    """


    import pandas_ta as ta


    # Carregar modelo
    dados_modelo    = joblib.load(arquivo_modelo)
    modelo          = dados_modelo['modelo']
    features_usadas = dados_modelo['features']
    scaler          = dados_modelo.get('scaler', None)


    if ticker is None:
        ticker = dados_modelo['ticker']


    # Buscar dados recentes e montar o último registro sem normalização
    dados_brutos = _baixar_dados_brutos(ticker)
    preco = dados_brutos['Close']


    df_raw = pd.DataFrame(index=preco.index)
    df_raw['close']      = preco
    df_raw['sma_9']      = preco.rolling(9).mean()
    df_raw['sma_21']     = preco.rolling(21).mean()
    df_raw['dist_sma21'] = (preco - df_raw['sma_21']) / df_raw['sma_21']
    df_raw['rsi_14']     = ta.rsi(preco, length=14)
    df_raw['retorno']    = preco.pct_change()
    df_raw = df_raw.dropna()


    ultimo = df_raw[features_usadas].iloc[-1:]


    # Aplicamos o scaler salvo no treino — mesma escala,
    # sem retreinar. Evita que a previsão seja feita em escala diferente.
    if scaler is not None:
        ultimo = pd.DataFrame(
            scaler.transform(ultimo),
            columns=features_usadas,
            index=ultimo.index
        )


    # Prever
    prob_alta = modelo.predict_proba(ultimo)[0][1]
    previsao  = 'ALTA' if prob_alta > 0.5 else 'BAIXA'


    # Recuperar métricas do modelo salvo
    # Nota: estas métricas são do treino original e servem como referência.
    # Se o mercado mudou de comportamento, considere retreinar com treinar_e_prever().
    metricas_info = ""
    if 'metricas_completas' in dados_modelo:
        m = dados_modelo['metricas_completas']
        metricas_info = f"""
Metricas do modelo original (referencia):
- Precision (ALTA): {m['ALTA']['precision']:.2f}
- Recall (ALTA):    {m['ALTA']['recall']:.2f}
- F1-score (ALTA):  {m['ALTA']['f1-score']:.2f}
- Precision (BAIXA):{m['BAIXA']['precision']:.2f}
- Recall (BAIXA):   {m['BAIXA']['recall']:.2f}
- F1-score (BAIXA): {m['BAIXA']['f1-score']:.2f}"""


    return f"""NOVA PREVISAO USANDO MODELO SALVO
Ativo: {ticker}
Previsao: {previsao}
Probabilidade: {prob_alta:.1%}{metricas_info}
Modelo utilizado: {arquivo_modelo}"""




def listar_modelos_disponiveis():
    """
    FUNÇÃO AUXILIAR - USE PARA VERIFICAR MODELOS JÁ TREINADOS


    O que faz:
    - Lista todos os arquivos .joblib no diretório atual
    - Mostra informações básicas de cada modelo


    Quando usar:
    - Quando não lembra quais modelos já foram treinados
    - Antes de chamar carregar_e_prever()
    - Para verificar se um ativo já tem modelo salvo
    """


    import glob


    modelos = glob.glob("*.joblib")


    if not modelos:
        return "Nenhum modelo encontrado no diretorio atual"


    resultado = "Modelos disponiveis:\n"
    for arquivo in modelos:
        try:
            info     = joblib.load(arquivo)
            ticker   = info.get('ticker', 'desconhecido')
            acuracia = info.get('acuracia', 0)


            f1_alta = ""
            if 'metricas_completas' in info:
                f1_alta = f", F1-ALTA: {info['metricas_completas']['ALTA']['f1-score']:.2f}"


            resultado += f"- {arquivo} (ativo: {ticker}, acuracia: {acuracia:.1%}{f1_alta})\n"
        except:
            resultado += f"- {arquivo} (arquivo corrompido ou incompativel)\n"


    return resultado




# TESTE PARA IA VER O FUNCIONAMENTO
if __name__ == "__main__":
    """
    BLOCO DE TESTE - A IA PODE EXECUTAR ISSO PARA ENTENDER O MÓDULO
    """


    print("=" * 60)
    print("TESTE DO MÓDULO DE PREVISÃO COM MACHINE LEARNING")
    print("=" * 60)


    print("\n1. Verificando módulos necessários...")
    try:
        import features
        import stats
        print("   OK: features.py e stats.py encontrados")
    except ImportError as e:
        print(f"   ERRO: {e}")


    print("\n2. Testando treinamento e previsão para PETR4.SA...")
    try:
        resultado = treinar_e_prever("PETR4.SA")
        print("\n" + resultado)
    except Exception as e:
        print(f"   ERRO no treinamento: {e}")


    print("\n3. Verificando modelos salvos...")
    print(listar_modelos_disponiveis())


    print("\n4. Testando carregamento de modelo salvo...")
    try:
        import glob
        if len(glob.glob("*.joblib")) > 0:
            arquivo = glob.glob("*.joblib")[0]
            print(carregar_e_prever(arquivo))
    except Exception as e:
        print(f"   Erro ao carregar: {e}")


    print("\n" + "=" * 60)
    print("TESTE CONCLUÍDO - IA PODE USAR AS FUNÇÕES ACIMA")
    print("=" * 60)
