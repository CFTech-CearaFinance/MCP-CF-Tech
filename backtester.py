import pandas as pd
import vectorbt as vbt
import joblib
import yfinance as yf
import pandas_ta as ta

def rodar_backtest(ticker: str, arquivo_modelo: str) -> str:
    """
    Simula o retorno financeiro do modelo de Machine Learning.
    Entrada: Ticker do ativo e o nome do arquivo .joblib do modelo.
    Retorno: Relatório financeiro comparando a estratégia com o Buy & Hold.
    """
    # 1. Carregar o modelo e o padronizador (Scaler) já treinados
    try:
        dados_modelo = joblib.load(arquivo_modelo)
        modelo = dados_modelo['modelo']
        features_usadas = dados_modelo['features']
        scaler = dados_modelo['scaler']
    except FileNotFoundError:
        return f"Erro: Arquivo {arquivo_modelo} não encontrado. Rode o ml_tendencia.py primeiro."

    # 2. Baixar dados históricos (Teste de 2 anos)
    dados = yf.download(ticker, period="2y", auto_adjust=True, progress=False)
    if isinstance(dados.columns, pd.MultiIndex):
        dados.columns = dados.columns.get_level_values(0)
    preco = dados['Close']

    # 3. Recriar as mesmas features que o modelo aprendeu a usar
    df_raw = pd.DataFrame(index=preco.index)
    df_raw['close'] = preco
    df_raw['sma_9'] = preco.rolling(9).mean()
    df_raw['sma_21'] = preco.rolling(21).mean()
    df_raw['dist_sma21'] = (preco - df_raw['sma_21']) / df_raw['sma_21']
    df_raw['rsi_14'] = ta.rsi(preco, length=14)
    df_raw['retorno'] = preco.pct_change()

    df_raw = df_raw.dropna()
    preco_teste = preco.loc[df_raw.index]

    # 4. Normalizar os dados (usando o MESMO scaler do treino) e Prever
    X = df_raw[features_usadas]
    X_scaled = pd.DataFrame(scaler.transform(X), columns=features_usadas, index=X.index)
    
    # O modelo retorna 1 (ALTA) ou 0 (BAIXA)
    previsoes = modelo.predict(X_scaled)

    # 5. Criar Sinais de Execução para o VectorBT
    ##Pegamos todos os 1s e 0s e entregamos para o vectorbt. Falamos para ele: "Comece com R$ 10.000,00. Toda vez que tiver um sinal de compra, compre tudo o que der. Se tiver sinal de venda, venda tudo. E lembre-se de cobrar 0.1% de taxa da B3 por operação.
    sinais_compra = previsoes == 1
    sinais_venda = previsoes == 0

    # 6. A Simulação Financeira (Portfólio)
    portfolio = vbt.Portfolio.from_signals(
        close=preco_teste,
        entries=sinais_compra,
        exits=sinais_venda,
        init_cash=10000.0, # Começamos com 10 mil reais
        fees=0.001         # Taxa de 0.1% por trade (simulando B3)
    )

    # 7. Extrair Métricas de Resultado
    retorno_estrategia = portfolio.total_return() * 100
    win_rate = portfolio.trades.win_rate() * 100
    max_drawdown = portfolio.max_drawdown() * 100
    
    # Calcular o retorno do mercado (comprar no dia 1 e esquecer)
    retorno_buy_hold = ((preco_teste.iloc[-1] - preco_teste.iloc[0]) / preco_teste.iloc[0]) * 100

    veredicto = "VENCEU O MERCADO" if retorno_estrategia > retorno_buy_hold else "PERDEU PARA O MERCADO"

    relatorio = f"""
    === BACKTEST VECTORIZADO: {ticker} ===
    • Capital Inicial: R$ 10.000,00
    • Capital Final: R$ {portfolio.final_value():.2f}
    
    [ PERFORMANCE ]
    • Retorno da IA: {retorno_estrategia:.2f}%
    • Retorno Buy & Hold: {retorno_buy_hold:.2f}%
    
    [ RISCO E PRECISÃO ]
    • Taxa de Acerto (Trades): {win_rate:.2f}%
    • Max Drawdown (Pior Queda): {max_drawdown:.2f}%
    
    RESULTADO FINAL: {veredicto}
    """
    return relatorio

# --- SIMULAÇÃO ---
if __name__ == "__main__":
    print("Iniciando Motor de Backtest...\n")
    # Usa o modelo que geramos na aula anterior
    print(rodar_backtest("PETR4.SA", "PETR4.SA_modelo.joblib"))


