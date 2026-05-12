import yfinance as yf
import pandas as pd
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler

def build_features(ticker: str) -> pd.DataFrame:
    """
    Data Factory: Baixa os dados brutos e gera as features normalizadas
    prontas para o modelo de Machine Learning.
    """
    
    # 1. Baixar Dados Brutos (Último 1 ano)
    dados = yf.download(ticker, period="1y", auto_adjust=True, progress=False)
    
    # Limpeza de segurança (MultiIndex do YFinance)
    if isinstance(dados.columns, pd.MultiIndex):
        dados.columns = dados.columns.get_level_values(0)
        
    # Separamos apenas o Fechamento e renomeamos para minúsculo
    df = dados[['Close']].copy()
    df = df.rename(columns={'Close': 'close'})

    # 2. Criação das Features (Contexto)
    df['sma_9'] = df['close'].rolling(9).mean()               # Tendência curta
    df['sma_21'] = df['close'].rolling(21).mean()             # Tendência média
    df['dist_sma21'] = (df['close'] - df['sma_21']) / df['sma_21'] # Distância do preço até a média
    df['rsi_14'] = ta.rsi(df['close'], length=14)             # Indicador de Momentum
    df['retorno'] = df['close'].pct_change()                  # Variação percentual diária

    # 3. Limpeza Final
    # As médias móveis e o RSI precisam de dias anteriores para calcular, 
    # então as primeiras linhas ficarão vazias (NaN). Precisamos removê-las.
    df = df.dropna()

    # 4. Normalização (Colocar tudo na mesma régua)
    features_para_normalizar = ['close', 'sma_9', 'sma_21', 'dist_sma21', 'rsi_14', 'retorno']
    
    scaler = StandardScaler()
    # O StandardScaler transforma os dados e os coloca de volta no DataFrame
    df[features_para_normalizar] = scaler.fit_transform(df[features_para_normalizar])

    return df

# --- TESTE LOCAL ---
if __name__ == "__main__":
    print("Iniciando a Data Factory...\n")
    
    ticker_teste = "PETR4.SA"
    print(f"Gerando features normalizadas para: {ticker_teste}")
    
    # Chamamos a nossa fábrica
    df_pronto_para_ml = build_features(ticker_teste)
    
    # Exibimos as 5 primeiras linhas processadas
    print("\n[ RESULTADO - HEAD DO DATAFRAME ]")
    print(df_pronto_para_ml.head())
    
    print("\nPerceba que os números não são mais 'Preços em Reais', mas sim valores padronizados (Z-Scores). O terreno está pronto!")

