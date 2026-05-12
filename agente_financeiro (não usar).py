from mcp.server.fastmcp import FastMCP
import yfinance as yf
import pandas as pd
import stats  # <-- Importando NOSSA biblioteca da aula 2

# 1. Criamos o servidor do Agente
mcp = FastMCP("Agente Quant CF Tech")

# 2. Ferramenta 1: Análise Individual
@mcp.tool()
def analisar_ativo(ticker: str) -> str:
    """
    Realiza uma análise quantitativa completa de um ativo.
    Entrada: Ticker do ativo (ex: 'PETR4.SA', 'VALE3.SA', 'AAPL').
    Retorno: Texto com Volatilidade, Sharpe e Drawdown.
    """
    # Tratamento simples do Ticker
    ticker = ticker.upper() 

    # --- PASSO A: Baixar Dados ---
    try:
        # Baixa últimos 2 anos. progress=False esconde a barrinha de carregamento
        dados = yf.download(ticker, period="2y", auto_adjust=True, progress=False)
        
        # --- PASSO B: Limpeza de Dados (Anti-Erro) ---
        # Se o pandas criar MultiIndex (cabeçalho duplo), removemos o nível extra
        if isinstance(dados.columns, pd.MultiIndex):
            dados.columns = dados.columns.get_level_values(0)
            
        # Selecionamos apenas o Fechamento
        precos = dados['Close']
        
        # Verificação de segurança: O ticker existe? Veio vazio?
        if len(precos) == 0:
            return f"Erro: Não encontrei dados para o ticker {ticker}. Verifique se adicionou .SA no final."
            
    except Exception as e:
        return f"Erro crítico ao acessar Yahoo Finance: {str(e)}"

    # --- PASSO C: O Cérebro Matemático ---
    # Aqui usamos as funções que criamos na Aula 2
    retornos = stats.calcular_retornos_log(precos)
    metricas = stats.calcular_metricas_risco(retornos)
    drawdown = stats.calcular_drawdown(precos)
    
    # --- PASSO D: A Resposta para a IA ---
    # Montamos um texto claro que a IA vai ler e interpretar
    relatorio = f"""
    --- ANÁLISE QUANT: {ticker} ---
    • Retorno Esperado (Anual): {metricas['retorno_anual_esp'] * 100:.2f}%
    • Volatilidade (Risco): {metricas['volatilidade'] * 100:.2f}%
    • Índice de Sharpe: {metricas['sharpe']:.2f}
    • Max Drawdown (Pior Queda): {drawdown * 100:.2f}%
    """
    
    # Adicionamos uma "opinião" automática baseada no Sharpe
    if metricas['sharpe'] > 1.0:
        relatorio += "\nVEREDICTO: Ativo Eficiente (Retorno compensa o risco)."
    elif metricas['sharpe'] < 0:
        relatorio += "\nVEREDICTO: Ineficiente (Retorno negativo no período)."
    else:
        relatorio += "\nVEREDICTO: Neutro / Observar."
        
    return relatorio

# 3. Ferramenta 2: Comparação (Desafio Extra)
@mcp.tool()
def comparar_ativos(ticker1: str, ticker2: str) -> str:
    """
    Compara dois ativos lado a lado.
    Entrada: Dois tickers (ex: 'PETR4.SA', 'VALE3.SA').
    Retorno: Os relatórios dos dois ativos.
    """
    # 1. Chamamos a nossa própria função interna para fazer o trabalho
    relatorio1 = analisar_ativo(ticker1)
    relatorio2 = analisar_ativo(ticker2)
    
    # 2. Montamos a resposta combinada
    resposta_final = f"""
    === COMPARAÇÃO DE ATIVOS ===
    
    {relatorio1}
    
    ----------------------------
    
    {relatorio2}
    """
    return resposta_final

# --- SIMULAÇÃO (Para rodar no terminal hoje) ---
if __name__ == "__main__":
    print("--- INICIANDO BATERIA DE TESTES ---")
    
    # Teste 1: Ferramenta Individual
    print("\n>>> Testando Análise Individual (PETR4):")
    print(analisar_ativo("PETR4.SA"))
    
    # Teste 2: Ferramenta de Comparação (Desafio)
    print("\n>>> Testando Comparação (BBAS3 vs ITUB4):")
    print(comparar_ativos("BBAS3.SA", "ITUB4.SA"))
    
    # Teste 3: Tratamento de Erro
    print("\n>>> Testando Ticker Inválido:")
    print(analisar_ativo("NAOEXISTE"))
