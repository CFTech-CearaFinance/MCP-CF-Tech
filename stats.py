import numpy as np
import pandas as pd

def calcular_retornos_log(series_precos):
    """
    Calcula retornos logarítmicos.
    Propriedade: São aditivos no tempo, essenciais para modelos estatísticos.
    """
    # Usamos o logaritmo natural (np.log) da razão entre preços
    return np.log(series_precos / series_precos.shift(1))

def calcular_metricas_risco(series_retornos):
    """
    Retorna um dicionário com Volatilidade e Sharpe Ratio.
    Assume Risk Free = 0 para simplificação didática, ou
    podemos fixar uma taxa base (ex: 10% ao ano).
    """
    # 1. Volatilidade Anualizada
    vol_diaria = series_retornos.std()
    vol_anual = vol_diaria * np.sqrt(252)
    
    # 2. Retorno Total Anualizado (Média simples expandida)
    # A média dos logs multiplicada pelo tempo nos dá o retorno esperado geométrico
    retorno_medio_diario = series_retornos.mean()
    retorno_anual = retorno_medio_diario * 252
    
    # 3. Índice de Sharpe (Simplificado: Risco Zero = 0%)
    # Sharpe = Retorno / Risco
    sharpe_ratio = retorno_anual / vol_anual if vol_anual > 0 else 0
    
    return {
        "volatilidade": vol_anual,
        "sharpe": sharpe_ratio,
        "retorno_anual_esp": retorno_anual
    }

def calcular_drawdown(series_precos):
    """
    Calcula o 'Max Drawdown' (A dor máxima do investidor).
    """
    pico = series_precos.cummax()
    drawdown = (series_precos - pico) / pico
    return drawdown.min()
