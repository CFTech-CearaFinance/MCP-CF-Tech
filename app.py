import streamlit as st
import google.generativeai as genai
import ml_tendencia 
import stats  # Trazendo o seu motor estatístico de volta!
import yfinance as yf
import pandas as pd
import os
import re

# --- 1. CONFIGURAÇÃO PROFISSIONAL DA PÁGINA ---
st.set_page_config(
    page_title="MCP do Ceará Finance",
    page_icon="📈",
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- 2. BARRA LATERAL (BRANDING E CONFIGURAÇÃO) ---
with st.sidebar:
    st.markdown("<h1 style='text-align: center; font-size: 35px;'>MCP do Ceará Finance</h1>", unsafe_allow_html=True)
    
    caminho_logo = "Logo Ceara Finance.png"
    if os.path.exists(caminho_logo):
        st.image(caminho_logo, use_container_width=True)
    else:
        st.error(f"Erro: Arquivo '{caminho_logo}' não encontrado.")
    
    st.markdown("---")
    st.header("⚙️ Configuração")
    api_key = st.text_input("Cole sua Chave API do Gemini:", type="password")
    
    st.markdown("---")
    st.markdown("Desenvolvido por **CF Tech**")

if not api_key:
    st.warning("👈 Por favor, insira sua chave da API do Gemini na barra lateral para começar.")
    st.stop()

# --- 3. NOVA FERRAMENTA DE ESTATÍSTICA ---
def analisar_risco_ativo(ticker: str) -> str:
    """
    Usa a biblioteca stats.py para calcular Volatilidade, Sharpe e Drawdown.
    O Gemini deve usar esta função sempre que o usuário perguntar sobre risco ou volatilidade.
    """
    try:
        dados = yf.download(ticker, period="1y", auto_adjust=True, progress=False)
        if isinstance(dados.columns, pd.MultiIndex):
            dados.columns = dados.columns.get_level_values(0)
        precos = dados['Close']
        
        retornos = stats.calcular_retornos_log(precos)
        metricas = stats.calcular_metricas_risco(retornos)
        dd = stats.calcular_drawdown(precos)
        
        return f"""
        Métricas de Risco ({ticker}):
        - Volatilidade Anual: {metricas['volatilidade']:.2%}
        - Índice de Sharpe: {metricas['sharpe']:.2f}
        - Max Drawdown: {dd:.2%}
        """
    except Exception as e:
        return f"Erro ao calcular risco: {e}"

# --- 4. CONFIGURANDO A INTELIGÊNCIA ---
genai.configure(api_key=api_key)

# Agora a IA tem Machine Learning E Estatística!
minhas_ferramentas = [
    ml_tendencia.treinar_e_prever,
    ml_tendencia.carregar_e_prever,
    ml_tendencia.listar_modelos_disponiveis,
    analisar_risco_ativo 
]

modelo = genai.GenerativeModel(
    model_name='gemini-2.5-flash',
    tools=minhas_ferramentas
)

# --- 5. LAYOUT PRINCIPAL DO TERMINAL ---
col_chat, col_resultados = st.columns([1, 1.5])

# === COLUNA 1: CHAT ===
with col_chat:
    st.subheader("💬 Terminal de Chat")
    
    if "mensagens" not in st.session_state:
        st.session_state.mensagens = []
        st.session_state.chat_gemini = modelo.start_chat(enable_automatic_function_calling=True)

    chat_container = st.container(height=500)
    with chat_container:
        for msg in st.session_state.mensagens:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    prompt = st.chat_input("Ex: Qual a volatilidade de MXRF11.SA?")

    if prompt:
        st.session_state.mensagens.append({"role": "user", "content": prompt})
        with chat_container:
             with st.chat_message("user"):
                st.markdown(prompt)

        with chat_container:
            with st.chat_message("assistant"):
                with st.spinner("Processando dados e consultando modelos..."):
                    try:
                        resposta = st.session_state.chat_gemini.send_message(prompt)
                        st.markdown("✅ Análise concluída! Gráficos atualizados ao lado.")
                        st.session_state.mensagens.append({"role": "assistant", "content": resposta.text})
                        st.rerun()
                    except Exception as e:
                        st.error(f"Ocorreu um erro técnico: {e}")

# === COLUNA 2: DASHBOARD GRÁFICO ===
with col_resultados:
    st.subheader("📊 Painel de Resultados")
    
    # Filtra as mensagens do usuário para descobrir qual ativo ele quer ver
    mensagens_usuario = [msg["content"] for msg in st.session_state.mensagens if msg["role"] == "user"]
    
    if mensagens_usuario:
        ultimo_pedido = mensagens_usuario[-1].upper()
        # Procura um padrão de Ticker da B3 (ex: PETR4.SA, MXRF11.SA)
        padrao_ticker = re.findall(r'[A-Z]{4}\d{1,2}\.SA', ultimo_pedido)
        
        if padrao_ticker:
            ticker_atual = padrao_ticker[0]
            st.success(f"Visualizando dados de: **{ticker_atual}**")
            
            # Baixa os dados silenciosamente para desenhar o gráfico
            df_grafico = yf.download(ticker_atual, period="1y", progress=False)
            
            if not df_grafico.empty:
                if isinstance(df_grafico.columns, pd.MultiIndex):
                    df_grafico.columns = df_grafico.columns.get_level_values(0)
                
                # Desenha um gráfico de linha lindão do preço
                st.markdown("#### Histórico de Preços (1 Ano)")
                st.line_chart(df_grafico['Close'])
                
                # E um gráfico de barras para o volume de negociação
                st.markdown("#### Volume de Negociação")
                st.bar_chart(df_grafico['Volume'])
                
        # Esconde o textão da IA dentro de uma caixa expansível para deixar a tela limpa
        tem_resposta = any(msg["role"] == "assistant" for msg in st.session_state.mensagens)
        if tem_resposta:
            ultima_resposta_ia = [msg["content"] for msg in st.session_state.mensagens if msg["role"] == "assistant"][-1]
            with st.expander("Ver Relatório em Texto Escrito pela IA", expanded=False):
                st.markdown(ultima_resposta_ia)
    else:
        st.info("Aguardando comando... \n\nDigite um ticker como 'MXRF11.SA' no chat para gerar os gráficos interativos.")