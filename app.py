import streamlit as st
from openai import OpenAI
import json
import ml_tendencia 
import stats  
import yfinance as yf
import pandas as pd
import os
import re

# --- 1. CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Terminal CF Tech | Agente Quant",
    page_icon="📈",
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- 2. BARRA LATERAL ---
with st.sidebar:
    st.markdown("<h1 style='text-align: center; font-size: 40px;'>MCP do Ceará Finance</h1>", unsafe_allow_html=True)
    
    caminho_logo = "Logo Ceará Finance.png"
    if os.path.exists(caminho_logo):
        st.image(caminho_logo, use_container_width=True)
    
    st.markdown("---")
    st.header("⚙️ Configuração DeepSeek")
    api_key = st.text_input("Cole sua Chave API do DeepSeek:", type="password")
    
    st.markdown("---")
    st.markdown("Desenvolvido por **CF Tech** (2026.1)")

if not api_key:
    st.warning("👈 Por favor, insira sua chave da API do DeepSeek na barra lateral para começar.")
    st.stop()

# --- 3. FERRAMENTA DE ESTATÍSTICA ---
def analisar_risco_ativo(ticker: str) -> str:
    try:
        dados = yf.download(ticker, period="1y", auto_adjust=True, progress=False)
        if isinstance(dados.columns, pd.MultiIndex):
            dados.columns = dados.columns.get_level_values(0)
        precos = dados['Close']
        
        retornos = stats.calcular_retornos_log(precos)
        metricas = stats.calcular_metricas_risco(retornos)
        dd = stats.calcular_drawdown(precos)
        
        return f"Métricas ({ticker}): Volatilidade: {metricas['volatilidade']:.2%}, Sharpe: {metricas['sharpe']:.2f}, Max Drawdown: {dd:.2%}"
    except Exception as e:
        return f"Erro ao calcular risco: {e}"

# --- 4. CONFIGURANDO O MOTOR DEEPSEEK ---
# Usamos a biblioteca da OpenAI, mas apontamos para a base de dados do DeepSeek
cliente_ia = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

# Mapeamento manual das ferramentas (O "Cardápio" para o DeepSeek)
minhas_ferramentas_deepseek = [
    {
        "type": "function",
        "function": {
            "name": "treinar_e_prever",
            "description": "Treina um modelo preditivo de Machine Learning para a tendência de um ativo financeiro (ações ou FIIs).",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "O código do ativo na B3 com o sufixo .SA (ex: PETR4.SA, MXRF11.SA)"}
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analisar_risco",
            "description": "Calcula a volatilidade, índice de sharpe e drawdown histórico de um ativo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "O código do ativo na B3 (ex: PETR4.SA)"}
                },
                "required": ["ticker"]
            }
        }
    }
]

def executar_ferramenta(nome_ferramenta, argumentos_json):
    """Lê o pedido da IA e roda o código Python real"""
    args = json.loads(argumentos_json)
    if nome_ferramenta == "treinar_e_prever":
        return ml_tendencia.treinar_e_prever(args.get("ticker"))
    elif nome_ferramenta == "analisar_risco":
        return analisar_risco_ativo(args.get("ticker"))
    return "Ferramenta não encontrada."

# --- 5. LAYOUT PRINCIPAL DO TERMINAL ---
col_chat, col_resultados = st.columns([1, 1.5])

with col_chat:
    st.subheader("💬 Terminal de Chat (DeepSeek)")
    
    if "mensagens" not in st.session_state:
        # A primeira mensagem é o "System Prompt" que ensina ele a agir
        st.session_state.mensagens = [{"role": "system", "content": "Você é um Agente Quant. Sempre que pedirem previsão ou risco, use suas ferramentas disponíveis."}]

    chat_container = st.container(height=500)
    with chat_container:
        for msg in st.session_state.mensagens:
            if msg["role"] != "system" and msg["role"] != "tool":
                with st.chat_message(msg["role"]):
                    st.markdown(msg.get("content", ""))

    prompt = st.chat_input("Ex: Faça uma análise de PETR4.SA e calcule o risco.")

    if prompt:
        st.session_state.mensagens.append({"role": "user", "content": prompt})
        with chat_container:
             with st.chat_message("user"):
                st.markdown(prompt)

        with chat_container:
            with st.chat_message("assistant"):
                with st.spinner("Conectando ao motor DeepSeek..."):
                    try:
                        # Passo 1: Envia a pergunta e as ferramentas para o DeepSeek
                        resposta = cliente_ia.chat.completions.create(
                            model="deepseek-chat",
                            messages=st.session_state.mensagens,
                            tools=minhas_ferramentas_deepseek,
                            temperature=0.1
                        )
                        
                        mensagem_ia = resposta.choices[0].message
                        
                        # Passo 2: O DeepSeek quis usar alguma ferramenta?
                        if mensagem_ia.tool_calls:
                            # Guarda o pedido da IA no histórico
                            st.session_state.mensagens.append(mensagem_ia)
                            
                            # Executa todas as ferramentas pedidas
                            for chamada in mensagem_ia.tool_calls:
                                st.write(f"⚙️ Executando script: `{chamada.function.name}`...")
                                resultado_real = executar_ferramenta(chamada.function.name, chamada.function.arguments)
                                
                                # Devolve o resultado matemático invisível para a IA
                                st.session_state.mensagens.append({
                                    "role": "tool",
                                    "tool_call_id": chamada.id,
                                    "content": str(resultado_real)
                                })
                            
                            # Passo 3: Pede para a IA gerar o texto final baseado na matemática
                            resposta_final = cliente_ia.chat.completions.create(
                                model="deepseek-chat",
                                messages=st.session_state.mensagens
                            )
                            texto_final = resposta_final.choices[0].message.content
                            st.session_state.mensagens.append({"role": "assistant", "content": texto_final})
                            st.markdown(texto_final)
                            
                        else:
                            # Se não precisou de ferramenta, só responde normal
                            st.session_state.mensagens.append({"role": "assistant", "content": mensagem_ia.content})
                            st.markdown(mensagem_ia.content)
                            
                    except Exception as e:
                        st.error(f"Ocorreu um erro técnico: {e}")
        
        st.rerun()

# === COLUNA 2: DASHBOARD GRÁFICO ===
with col_resultados:
    st.subheader("📊 Painel de Resultados")
    
    mensagens_usuario = [msg["content"] for msg in st.session_state.mensagens if msg["role"] == "user"]
    if mensagens_usuario:
        ultimo_pedido = mensagens_usuario[-1].upper()
        padrao_ticker = re.findall(r'[A-Z]{4}\d{1,2}\.SA', ultimo_pedido)
        
        if padrao_ticker:
            ticker_atual = padrao_ticker[0]
            st.success(f"Visualizando dados de: **{ticker_atual}**")
            df_grafico = yf.download(ticker_atual, period="1y", progress=False)
            
            if not df_grafico.empty:
                if isinstance(df_grafico.columns, pd.MultiIndex):
                    df_grafico.columns = df_grafico.columns.get_level_values(0)
                st.markdown("#### Histórico de Preços (1 Ano)")
                st.line_chart(df_grafico['Close'])
                
        tem_resposta = any(msg["role"] == "assistant" for msg in st.session_state.mensagens)
        if tem_resposta:
            ultima_resposta_ia = [msg.get("content", "") for msg in st.session_state.mensagens if msg["role"] == "assistant"][-1]
            with st.expander("Ver Relatório em Texto Escrito pela IA", expanded=False):
                st.markdown(ultima_resposta_ia)
    else:
        st.info("Aguardando comando... Digite um ticker como 'MXRF11.SA' para gerar os gráficos.")
