import streamlit as st
from groq import Groq
from anthropic import Anthropic
import requests
import xml.etree.ElementTree as ET
import json
from bs4 import BeautifulSoup  # Biblioteca para ler a SciELO
import sqlite3
from datetime import datetime


# ==========================================
# CONFIGURAÇÃO DO BANCO DE DADOS (SQLite)
# ==========================================
def init_db():
    conn = sqlite3.connect('historico_saude.db', check_same_thread=False)
    c = conn.cursor()
    # Tabela para as sessões de chat
    c.execute('''CREATE TABLE IF NOT EXISTS sessoes 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, titulo TEXT, data_criacao TEXT)''')
    # Tabela para as mensagens de cada sessão
    c.execute('''CREATE TABLE IF NOT EXISTS mensagens 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, sessao_id INTEGER, role TEXT, content TEXT)''')
    conn.commit()
    return conn


conn = init_db()

# ==========================================
# CONFIGURAÇÃO DA PÁGINA
# ==========================================
st.set_page_config(page_title="Assistente Clínico", page_icon="🩺", layout="centered")

st.title("🩺 Assistente Clínico")

# ==========================================
# BARRA LATERAL: HISTÓRICO E CONFIGURAÇÕES
# ==========================================
with st.sidebar:
    st.header("🔑 Configurações")

    # Tenta carregar as chaves ocultas do Streamlit Secrets
    try:
        groq_api_key = st.secrets["GROQ_API_KEY"]
        claude_api_key = st.secrets["CLAUDE_API_KEY"]
        st.success("🟢 Sistema Conectado e Pronto")
    except KeyError:
        # Se não encontrar as chaves (ex: rodando localmente sem configurar), mostra os campos
        st.warning("⚠️ Chaves não configuradas no servidor.")
        groq_api_key = st.text_input("Groq API Key", type="password")
        claude_api_key = st.text_input("Anthropic (Claude) API Key", type="password")

    st.divider()

    # --- SEÇÃO DE HISTÓRICO ---
    st.header("📜 Histórico de Conversas")

    if st.button("➕ Nova Conversa", use_container_width=True):
        st.session_state.sessao_id = None
        st.session_state.messages = []
        st.rerun()

    # Listar conversas anteriores do banco de dados
    cursor = conn.cursor()
    cursor.execute("SELECT id, titulo FROM sessoes ORDER BY id DESC")
    conversas = cursor.fetchall()

    for id_conversas, titulo in conversas:
        if st.button(f"💬 {titulo[:25]}...", key=f"hist_{id_conversas}", use_container_width=True):
            st.session_state.sessao_id = id_conversas
            st.rerun()

    st.divider()

    # --- PARÂMETROS ---
    st.markdown("**Parâmetros de Busca (Federada)**")
    max_artigos = st.slider("Artigos por base (PubMed e SciELO)", 1, 10, 4)
    st.caption(f"Total analisado: até {max_artigos * 2} artigos por pergunta.")

# ==========================================
# LÓGICA DE CARREGAMENTO DE MENSAGENS DO DB
# ==========================================
if "sessao_id" not in st.session_state:
    st.session_state.sessao_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# Se uma sessão foi selecionada na barra lateral, carrega as mensagens dela
if st.session_state.sessao_id is not None:
    cursor = conn.cursor()
    cursor.execute("SELECT role, content FROM mensagens WHERE sessao_id = ? ORDER BY id ASC",
                   (st.session_state.sessao_id,))
    st.session_state.messages = [{"role": r, "content": c} for r, c in cursor.fetchall()]
elif not st.session_state.messages:
    # Se não tem sessão e a lista tá vazia, garante que a tela fique limpa
    st.session_state.messages = []


# ==========================================
# FUNÇÕES DO MOTOR (BACKEND)
# ==========================================
def extrair_termos_federados(pergunta, api_key):
    """Usa o Groq para gerar buscas separadas (Inglês e Português) em formato JSON."""
    client = Groq(api_key=api_key)
    prompt = f"""
    Você é um bibliotecário médico especialista. 
    Analise a dúvida clínica do usuário e crie duas strings de busca otimizadas:
    1. Para o PubMed (em INGLÊS, usando termos MeSH unidos por AND/OR).
    2. Para a SciELO (em PORTUGUÊS, usando descritores DeCS unidos por AND/OR).

    Retorne EXCLUSIVAMENTE um objeto JSON válido no formato exato abaixo, sem explicações:
    {{
        "busca_pubmed": "termos em ingles aqui",
        "busca_scielo": "termos em portugues aqui"
    }}

    Dúvida do Usuário: "{pergunta}"
    """
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"erro": f"Erro no Groq: {e}"}


def buscar_pubmed(termo_busca, max_resultados):
    """Busca abstracts reais no NCBI (PubMed)."""
    base_url_search = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params_search = {"db": "pubmed", "term": termo_busca, "retmax": max_resultados, "retmode": "json"}

    try:
        res = requests.get(base_url_search, params=params_search).json()
        ids = res.get("esearchresult", {}).get("idlist", [])

        if not ids: return ""

        base_url_fetch = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params_fetch = {"db": "pubmed", "id": ",".join(ids), "rettype": "abstract", "retmode": "xml"}
        fetch_res = requests.get(base_url_fetch, params=params_fetch)

        root = ET.fromstring(fetch_res.content)
        contexto = []

        for art in root.findall(".//PubmedArticle"):
            title_element = art.find(".//ArticleTitle")
            title = title_element.text if title_element is not None else "Sem título"
            abs_text = "".join([t.text for t in art.findall(".//AbstractText") if t.text])
            if abs_text:
                contexto.append(f"[Fonte: PubMed] TÍTULO: {title}\nRESUMO: {abs_text}")

        return "\n\n".join(contexto)
    except Exception as e:
        return f"Erro no PubMed: {e}"


def buscar_scielo(termo_busca, max_resultados):
    """Extrai abstracts reais da SciELO via Web Scraping."""
    url = f"https://search.scielo.org/?q={termo_busca}&count={max_resultados}"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')

        artigos = soup.find_all('div', class_='item')
        contexto = []

        for art in artigos[:max_resultados]:
            titulo_tag = art.find('strong', class_='title')
            titulo = titulo_tag.text.strip() if titulo_tag else "Sem título"

            abstract_div = art.find('div', class_='user-abstract') or art.find('div', class_='abstract')
            abs_text = abstract_div.text.strip() if abstract_div else ""

            if abs_text:
                contexto.append(f"[Fonte: SciELO] TÍTULO: {titulo}\nRESUMO: {abs_text}")

        return "\n\n".join(contexto)
    except Exception as e:
        return f"Erro na SciELO: {e}"


def sintese_clinica_claude(pergunta, contexto_global, api_key):
    """Usa o Claude para fundir as evidências globais (PubMed) e regionais (SciELO)."""
    client = Anthropic(api_key=api_key)
    system_prompt = """
    Você é um assistente médico acadêmico de excelência. Sua função é responder à pergunta do usuário baseando-se EXCLUSIVA E ESTRITAMENTE nos dados fornecidos (Abstracts do PubMed e da SciELO).

    REGRAS ABSOLUTAS:
    1. Não utilize nenhum conhecimento prévio ou externo. Se a resposta não estiver nos textos, diga que não há evidências suficientes.
    2. Não faça presunções, não invente tratamentos e evite achismos.
    3. Se houver divergência entre as diretrizes globais (PubMed) e regionais (SciELO), exponha essas diferenças clinicamente.
    4. NÃO cite os nomes dos artigos no meio do texto e não use expressões como "De acordo com o artigo...". Construa um texto fluido, direto e unificado.
    5. NÃO liste as referências ou fontes no final da resposta, a não ser que o usuário tenha pedido isso expressamente.
    6. Responda em português claro e profissional.
    """

    prompt_usuario = f"Evidências Científicas Recuperadas:\n{contexto_global}\n\nPergunta do Usuário: {pergunta}"

    try:
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1500,
            temperature=0.2,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt_usuario}]
        )
        return message.content[0].text
    except Exception as e:
        return f"Erro no Claude: {e}"


# ==========================================
# INTERFACE DE CHAT (UI)
# ==========================================
# Renderiza histórico atual da tela
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input do usuário
if prompt := st.chat_input("Sua dúvida clínica..."):

    # Valida chaves
    if not groq_api_key or not claude_api_key:
        st.warning("⚠️ Por favor, insira as chaves de API na barra lateral para continuar.")
        st.stop()

    # --- SALVANDO A SESSÃO E A MENSAGEM DO USUÁRIO NO BANCO ---
    cursor = conn.cursor()

    if st.session_state.sessao_id is None:
        # Cria uma nova sessão usando a data e a primeira pergunta como título
        data_atual = datetime.now().strftime("%d/%m/%Y %H:%M")
        cursor.execute("INSERT INTO sessoes (titulo, data_criacao) VALUES (?, ?)", (prompt, data_atual))
        st.session_state.sessao_id = cursor.lastrowid
        conn.commit()

    # Adiciona na tela e no banco a pergunta do usuário
    st.session_state.messages.append({"role": "user", "content": prompt})
    cursor.execute("INSERT INTO mensagens (sessao_id, role, content) VALUES (?, ?, ?)",
                   (st.session_state.sessao_id, "user", prompt))
    conn.commit()

    with st.chat_message("user"):
        st.markdown(prompt)

    # Processamento em pipeline limpo e federado
    with st.chat_message("assistant"):
        resposta_final = ""

        with st.spinner("Buscando evidências (PubMed e SciELO)..."):
            termos_json = extrair_termos_federados(prompt, groq_api_key)

            if "erro" in termos_json:
                resposta_final = termos_json["erro"]
            else:
                contexto_pubmed = buscar_pubmed(termos_json.get("busca_pubmed", ""), max_artigos)
                contexto_scielo = buscar_scielo(termos_json.get("busca_scielo", ""), max_artigos)

                contexto_global = f"{contexto_pubmed}\n\n{contexto_scielo}".strip()

                if not contexto_global:
                    # ==========================================
                    # FALLBACK: MODO LIVRO-TEXTO (ANATOMIA/FISIO)
                    # ==========================================
                    fallback_prompt = """
                    Você é um professor titular de medicina em uma universidade de excelência. O usuário fez uma pergunta sobre ciências básicas da saúde (Anatomia, Fisiologia, Histologia, Patologia, Farmacologia, etc.) onde artigos científicos não são a fonte principal de estudo.

                    Sua missão é dar uma "aula" estruturada, didática e de alto rigor acadêmico, baseando-se EXCLUSIVAMENTE na literatura médica clássica e consolidada.

                    REGRAS ABSOLUTAS:
                    1. AVISO INICIAL: Você DEVE iniciar a sua resposta EXATAMENTE com o seguinte texto em citação:
                    "> ⚠️ **Modo Ciências Básicas:** Não há ensaios clínicos aplicáveis a esta dúvida estrutural. A resposta abaixo foi gerada com base na literatura acadêmica clássica."

                    2. ESTRUTURA VISUAL: Organize a resposta de forma altamente escaneável:
                       - Use títulos (###) para dividir os tópicos da pergunta.
                       - Use **negrito** exclusivamente para destacar termos técnicos, anatômicos ou fisiológicos essenciais.
                       - Use tópicos (bullet points) para listar características, funções ou camadas. Evite parágrafos muito longos.

                    3. DIDÁTICA: Vá direto ao ponto acadêmico. Descreva as estruturas, vias e funções com precisão cirúrgica de livro-texto.

                    4. BIBLIOGRAFIA DE REFERÊNCIA: Ao final da resposta, adicione uma seção chamada "📚 **Literatura Sugerida**" e liste de 2 a 3 livros-texto clássicos reais onde o usuário pode aprofundar aquele tema específico (ex: Moore ou Gray para Anatomia; Guyton ou Silverthorn para Fisiologia; Robbins para Patologia Básica).

                    5. SEGURANÇA: Não invente vias ou estruturas. Se o tema for ambíguo, atenha-se à base estrutural inquestionável.
                    """

                    client_claude = Anthropic(api_key=claude_api_key)
                    try:
                        msg = client_claude.messages.create(
                            model="claude-sonnet-4-6",
                            max_tokens=2000,
                            temperature=0.3,  # Mantido baixo para garantir foco estrutural
                            system=fallback_prompt,
                            messages=[{"role": "user", "content": prompt}]
                        )
                        resposta_final = msg.content[0].text
                    except Exception as e:
                        resposta_final = f"Erro no Claude (Modo Básico): {e}"
                else:
                    resposta_final = sintese_clinica_claude(prompt, contexto_global, claude_api_key)

        # Exibe a resposta final na interface
        st.markdown(resposta_final)

        # --- SALVANDO A RESPOSTA DA IA NO BANCO ---
        st.session_state.messages.append({"role": "assistant", "content": resposta_final})
        cursor.execute("INSERT INTO mensagens (sessao_id, role, content) VALUES (?, ?, ?)",
                       (st.session_state.sessao_id, "assistant", resposta_final))
        conn.commit()