import streamlit as st
from groq import Groq
from anthropic import Anthropic
import cohere
from cohere.errors import CohereAPIError
import requests
import xml.etree.ElementTree as ET
import json
from bs4 import BeautifulSoup
import sqlite3
from datetime import datetime


# ==========================================
# CONFIGURAÇÃO DO BANCO DE DADOS (SQLite)
# ==========================================
def init_db():
    conn = sqlite3.connect('historico_saude.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS sessoes 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, titulo TEXT, data_criacao TEXT)''')
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

    try:
        groq_api_key = st.secrets["GROQ_API_KEY"]
        claude_api_key = st.secrets["CLAUDE_API_KEY"]
        cohere_api_key = st.secrets["COHERE_API_KEY"]
        springer_api_key = st.secrets.get("SPRINGER_API_KEY", "")
        elsevier_api_key = st.secrets.get("ELSEVIER_API_KEY", "")
        st.success("🟢 Sistemas Conectados e Prontos")
    except KeyError:
        st.warning("⚠️ Chaves principais não configuradas.")
        groq_api_key = st.text_input("Groq API Key", type="password")
        claude_api_key = st.text_input("Anthropic API Key", type="password")
        cohere_api_key = st.text_input("Cohere API Key (Reranker)", type="password")
        springer_api_key = ""
        elsevier_api_key = ""

    st.divider()
    st.header("📜 Histórico de Conversas")

    if st.button("➕ Nova Conversa", use_container_width=True):
        st.session_state.sessao_id = None
        st.session_state.messages = []
        st.rerun()

    cursor = conn.cursor()
    cursor.execute("SELECT id, titulo FROM sessoes ORDER BY id DESC")
    for id_conversas, titulo in cursor.fetchall():
        if st.button(f"💬 {titulo[:25]}...", key=f"hist_{id_conversas}", use_container_width=True):
            st.session_state.sessao_id = id_conversas
            st.rerun()

    st.divider()
    st.markdown("**Parâmetros de Busca (Federada)**")
    max_artigos = st.slider("Captura inicial por base", 1, 20, 10)
    st.caption(f"O Reranker extrairá o Filé Mignon de até {max_artigos * 4} artigos combinados.")

# ==========================================
# LÓGICA DE CARREGAMENTO DE MENSAGENS DO DB
# ==========================================
if "sessao_id" not in st.session_state: st.session_state.sessao_id = None
if "messages" not in st.session_state: st.session_state.messages = []

if st.session_state.sessao_id is not None:
    cursor = conn.cursor()
    cursor.execute("SELECT role, content FROM mensagens WHERE sessao_id = ? ORDER BY id ASC",
                   (st.session_state.sessao_id,))
    st.session_state.messages = [{"role": r, "content": c} for r, c in cursor.fetchall()]


# ==========================================
# FUNÇÕES DO MOTOR (BACKEND) - ADAPTADAS PARA LISTAS
# ==========================================
def extrair_termos_federados(pergunta, api_key):
    client = Groq(api_key=api_key)
    prompt = f"""
    Você é um bibliotecário médico especialista. 
    Analise a dúvida clínica do usuário e crie duas strings de busca otimizadas:
    1. Busca Global (em INGLÊS, usando termos MeSH unidos por AND/OR).
    2. Busca Regional (em PORTUGUÊS, usando descritores DeCS unidos por AND/OR).

    Retorne EXCLUSIVAMENTE um objeto JSON válido no formato exato abaixo, sem explicações:
    {{
        "busca_ingles": "termos em ingles aqui",
        "busca_portugues": "termos em portugues aqui"
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
    base_url_search = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params_search = {"db": "pubmed", "term": termo_busca, "retmax": max_resultados, "retmode": "json"}
    try:
        res = requests.get(base_url_search, params=params_search).json()
        ids = res.get("esearchresult", {}).get("idlist", [])
        if not ids: return []
        base_url_fetch = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params_fetch = {"db": "pubmed", "id": ",".join(ids), "rettype": "abstract", "retmode": "xml"}
        fetch_res = requests.get(base_url_fetch, params=params_fetch)
        root = ET.fromstring(fetch_res.content)
        artigos = []
        for art in root.findall(".//PubmedArticle"):
            title = art.find(".//ArticleTitle").text if art.find(".//ArticleTitle") is not None else "Sem título"
            abs_text = "".join([t.text for t in art.findall(".//AbstractText") if t.text])
            if abs_text: artigos.append({"fonte": "PubMed", "titulo": title, "texto": abs_text})
        return artigos
    except Exception:
        return []


def buscar_scielo(termo_busca, max_resultados):
    url = f"https://search.scielo.org/?q={termo_busca}&count={max_resultados}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        itens = soup.find_all('div', class_='item')
        artigos = []
        for art in itens[:max_resultados]:
            titulo = art.find('strong', class_='title').text.strip() if art.find('strong',
                                                                                 class_='title') else "Sem título"
            abstract_div = art.find('div', class_='user-abstract') or art.find('div', class_='abstract')
            abs_text = abstract_div.text.strip() if abstract_div else ""
            if abs_text: artigos.append({"fonte": "SciELO", "titulo": titulo, "texto": abs_text})
        return artigos
    except Exception:
        return []


def buscar_springer(termo_busca, max_resultados, api_key):
    if not api_key: return []
    url = "http://api.springernature.com/meta/v2/json"
    params = {"q": termo_busca, "api_key": api_key, "p": max_resultados}
    try:
        response = requests.get(url, params=params).json()
        artigos = []
        for art in response.get('records', []):
            titulo = art.get('title', 'Sem título')
            abs_text = art.get('abstract', '')
            if abs_text: artigos.append({"fonte": "Springer Nature", "titulo": titulo, "texto": abs_text})
        return artigos
    except Exception:
        return []


def buscar_elsevier(termo_busca, max_resultados, api_key):
    if not api_key: return []
    url = "https://api.elsevier.com/content/search/scopus"
    headers = {"X-ELS-APIKey": api_key, "Accept": "application/json"}
    params = {"query": f"TITLE-ABS-KEY({termo_busca})", "count": max_resultados}
    try:
        response = requests.get(url, headers=headers, params=params).json()
        artigos = []
        for art in response.get('search-results', {}).get('entry', []):
            titulo = art.get('dc:title', 'Sem título')
            abs_text = art.get('dc:description', '')
            if abs_text: artigos.append({"fonte": "Elsevier Scopus", "titulo": titulo, "texto": abs_text})
        return artigos
    except Exception:
        return []


# ==========================================
# MOTOR DE RERANKING COM TRATAMENTO DE ERROS
# ==========================================
def rerank_evidencias(pergunta, lista_documentos, api_key):
    if not lista_documentos: return []
    if not api_key or len(api_key) < 10: return lista_documentos[:5]

    co = cohere.Client(api_key)
    docs_texto = [f"{d['titulo']}: {d['texto']}" for d in lista_documentos]

    try:
        results = co.rerank(
            query=pergunta,
            documents=docs_texto,
            top_n=5,
            model='rerank-multilingual-v3.0'
        )
        filtrados = [lista_documentos[r.index] for r in results.results]
        return filtrados
    except CohereAPIError as e:
        st.warning(f"⚠️ Reranker indisponível ({e.status_code}). Acionando ordenação padrão das bases.")
        return lista_documentos[:5]
    except Exception:
        st.warning("⚠️ Instabilidade na rede do Reranker. Acionando fallback seguro.")
        return lista_documentos[:5]


# ==========================================
# SÍNTESE COM PROMPT CACHING (ANTHROPIC)
# ==========================================
def sintese_clinica_final(pergunta, evidencias_filtradas, api_key):
    client = Anthropic(api_key=api_key)
    contexto_str = "\n\n".join([f"[{d['fonte']}] {d['titulo']}: {d['texto']}" for d in evidencias_filtradas])

    system_prompt = """
    Você é um assistente médico acadêmico de excelência. Sua função é responder à pergunta do usuário baseando-se EXCLUSIVA E ESTRITAMENTE nos dados fornecidos.

    REGRAS ABSOLUTAS:
    1. Não utilize nenhum conhecimento prévio ou externo. Se a resposta não estiver nos textos, diga que não há evidências suficientes.
    2. Não faça presunções, não invente tratamentos e evite achismos.
    3. Se houver divergência entre as bases globais (PubMed, Springer, Elsevier) e regionais (SciELO), exponha essas diferenças clinicamente.
    4. NÃO cite os nomes dos artigos ou bases no meio do texto. Construa um texto fluido, direto e unificado.
    5. NÃO liste as referências ou fontes no final da resposta, a não ser que o usuário tenha pedido expressamente.
    6. Responda em português claro e profissional.
    """

    try:
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            temperature=0.2,
            system=[
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"}
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Evidências Científicas Recuperadas:\n{contexto_str}",
                            "cache_control": {"type": "ephemeral"}
                        },
                        {
                            "type": "text",
                            "text": f"Pergunta do Usuário: {pergunta}"
                        }
                    ]
                }
            ]
        )
        return message.content[0].text
    except Exception as e:
        return f"Erro na síntese: {e}"


# ==========================================
# INTERFACE DE CHAT E FLUXO PRINCIPAL
# ==========================================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Sua dúvida clínica..."):
    if not groq_api_key or not claude_api_key:
        st.warning("⚠️ Por favor, insira as chaves de API principais para continuar.")
        st.stop()

    cursor = conn.cursor()
    if st.session_state.sessao_id is None:
        data_atual = datetime.now().strftime("%d/%m/%Y %H:%M")
        cursor.execute("INSERT INTO sessoes (titulo, data_criacao) VALUES (?, ?)", (prompt, data_atual))
        st.session_state.sessao_id = cursor.lastrowid
        conn.commit()

    st.session_state.messages.append({"role": "user", "content": prompt})
    cursor.execute("INSERT INTO mensagens (sessao_id, role, content) VALUES (?, ?, ?)",
                   (st.session_state.sessao_id, "user", prompt))
    conn.commit()

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        resposta_final = ""

        with st.status("Minerando bases de evidência globais...", expanded=True) as status:
            st.write("🧠 Otimizando termos clínicos...")
            termos_json = extrair_termos_federados(prompt, groq_api_key)

            if "erro" in termos_json:
                resposta_final = termos_json["erro"]
                status.update(label="Falha na otimização de busca.", state="error")
            else:
                termo_en = termos_json.get("busca_ingles", "")
                termo_pt = termos_json.get("busca_portugues", "")

                st.write("🔍 Extraindo dados: PubMed e SciELO...")
                all_docs = []
                all_docs.extend(buscar_pubmed(termo_en, max_artigos))
                all_docs.extend(buscar_scielo(termo_pt, max_artigos))

                st.write("📚 Extraindo dados: Springer Nature e Elsevier...")
                all_docs.extend(buscar_springer(termo_en, max_artigos, springer_api_key))
                all_docs.extend(buscar_elsevier(termo_en, max_artigos, elsevier_api_key))

                if not all_docs:
                    st.write("⚠️ Nenhuma evidência nova. Acionando literatura clássica...")

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

                    4. BIBLIOGRAFIA DE REFERÊNCIA: Ao final da resposta, adicione uma seção chamada "📚 **Literatura Sugerida**" e liste de 2 a 3 livros-texto clássicos reais onde o usuário pode aprofundar aquele tema específico.

                    5. SEGURANÇA: Não invente vias ou estruturas. Se o tema for ambíguo, atenha-se à base estrutural inquestionável.
                    """

                    client_claude = Anthropic(api_key=claude_api_key)
                    try:
                        msg = client_claude.messages.create(
                            model="claude-sonnet-4-6",
                            max_tokens=4096,
                            temperature=0.3,
                            system=fallback_prompt,
                            messages=[{"role": "user", "content": prompt}]
                        )
                        resposta_final = msg.content[0].text
                        status.update(label="Síntese acadêmica concluída!", state="complete")
                    except Exception as e:
                        resposta_final = f"Erro no Claude (Modo Básico): {e}"
                        status.update(label="Falha na síntese.", state="error")
                else:
                    st.write(f"⚖️ Filtrando relevância máxima de {len(all_docs)} artigos combinados...")
                    docs_selecionados = rerank_evidencias(prompt, all_docs, cohere_api_key)

                    st.write("🔬 Assistente sintetizando os achados em Cache...")
                    resposta_final = sintese_clinica_final(prompt, docs_selecionados, claude_api_key)
                    status.update(label="Síntese clínica concluída com sucesso!", state="complete")

        st.markdown(resposta_final)

        st.session_state.messages.append({"role": "assistant", "content": resposta_final})
        cursor.execute("INSERT INTO mensagens (sessao_id, role, content) VALUES (?, ?, ?)",
                       (st.session_state.sessao_id, "assistant", resposta_final))
        conn.commit()