import streamlit as st
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import asyncio
import time
import json
import os
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

# Charger les variables d'environnement
load_dotenv(dotenv_path=".env")

# Configuration des chemins de sauvegarde
HISTORY_DIR = "conversation_history"
if not os.path.exists(HISTORY_DIR):
    os.makedirs(HISTORY_DIR)

# Initialiser l'historique de conversation
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Bonjour ! 👋 Je suis l'Agent IA d'Open Data Gouv CI. Comment puis-je vous aider aujourd'hui ?",
        }
    ]

# Initialiser les états de session nécessaires
if "generate_summary" not in st.session_state:
    st.session_state.generate_summary = False

if "conversation_loaded" not in st.session_state:
    st.session_state.conversation_loaded = False

# =================================================================
# FONCTIONS UTILITAIRES (définies avant leur utilisation)
# =================================================================


def save_conversation(messages):
    """Sauvegarde la conversation dans un fichier JSON"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"conversation_{timestamp}.json"
    filepath = os.path.join(HISTORY_DIR, filename)

    conversation_data = {
        "timestamp": datetime.now().isoformat(),
        "messages": messages,
        "message_count": len([msg for msg in messages if msg["role"] == "user"]),
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(conversation_data, f, ensure_ascii=False, indent=2)
    return filename


def get_saved_conversations():
    """Récupère la liste des conversations sauvegardées"""
    if not os.path.exists(HISTORY_DIR):
        return []

    files = []
    for filename in os.listdir(HISTORY_DIR):
        if filename.endswith(".json"):
            filepath = os.path.join(HISTORY_DIR, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    timestamp = data.get("timestamp", "")
                    message_count = data.get("message_count", 0)
                    display_name = (
                        f"{filename.replace('.json', '')} ({message_count} messages)"
                    )
                    files.append((display_name, filename))
            except:
                continue

    # Trier par date (plus récent en premier)
    files.sort(reverse=True)
    return [display_name for display_name, _ in files]


def load_conversation(filename):
    """Charge une conversation depuis un fichier"""
    # Extraire le vrai nom de fichier
    real_filename = filename.split(" (")[0] + ".json"
    filepath = os.path.join(HISTORY_DIR, real_filename)

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data["messages"]
    except Exception as e:
        st.error(f"Erreur lors du chargement: {str(e)}")
        return None


def handle_conversation_selection():
    """Gère le chargement de conversation sélectionnée"""
    if (
        "selected_conversation" in st.session_state
        and st.session_state.selected_conversation
    ):
        if (
            st.session_state.selected_conversation != ""
            and not st.session_state.conversation_loaded
        ):
            messages = load_conversation(st.session_state.selected_conversation)
            if messages:
                st.session_state.messages = messages
                st.session_state.conversation_loaded = True
                st.rerun()


def reset_conversation_selection():
    """Remet à zéro la sélection de conversation"""
    if "selected_conversation" in st.session_state:
        # Ne pas modifier directement, utiliser une approche différente
        st.session_state.conversation_loaded = False


def export_to_markdown(messages):
    """Exporte la conversation en format Markdown"""
    markdown_content = f"# Conversation OpenDataGouv CI\n\n"
    markdown_content += (
        f"**Date d'export:** {datetime.now().strftime('%d/%m/%Y à %H:%M:%S')}\n\n"
    )
    markdown_content += "---\n\n"

    for i, message in enumerate(messages):
        if message["role"] == "user":
            markdown_content += f"## 👤 Utilisateur\n\n{message['content']}\n\n"
        else:
            markdown_content += f"## 🤖 Assistant IA\n\n{message['content']}\n\n"

        if i < len(messages) - 1:
            markdown_content += "---\n\n"

    return markdown_content


def format_response_with_markdown(response):
    """Formate la réponse avec un meilleur rendu Markdown"""
    # Détecter et formater les tableaux
    if "|" in response and response.count("|") >= 6:  # Probable tableau
        lines = response.split("\n")
        formatted_lines = []

        for line in lines:
            if "|" in line and line.count("|") >= 2:
                formatted_lines.append(line)
            else:
                formatted_lines.append(line)

        response = "\n".join(formatted_lines)

    # Améliorer les listes
    lines = response.split("\n")
    formatted_lines = []

    for line in lines:
        if line.strip().startswith("- "):
            formatted_lines.append(line)
        elif line.strip().startswith("* "):
            formatted_lines.append(line.replace("* ", "- "))
        elif any(line.strip().startswith(f"{i}. ") for i in range(1, 20)):
            formatted_lines.append(line)
        else:
            formatted_lines.append(line)

    return "\n".join(formatted_lines)


def create_chart_from_data(data_text):
    """Crée un graphique à partir de données textuelles si possible"""
    try:
        lines = data_text.split("\n")
        table_lines = [line for line in lines if "|" in line and line.count("|") >= 2]

        if len(table_lines) >= 2:
            headers = [col.strip() for col in table_lines[0].split("|")[1:-1]]
            table_data = []

            for line in table_lines[2:]:
                if line.strip() and "|" in line:
                    row = [col.strip() for col in line.split("|")[1:-1]]
                    if len(row) == len(headers):
                        table_data.append(row)

            if table_data:
                df = pd.DataFrame(table_data, columns=headers)
                numeric_cols = []

                for col in df.columns:
                    try:
                        pd.to_numeric(df[col])
                        numeric_cols.append(col)
                    except:
                        continue

                if len(numeric_cols) >= 1:
                    fig = px.bar(df, x=df.columns[0], y=numeric_cols[0])
                    return fig
    except:
        pass
    return None


async def generate_conversation_summary(messages):
    """Génère un résumé de la conversation en cours"""
    conversation_messages = [
        msg
        for msg in messages
        if not (
            msg["role"] == "assistant"
            and (
                "Bonjour ! 👋 Je suis l'Agent IA" in msg["content"]
                or "📝 **Résumé de notre conversation :**" in msg["content"]
            )
        )
    ]

    if len(conversation_messages) < 2:
        return "Pas assez de messages pour générer un résumé."

    conversation_text = ""
    for msg in conversation_messages:
        role = "Utilisateur" if msg["role"] == "user" else "Assistant"
        conversation_text += f"{role}: {msg['content']}\n\n"

    summary_prompt = f"""
    Veuillez générer un résumé concis de cette conversation sur les données ouvertes de Côte d'Ivoire.
    Le résumé doit inclure :
    - Les principales questions posées par l'utilisateur
    - Les sujets de données abordés
    - Les informations clés fournies
    - Les actions ou analyses effectuées

    Conversation:
    {conversation_text}

    Résumé:
    """

    try:
        summary_messages = [{"role": "user", "content": summary_prompt}]
        response = await agent.ainvoke({"messages": summary_messages})
        return response["messages"][-1].content
    except Exception as e:
        return f"Erreur lors de la génération du résumé: {str(e)}"


async def run_agent_with_memory(messages):
    """Exécute l'agent avec mémoire de conversation"""
    formatted_messages = [
        {"role": msg["role"], "content": msg["content"]} for msg in messages
    ]
    response = await agent.ainvoke({"messages": formatted_messages})
    return response["messages"][-1].content


# =================================================================
# INITIALISATION DE L'INTERFACE STREAMLIT
# =================================================================

st.set_page_config(
    page_title="OpenDataGouv CI - Agentic AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Style CSS personnalisé
st.markdown(
    """
    <style>
        .stChatInput {position: fixed; bottom: 20px; width: 70%;}
        .stChatMessage {padding: 12px; border-radius: 12px;}
        .user-message {background-color: #f0f2f6;}
        .assistant-message {background-color: #e6f7ff;}
        .stSpinner > div {text-align: center;}
        .stMarkdown {font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;}
        .title-text {color: #1a5276; text-align: center;}
        .sidebar .sidebar-content {background-color: #f8f9fa;}
        .summary-box {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }
        .summary-title {
            color: #495057;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .markdown-content {
            line-height: 1.6;
        }
        .markdown-content table {
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
        }
        .markdown-content th, .markdown-content td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        .markdown-content th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        .markdown-content tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .markdown-content blockquote {
            border-left: 4px solid #007bff;
            margin: 15px 0;
            padding: 10px 20px;
            background-color: #f8f9fa;
        }
        .markdown-content code {
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: monospace;
        }
        .markdown-content pre {
            background-color: #f4f4f4;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }
        .save-indicator {
            color: #28a745;
            font-size: 12px;
            margin-top: 5px;
        }
    </style>
""",
    unsafe_allow_html=True,
)

# =================================================================
# INITIALISATION DE L'AGENT
# =================================================================


@st.cache_resource
def initialize_agent():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def _initialize():
        client = MultiServerMCPClient(
            {
                "local_mcp": {
                    "url": "http://127.0.0.1:8000/mcp",
                    "transport": "streamable_http",
                }
            }
        )
        tools = await client.get_tools()
        return create_react_agent("openai:gpt-4o-mini", tools)

    return loop.run_until_complete(_initialize())


try:
    agent = initialize_agent()
except Exception as e:
    st.error(f"Erreur lors de l'initialisation de l'agent: {str(e)}")
    st.stop()

# =================================================================
# SIDEBAR
# =================================================================

with st.sidebar:
    st.image(
        "https://data.gouv.ci/api/v1/portals/yCWsyaGpA/assets/logo?draft=false&hash=fit-290f57f7b3.png",
        width=150,
    )
    st.markdown("### Agent IA OpenDataGouv")
    st.markdown(
        """
    Posez vos questions sur les données ouvertes de la Côte d'Ivoire.
    L'agent peut vous aider à:
    - Trouver des jeux de données
    - Expliquer des indicateurs
    - Générer des visualisations
    """
    )
    st.markdown("---")

    # Section de sauvegarde/chargement
    st.markdown("### 💾 Historique")

    if "messages" in st.session_state and len(st.session_state.messages) > 1:
        if st.button(
            "Sauvegarder la conversation", use_container_width=True, type="primary"
        ):
            filename = save_conversation(st.session_state.messages)
            st.success(f"Conversation sauvegardée : {filename}")
            time.sleep(1)
            st.rerun()

    saved_files = get_saved_conversations()
    if saved_files:
        # Utiliser une clé différente pour éviter les conflits
        selected = st.selectbox(
            "Charger une conversation",
            options=[""] + saved_files,
            key="conversation_selector",
            index=0,
        )

        # Gérer la sélection avec un bouton séparé
        if selected and selected != "":
            if st.button("Charger cette conversation", use_container_width=True):
                messages = load_conversation(selected)
                if messages:
                    st.session_state.messages = messages
                    st.success("Conversation chargée avec succès!")
                    time.sleep(1)
                    st.rerun()

    if "messages" in st.session_state and len(st.session_state.messages) > 1:
        if st.button("Exporter en Markdown", use_container_width=True):
            markdown_content = export_to_markdown(st.session_state.messages)
            st.download_button(
                label="Télécharger le fichier Markdown",
                data=markdown_content,
                file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
            )

    st.markdown("---")

    conversation_length = (
        len([msg for msg in st.session_state.messages if msg["role"] == "user"])
        if "messages" in st.session_state
        else 0
    )
    st.markdown(f"**Messages échangés:** {conversation_length}")

    if conversation_length >= 3:
        st.markdown("### 📝 Résumé de conversation")
        if st.button("Générer un résumé", use_container_width=True, type="secondary"):
            st.session_state.generate_summary = True
            st.rerun()

    st.markdown("---")

    if st.button("Effacer la conversation", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Bonjour ! 👋 Je suis l'Agent IA d'Open Data Gouv CI. Comment puis-je vous aider aujourd'hui ?",
            }
        ]
        st.session_state.generate_summary = False
        st.session_state.conversation_loaded = False
        # Forcer la réinitialisation des widgets
        if "conversation_selector" in st.session_state:
            del st.session_state["conversation_selector"]
        st.rerun()

    st.markdown("**Version:** 1.0.0")
    st.markdown("**Développer par Data354**")

# =================================================================
# GESTION DES MESSAGES ET DE L'INTERFACE
# =================================================================

# Gérer la génération du résumé
if st.session_state.get("generate_summary", False):
    with st.spinner("Génération du résumé en cours..."):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            summary = loop.run_until_complete(
                generate_conversation_summary(st.session_state.messages)
            )
            summary_message = {
                "role": "assistant",
                "content": f"📝 **Résumé de notre conversation :**\n\n{summary}",
            }
            st.session_state.messages.append(summary_message)
        except Exception as e:
            error_message = {
                "role": "assistant",
                "content": f"❌ Erreur lors de la génération du résumé: {str(e)}",
            }
            st.session_state.messages.append(error_message)
        finally:
            loop.close()
            st.session_state.generate_summary = False
            st.rerun()

# Afficher l'historique des messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        formatted_content = format_response_with_markdown(message["content"])
        st.markdown(formatted_content, unsafe_allow_html=True)

        if message["role"] == "assistant":
            chart = create_chart_from_data(message["content"])
            if chart:
                st.plotly_chart(chart, use_container_width=True)

# Gestion des entrées utilisateur
if prompt := st.chat_input("Posez votre question ici..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        with st.spinner("Reflexion..."):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                response = loop.run_until_complete(
                    run_agent_with_memory(st.session_state.messages)
                )
                response = format_response_with_markdown(response)

                for chunk in response.split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    message_placeholder.markdown(
                        full_response + "▌", unsafe_allow_html=True
                    )

                message_placeholder.markdown(full_response, unsafe_allow_html=True)

                chart = create_chart_from_data(full_response)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)

                if "merci" in full_response.lower():
                    full_response += " 😊"
                elif "données" in full_response.lower():
                    full_response += " 📊"

            except Exception as e:
                full_response = f"⚠️ Désolé, une erreur est survenue: {str(e)}"
                message_placeholder.markdown(full_response, unsafe_allow_html=True)
            finally:
                loop.close()

    st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Auto-sauvegarde toutes les 5 réponses
    user_messages = len(
        [msg for msg in st.session_state.messages if msg["role"] == "user"]
    )
    if user_messages % 5 == 0 and user_messages > 0:
        try:
            save_conversation(st.session_state.messages)
            st.sidebar.markdown(
                '<div class="save-indicator">💾 Conversation auto-sauvegardée</div>',
                unsafe_allow_html=True,
            )
        except:
            pass
