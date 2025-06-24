import streamlit as st
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import asyncio
import time
import pandas as pd
import plotly.express as px
import re
import markdown
from markdown.extensions import tables

# Load environment variables
load_dotenv(dotenv_path=".env")

# Initialize conversation history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Bonjour ! 👋 Je suis l'Agent IA d'Open Data Gouv CI. Comment puis-je vous aider aujourd'hui ?",
        }
    ]

# Initialize session states
if "generate_summary" not in st.session_state:
    st.session_state.generate_summary = False
if "stop_generation" not in st.session_state:
    st.session_state.stop_generation = False
if "is_generating" not in st.session_state:
    st.session_state.is_generating = False

# =================================================================
# UTILITY FUNCTIONS FOR TABLE RENDERING (FINAL REVISION)
# =================================================================


def _parse_line_into_cells(line):
    """
    Robustly parses a potential table line into cells, preserving empty cells.
    Handles lines that may or may not start/end with '|'.
    """
    # Remove leading/trailing whitespace, then split by '|'.
    # Carefully handle potential empty parts due to multiple || or leading/trailing |

    # First, strip whitespace
    clean_line = line.strip()

    # Remove markdown code block delimiters if they are present in the line
    clean_line = re.sub(r"```markdown\s*", "", clean_line, flags=re.IGNORECASE)

    # If the line starts and ends with a single '|', remove them to simplify splitting
    if clean_line.startswith("|") and clean_line.endswith("|"):
        clean_line = clean_line[1:-1]

    # Split by '|' and strip whitespace from each part
    cells = [cell.strip() for cell in clean_line.split("|")]
    return cells


def reconstruct_markdown_table(headers, data_rows):
    """
    Reconstructs a perfect Markdown table string from parsed headers and data.
    """
    if not headers:
        return ""

    num_cols = len(headers)
    table_lines = []

    # Add header row
    table_lines.append("| " + " | ".join(headers) + " |")

    # Add separator row
    table_lines.append("|" + "---|" * num_cols)

    # Add data rows, ensuring consistent column count and preserving empty cells
    for row in data_rows:
        row_padded = list(row)  # Make a copy to pad
        if len(row_padded) < num_cols:
            row_padded.extend([""] * (num_cols - len(row_padded)))
        elif len(row_padded) > num_cols:
            row_padded = row_padded[:num_cols]

        table_lines.append("| " + " | ".join(row_padded) + " |")

    return "\n".join(table_lines)


def _split_embedded_table_line(single_line_text):
    """
    Attempts to split a single string containing an embedded, malformed Markdown table
    into proper, multi-line Markdown table structure (header, separator, data rows).
    This function specifically targets the agent's common single-line output.
    """
    # Clean potential ```markdown prefix
    cleaned_text = re.sub(
        r"```markdown\s*", "", single_line_text, flags=re.IGNORECASE
    ).strip()

    # Look for the introductory text followed by a pipe
    # e.g., "Voici la liste des outils disponibles: | Nom de l'outil | Description |..."
    intro_text_match = re.match(r"^(.*?)\s*\|(.+)$", cleaned_text)

    if not intro_text_match:
        return [
            single_line_text
        ]  # No pipe-delimited content found, return original line

    intro_text = intro_text_match.group(1).strip()
    table_flat_content = intro_text_match.group(2).strip()

    output_lines = []
    if intro_text:
        output_lines.append(intro_text)

    # Now, parse the flat table content.
    # We are looking for: Header Cells | Separator | Data Cells

    # The agent's output tends to be: Header_A | Header_B |---|---| Data_1A | Data_1B | Data_2A | Data_2B |
    # We need to find where the header ends and the separator begins.

    # Find the separator pattern (e.g., "|---|---|" or "|--|--|")
    # This regex is robust for the separator line within the flat string.
    # It finds a sequence of | followed by dashes, repeated for columns.
    separator_idx_match = re.search(
        r"(\|\s*[-]+\s*\|(?:[\s-]*\|[\s-]*\|)*)", table_flat_content
    )

    if not separator_idx_match:
        # Fallback: if no separator is found, try to infer header/data just by content.
        # This is less reliable but might catch very simple cases.
        # Given agent behavior, this path is less likely to yield correct tables.
        return [single_line_text]

    separator_full_string = separator_idx_match.group(0)  # e.g., "|---|---|"

    header_flat_content = table_flat_content[: separator_idx_match.start()].strip()
    data_flat_content = table_flat_content[separator_idx_match.end() :].strip()

    # Parse headers
    headers = _parse_line_into_cells(header_flat_content)
    headers = [
        h for h in headers if h.strip()
    ]  # Filter out any truly empty header parts

    if not headers:
        return [single_line_text]  # If no valid headers, cannot form a table

    num_cols = len(headers)

    # Add header line
    output_lines.append("| " + " | ".join(headers) + " |")
    # Add separator line
    output_lines.append("|" + "---|" * num_cols)

    # Parse and add data rows
    # Split the data content by '|' to get individual cell contents
    raw_data_cells = _parse_line_into_cells(data_flat_content)

    # Filter out empty strings that might result from extra pipes at start/end or consecutive pipes
    raw_data_cells = [cell for cell in raw_data_cells if cell.strip() != ""]

    # Group raw_data_cells into rows based on num_cols
    current_row_cells = []
    for cell in raw_data_cells:
        current_row_cells.append(cell)
        if len(current_row_cells) == num_cols:
            output_lines.append("| " + " | ".join(current_row_cells) + " |")
            current_row_cells = []

    # Handle any remaining cells for a partial last row
    if current_row_cells:
        current_row_cells.extend([""] * (num_cols - len(current_row_cells)))
        output_lines.append("| " + " | ".join(current_row_cells) + " |")

    return output_lines


def clean_and_reconstruct_tables_in_text(text):
    """
    Scans the entire text, processes single-line embedded tables first,
    then identifies potential multi-line table blocks, and reconstructs
    them into perfectly formatted Markdown tables.
    """
    initial_lines = text.split("\n")
    processed_lines_from_embedded = []

    # First, try to fix embedded single-line tables
    for line in initial_lines:
        fixed_lines = _split_embedded_table_line(line)
        processed_lines_from_embedded.extend(fixed_lines)

    # Now, process these potentially multi-line formatted outputs
    lines = processed_lines_from_embedded
    cleaned_parts = []

    current_table_lines_raw = []
    in_potential_table_block = False

    for i, line in enumerate(lines):
        # A robust check for a line that might be part of a markdown table.
        # It should contain pipes and not be a pure separator line (e.g., "|---|---|" )
        is_potential_data_row = (
            "|" in line
            and len(_parse_line_into_cells(line))
            >= 1  # Must have at least 1 inferred column
            and not re.fullmatch(r"^\s*\|[\s-]*\|[\s-]*\|[\s-]*\|\s*$", line.strip())
        )  # Not a pure separator line

        # Check for separator line pattern (like "---|---") explicitly
        is_separator_line = re.fullmatch(
            r"^\s*\|[\s-]*\|[\s-]*\|[\s-]*\|\s*$", line.strip()
        )

        if is_potential_data_row or is_separator_line:
            current_table_lines_raw.append(line)
            in_potential_table_block = True
        else:
            if in_potential_table_block and current_table_lines_raw:
                # Process the accumulated raw table lines
                headers = None
                data_rows = []

                # Find the actual header line (first non-separator, non-empty line)
                temp_header_line = None
                header_index = -1
                for idx, r_line in enumerate(current_table_lines_raw):
                    if not re.fullmatch(
                        r"^\s*\|[\s-]*\|[\s-]*\|[\s-]*\|\s*$", r_line.strip()
                    ) and any(c.isalnum() for c in r_line):
                        temp_header_line = r_line
                        header_index = idx
                        break

                if temp_header_line:
                    headers = _parse_line_into_cells(temp_header_line)
                    headers = [
                        h for h in headers if h.strip()
                    ]  # Keep only non-empty header parts

                if not headers:  # If no valid headers, treat as non-table block
                    cleaned_parts.extend(current_table_lines_raw)
                else:
                    # Parse data rows
                    for j, raw_line in enumerate(current_table_lines_raw):
                        if (
                            j == header_index
                        ):  # Skip the actual header line as it's processed
                            continue

                        parsed_cells = _parse_line_into_cells(raw_line)
                        # Filter out purely separator lines if any remain
                        if not all(
                            re.fullmatch(r"-+", cell.replace(" ", ""))
                            for cell in parsed_cells
                        ):
                            # Preserve empty rows, but ensure they match header column count
                            if len(parsed_cells) != len(headers):
                                if len(parsed_cells) < len(headers):
                                    parsed_cells.extend(
                                        [""] * (len(headers) - len(parsed_cells))
                                    )
                                else:
                                    parsed_cells = parsed_cells[: len(headers)]
                            data_rows.append(parsed_cells)

                    # Reconstruct and add to cleaned parts
                    if data_rows:  # Only reconstruct if there's actual data
                        cleaned_parts.append(
                            reconstruct_markdown_table(headers, data_rows)
                        )
                    else:  # If only header was found but no data, just add original lines
                        # (This case should be rare if _split_embedded_table_line works well)
                        cleaned_parts.extend(current_table_lines_raw)

                current_table_lines_raw = []  # Reset
                in_potential_table_block = False

            cleaned_parts.append(line)  # Add non-table line

    # Process any remaining table block at the end of the text
    if in_potential_table_block and current_table_lines_raw:
        headers = None
        data_rows = []

        temp_header_line = None
        header_index = -1
        for idx, r_line in enumerate(current_table_lines_raw):
            if not re.fullmatch(
                r"^\s*\|[\s-]*\|[\s-]*\|[\s-]*\|\s*$", r_line.strip()
            ) and any(c.isalnum() for c in r_line):
                temp_header_line = r_line
                header_index = idx
                break

        if temp_header_line:
            headers = _parse_line_into_cells(temp_header_line)
            headers = [h for h in headers if h.strip()]

        if headers:
            for j, raw_line in enumerate(current_table_lines_raw):
                if j == header_index:
                    continue
                parsed_cells = _parse_line_into_cells(raw_line)
                if not all(
                    re.fullmatch(r"-+", cell.replace(" ", "")) for cell in parsed_cells
                ):
                    if len(parsed_cells) != len(headers):
                        if len(parsed_cells) < len(headers):
                            parsed_cells.extend(
                                [""] * (len(headers) - len(parsed_cells))
                            )
                        else:
                            parsed_cells = parsed_cells[: len(headers)]
                    data_rows.append(parsed_cells)

            if data_rows:
                cleaned_parts.append(reconstruct_markdown_table(headers, data_rows))
            else:
                cleaned_parts.extend(current_table_lines_raw)
        else:
            cleaned_parts.extend(current_table_lines_raw)

    return "\n".join(cleaned_parts)


def render_message_content(content):
    """
    Renders message content, applying robust table formatting before Markdown conversion.
    """
    # Step 1: Clean and reconstruct any table-like structures
    formatted_content = clean_and_reconstruct_tables_in_text(content)

    # Step 2: Use Markdown to convert to HTML, especially for tables
    md = markdown.Markdown(extensions=[tables.makeExtension(use_align_attribute=True)])
    html_content = md.convert(formatted_content)

    # Step 3: Render in Streamlit
    if "<table>" in html_content:
        st.markdown(html_content, unsafe_allow_html=True)
    else:
        # If no table was detected or generated, render as plain Markdown
        st.markdown(formatted_content)


def extract_table_data(text):
    """
    Extracts table data from text, assuming clean Markdown table format after `clean_and_reconstruct_tables_in_text`.
    Used for chart creation.
    """
    # Ensure text is pre-processed before extraction
    pre_processed_text = clean_and_reconstruct_tables_in_text(text)

    lines = pre_processed_text.split("\n")
    headers = None
    table_rows = []

    for line in lines:
        stripped_line = line.strip()
        # Ensure it's a markdown table row (starts/ends with |) and not a separator
        # Also ensure it has some content beyond just pipes and spaces
        if (
            stripped_line.startswith("|")
            and stripped_line.endswith("|")
            and not re.fullmatch(r"\|[\s-]*\|[\s-]*\|", stripped_line)
            and any(c.isalnum() for c in stripped_line)
        ):
            parts = [p.strip() for p in stripped_line.strip("|").split("|")]
            # Don't filter out empty parts here, preserve cell count for df
            if parts:  # Only consider non-empty lines that truly represent rows
                if headers is None:
                    headers = parts
                else:
                    table_rows.append(parts)

    return headers, table_rows


def create_chart_from_data(data_text):
    """Create a chart from textual data if possible"""
    try:
        # Use the already cleaned text
        pre_cleaned_text = clean_and_reconstruct_tables_in_text(data_text)
        headers, table_rows = extract_table_data(pre_cleaned_text)

        if headers and table_rows and len(headers) >= 2:
            # Create DataFrame
            aligned_table_rows = []
            for row in table_rows:
                if len(row) < len(headers):
                    aligned_table_rows.append(row + [""] * (len(headers) - len(row)))
                elif len(row) > len(headers):
                    aligned_table_rows.append(row[: len(headers)])
                else:
                    aligned_table_rows.append(row)

            df = pd.DataFrame(aligned_table_rows, columns=headers)

            # Try to convert numeric columns
            numeric_cols = []
            for col in df.columns:
                if col == df.columns[0]:
                    continue  # Skip first column, often categorical label

                try:
                    # Clean numeric data: remove commas, spaces, and thin spaces (\u202f)
                    cleaned_col_data = (
                        df[col]
                        .astype(str)
                        .str.replace(",", "")
                        .str.replace(" ", "")
                        .str.replace("\u202f", "")
                    )
                    temp_series = pd.to_numeric(cleaned_col_data, errors="coerce")

                    if (
                        temp_series.notna().any()
                    ):  # Only add if at least one numeric value exists
                        df[col] = temp_series
                        numeric_cols.append(col)
                except Exception:
                    # Not a numeric column, continue
                    pass

            if len(numeric_cols) >= 1:
                # Filter out rows where the primary numeric column is NaN
                df_filtered = df.dropna(subset=[numeric_cols[0]])

                if not df_filtered.empty:
                    # Use the first numeric column for the y-axis
                    fig = px.bar(
                        df_filtered,
                        x=df_filtered.columns[0],
                        y=numeric_cols[0],
                        title=f"Graphique: {numeric_cols[0]} par {df_filtered.columns[0]}",
                        labels={
                            df_filtered.columns[0]: df_filtered.columns[0],
                            numeric_cols[0]: numeric_cols[0],
                        },
                    )
                    fig.update_layout(showlegend=False)
                    return fig
    except Exception as e:
        print(f"Error creating chart from data: {e}")
    return None


async def generate_conversation_summary(messages):
    """Generate a summary of the current conversation"""
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
        return "Not enough messages to generate a summary."

    conversation_text = ""
    for msg in conversation_messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        conversation_text += f"{role}: {msg['content']}\n\n"

    summary_prompt = f"""
    Please generate a concise summary of this conversation about Côte d'Ivoire open data.
    The summary should include:
    - Main questions asked by the user
    - Data topics discussed
    - Key information provided
    - Actions or analyses performed

    Conversation:
    {conversation_text}

    Summary:
    """

    try:
        summary_messages = [{"role": "user", "content": summary_prompt}]
        response = await agent.ainvoke({"messages": summary_messages})
        return response["messages"][-1].content
    except Exception as e:
        return f"Error generating summary: {str(e)}"


def enhance_prompt_for_tables(prompt):
    """
    This function now serves more as a placeholder. The primary instructions
    for table formatting are in the system instruction.
    """
    return prompt


async def run_agent_with_memory(messages):
    """Run agent with conversation memory and enhanced table instructions"""
    formatted_messages = [
        {"role": msg["role"], "content": msg["content"]} for msg in messages
    ]

    # Enhanced system instruction for better table generation
    system_instruction = """
    Tu es l'Agent IA d'Open Data Gouv CI. Ton objectif est de fournir des informations précises et structurées sur les données ouvertes dont tu as accès de la Côte d'Ivoire.

    **RÈGLES CRITIQUES POUR LES TABLEAUX :**
    Lorsque tu dois présenter des listes d'outils, de datasets, de fichiers, ou toute autre information qui peut être organisée en colonnes, tu **DOIS TOUJOURS** utiliser le format de tableau Markdown STRICT.

    **POINTS ESSENTIELS À RESPECTER IMPÉRATIVEMENT :**
    1.  **Réponse Complète et Délimitée :** Si la réponse contient un tableau, le tableau DOIT être la partie principale et claire de la réponse. Place le texte introductif (ex: "Voici la liste des outils disponibles :") AVANT le bloc du tableau, et assure-toi qu'il y a un double saut de ligne (`\\n\\n`) entre le texte et le début du tableau.
    2.  **Bloc de tableau autonome :** Le tableau Markdown doit être un bloc de texte propre et autonome. NE JAMAIS entrelacer des phrases, des mots ou des caractères supplémentaires entre les lignes du tableau lui-même.
    3.  **Lignes séparées :** Chaque ligne du tableau (en-têtes, ligne de séparation, lignes de données) DOIT être sur une ligne distincte, terminée par un saut de ligne (`\\n`).
    4.  **Délimiteurs de ligne :** Chaque ligne du tableau (en-têtes et données) DOIT commencer et se terminer par un pipe (`|`).
    5.  **Ligne de séparation des en-têtes :** Une seule ligne de séparation (`|---|---|...|`) est OBLIGATOIRE, juste après les en-têtes. Elle doit avoir le même nombre de sections que les en-têtes.
    6.  **Consistance des colonnes :** Chaque ligne de données DOIT avoir le même nombre de colonnes que l'en-tête. Utilise des cellules vides (` ` ou `""`) si des données sont manquantes pour une cellule spécifique, mais NE LAISSE PAS une ligne entière de tableau vide si elle n'a pas de sens (comme `| | |`).
    7.  **Contenu des cellules :** **NE JAMAIS** utiliser les caractères `|` ou `---` à l'intérieur du texte des cellules, car cela briserait le format du tableau.
    8.  **Espacement :** Utilise des espaces clairs autour des pipes dans les cellules (ex: `| Cellule 1 | Cellule 2 |`).
    9.  **Pas de ```markdown :** NE PAS inclure les délimiteurs de bloc de code Markdown (```markdown) dans ta réponse, car le système de rendu les ajoute déjà si nécessaire. Réponds directement avec le texte Markdown du tableau.

    **TRÈS IMPORTANT :** Pour une demande comme "liste moi les outils", ta réponse devrait ressembler à ceci :
    "Voici la liste des outils disponibles :\n\n| Nom de l'outil | Description |\n|---|---|\n| lister_datasets_disponibles | Liste tous les datasets disponibles sur le serveur. |\n| lister_fichiers | Permet de lister les fichiers d'un dataset spécifique. |\n| obtenir_values_agg | Récupère des informations agrégées basées sur les valeurs d'une colonne. |"
    Assure-toi que les sauts de ligne sont présents entre le texte introductif et le tableau, et entre chaque ligne du tableau.
    """

    # Add system instruction at the beginning
    if formatted_messages and formatted_messages[0]["role"] == "system":
        formatted_messages[0]["content"] = system_instruction
    else:
        formatted_messages.insert(0, {"role": "system", "content": system_instruction})

    response = await agent.ainvoke({"messages": formatted_messages})
    return response["messages"][-1].content


# =================================================================
# STREAMLIT INTERFACE INITIALIZATION
# =================================================================

st.set_page_config(
    page_title="OpenDataGouv CI - Agentic AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better table styling
st.markdown(
    """
    <style>
        .stChatMessage {
            padding: 12px; 
            border-radius: 12px;
        }
        
        /* CSS amélioré pour les tableaux avec meilleur contraste */
        .stMarkdown table {
            border-collapse: collapse !important;
            width: 100% !important;
            margin: 15px 0 !important;
            border: 2px solid #dee2e6 !important;
            font-size: 14px !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
            background-color: white !important;
        }
        
        .stMarkdown th {
            background: linear-gradient(135deg, #6c757d 0%, #495057 100%) !important;
            border: 1px solid #495057 !important;
            padding: 15px 20px !important;
            text-align: left !important;
            font-weight: 700 !important;
            color: white !important;
            font-size: 15px !important;
            text-transform: uppercase !important;
            letter-spacing: 0.5px !important;
        }
        
        .stMarkdown td {
            border: 1px solid #dee2e6 !important;
            padding: 12px 20px !important;
            text-align: left !important;
            color: #212529 !important;
            background-color: white !important;
            font-weight: 500 !important;
        }
        
        .stMarkdown tr:nth-child(even) td {
            background-color: #f8f9fa !important;
        }
        
        .stMarkdown tr:hover td {
            background-color: #e3f2fd !important;
            transform: scale(1.01) !important;
            transition: all 0.2s ease !important;
        }
        
        /* Style spécial pour les lignes avec fond sombre */
        .stMarkdown tbody tr:nth-child(odd) {
            background-color: white !important;
        }
        
        .stMarkdown tbody tr:nth-child(even) {
            background-color: #f8f9fa !important;
        }
        
        /* Assurer la lisibilité du texte */
        .stMarkdown tbody td {
            color: #212529 !important;
            font-weight: 500 !important;
        }
        
        /* Style pour le bouton d'arrêt */
        .stop-button {
            background-color: #dc3545 !important;
            color: white !important;
            border: none !important;
            padding: 8px 16px !important;
            border-radius: 4px !important;
            cursor: pointer !important;
            transition: background-color 0.2s !important;
        }
        
        .stop-button:hover {
            background-color: #c82333 !important;
        }
        
        /* Styles pour assurer la compatibilité */
        div[data-testid="stMarkdownContainer"] table {
            border-collapse: collapse !important;
            width: 100% !important;
            background-color: white !important;
        }
        
        div[data-testid="stMarkdownContainer"] th {
            background: linear-gradient(135deg, #6c757d 0%, #495057 100%) !important;
            color: white !important;
            font-weight: bold !important;
            padding: 15px 20px !important;
            border: 1px solid #495057 !important;
        }
        
        div[data-testid="stMarkdownContainer"] td {
            background-color: white !important;
            color: #212529 !important;
            padding: 12px 20px !important;
            border: 1px solid #dee2e6 !important;
        }
        
        div[data-testid="stMarkdownContainer"] tr:nth-child(even) td {
            background-color: #f8f9fa !important;
        }
        
        div[data-testid="stMarkdownContainer"] tr:hover td {
            background-color: #e3f2fd !important;
        }
        
        /* Force le contraste pour tous les éléments de tableau */
        table * {
            color: inherit !important;
        }
        
        /* Style spécifique pour le mode sombre de Streamlit */
        .stApp[data-theme="dark"] .stMarkdown table {
            background-color: #1e1e1e !important;
            border: 2px solid #404040 !important;
        }
        
        .stApp[data-theme="dark"] .stMarkdown th {
            background: linear-gradient(135deg, #404040 0%, #2d2d2d 100%) !important;
            color: #ffffff !important;
            border: 1px solid #2d2d2d !important;
        }
        
        .stApp[data-theme="dark"] .stMarkdown td {
            background-color: #1e1e1e !important;
            color: #ffffff !important;
            border: 1px solid #404040 !important;
        }
        
        .stApp[data-theme="dark"] .stMarkdown tr:nth-child(even) td {
            background-color: #2d2d2d !important;
        }
        
        .stApp[data-theme="dark"] .stMarkdown tr:hover td {
            background-color: #404040 !important;
        }
    </style>
""",
    unsafe_allow_html=True,
)

# =================================================================
# AGENT INITIALIZATION
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
        return create_react_agent("openai:gpt-4o", tools)

    return loop.run_until_complete(_initialize())


try:
    agent = initialize_agent()
except Exception as e:
    st.error(f"Error initializing agent: {str(e)}")
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
    L'agent peut vous aider à :
    - Trouver des jeux de données
    - Expliquer des indicateurs
    - Analyser des données (si les outils sont disponibles)
    """
    )
    st.markdown("---")

    # Filter out initial greeting and summary messages for actual conversation length
    conversation_length = (
        len(
            [
                msg
                for msg in st.session_state.messages
                if msg["role"] == "user"
                or (
                    msg["role"] == "assistant"
                    and "Bonjour ! 👋 Je suis l'Agent IA" not in msg["content"]
                    and "📝 **Résumé de notre conversation :**" not in msg["content"]
                )
            ]
        )
        // 2
    )

    st.markdown(f"**Messages échangés :** {conversation_length}")

    if conversation_length >= 2:
        st.markdown("### 📝 Résumé de la conversation")
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
        st.session_state.stop_generation = False
        st.session_state.is_generating = False
        st.rerun()

    st.markdown("**Version :** 1.0.6")  # Updated version
    st.markdown("**Développé par Data354**")

# =================================================================
# MESSAGE HANDLING AND INTERFACE
# =================================================================

# Handle summary generation
if st.session_state.get("generate_summary", False):
    with st.spinner("Génération du résumé..."):
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
                "content": f"❌ Erreur lors de la génération du résumé : {str(e)}",
            }
            st.session_state.messages.append(error_message)
        finally:
            loop.close()
            st.session_state.generate_summary = False
            st.rerun()

# Display message history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        render_message_content(message["content"])

        # Create chart if possible, only for assistant messages
        if message["role"] == "assistant":
            chart = create_chart_from_data(message["content"])
            if chart:
                st.plotly_chart(chart, use_container_width=True)

# Handle user input
if prompt := st.chat_input(
    "Posez votre question ici...",
    disabled=st.session_state.is_generating,
):
    enhanced_prompt = enhance_prompt_for_tables(prompt)

    st.session_state.messages.append({"role": "user", "content": enhanced_prompt})
    st.session_state.stop_generation = False
    st.session_state.is_generating = True

    with st.chat_message("user"):
        st.markdown(prompt)

    st.rerun()

# Check if we need to generate a response
if st.session_state.is_generating and not st.session_state.stop_generation:
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        stop_placeholder = st.empty()

        # Show stop button
        if stop_placeholder.button(
            "⏹️ Arrêter la génération", key="stop_btn", type="secondary"
        ):
            st.session_state.stop_generation = True
            st.session_state.is_generating = False
            st.rerun()

        full_response = ""

        with st.spinner("Réflexion en cours..."):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                response = loop.run_until_complete(
                    run_agent_with_memory(st.session_state.messages)
                )

                # Typing animation with stop capability
                words = response.split()
                for i, word in enumerate(words):
                    if st.session_state.stop_generation:
                        full_response += f"\n\n*[Génération arrêtée par l'utilisateur après {i+1}/{len(words)} mots]*"
                        break

                    full_response += word + " "
                    time.sleep(0.02)

                    # Update message continuously
                    message_placeholder.markdown(full_response + "▌")

                # Final display with proper table formatting
                message_placeholder.empty()
                render_message_content(full_response)
                stop_placeholder.empty()

                # Create chart if possible
                if not st.session_state.stop_generation:
                    chart = create_chart_from_data(full_response)
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)

            except Exception as e:
                full_response = f"⚠️ Désolé, une erreur est survenue : {str(e)}"
                message_placeholder.markdown(full_response)
                stop_placeholder.empty()
            finally:
                loop.close()
                st.session_state.is_generating = False

        # Add final response to history
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )
        st.rerun()

# Show warning if generation was stopped (after rerunning)
if st.session_state.stop_generation:
    st.session_state.stop_generation = False
