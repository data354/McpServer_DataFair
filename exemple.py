import streamlit as st
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
import asyncio

# Initialisation de l'agent
@st.cache_resource
def init_agent():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    async def _init():
        client = MultiServerMCPClient({
            "mcp_server": {
                "url": "https://mcp-server-626474317752.europe-west1.run.app/mcp/",
                "transport": "streamable_http"
            }
        })
        tools = await client.get_tools()
        return create_react_agent("openai:gpt-4o", tools)
    
    return loop.run_until_complete(_init())

# Interface simplifiée
st.title("🤖 Agent MCP Minimaliste")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Posez votre question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("assistant"):
        with st.spinner("Réflexion..."):
            agent = init_agent()
            loop = asyncio.new_event_loop()
            response = loop.run_until_complete(
                agent.ainvoke({"messages": [{"role": "user", "content": prompt}]})
            )
            answer = response["messages"][-1].content
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})







            