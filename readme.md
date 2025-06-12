# McpServer_DataFair

🚀 **A Model Context Protocol (MCP) server for DataFair portal integration with Streamlit client example**

## 📋 Overview

This project provides access to DataFair portal data through an MCP server and offers a concrete client example demonstrating how to query the server and leverage the obtained results. The implementation showcases the power of MCP architecture for data integration and AI-powered data exploration.

## 🏗️ Architecture

The project consists of two main components:

- **MCP Server** (`mcp_server.py`): Exposes DataFair portal data through the Model Context Protocol
- **Streamlit Client** (`client.py`): Interactive web interface that demonstrates server interaction capabilities

## ⚡ Quick Start

### Prerequisites

- Python 3+
- Required dependencies (install via `pip install -r requirements.txt`)
- DataFair portal access credentials

### Environment Setup

1. Clone the repository:

```bash
git clone https://github.com/data354/McpServer_DataFair.git
cd McpServer_DataFair
```

2. Create and configure your environment file:

```bash
cp .env.example .env
# Edit .env with your DataFair credentials
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 Running the Project

### Option 1: Standard Server Launch

Launch the MCP server in listening mode:

```bash
python mcp_server.py
```

The server will start and listen for incoming MCP connections.

### Option 2: Development Mode with MCP SDK

For development and testing with MCP SDK tools:

```bash
mcp dev mcp_server.py
```

This launches the server with MCP development tools and enhanced debugging capabilities.

### Running the Streamlit Client

In a separate terminal, launch the interactive Streamlit client:

```bash
streamlit run client.py
```

The web interface will be available at `http://localhost:8501`

## 🎯 Features

### MCP Server Capabilities

- ✅ DataFair portal data exposure
- ✅ RESTful API integration
- ✅ Real-time data querying
- ✅ Structured data responses
- ✅ Error handling and logging

### Streamlit Client Features

- 🤖 AI-powered data exploration
- 💬 Conversational interface
- 📊 Data visualization
- 💾 Conversation history
- 📱 Responsive design
- 🔄 Real-time data updates

## 🛠️ Usage Example

1. **Start the MCP server** using one of the methods above
2. **Launch the Streamlit client** to access the web interface
3. **Ask questions** about DataFair data through the chat interface
4. **Explore visualizations** and export results as needed

## 📁 Project Structure

```
McpServer_DataFair/
├── mcp_server.py          # MCP server implementation
├── client.py              # Streamlit client application
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## 🔧 Configuration

The project uses environment variables for configuration. Key variables include:

- `DATAFAIR_API_URL`: DataFair portal API endpoint
- `DATAFAIR_API_KEY`: Authentication key for DataFair access
- `OPENAI_API_KEY`: OpenAI API key for AI functionality

## 🎉 Conclusion

This project demonstrates the seamless integration between DataFair portal data and modern AI interfaces through the Model Context Protocol. By combining MCP's standardized communication with Streamlit's interactive capabilities, we've created a powerful tool for data exploration and analysis.

The MCP architecture ensures scalable and maintainable data access, while the Streamlit client provides an intuitive interface for both technical and non-technical users. This combination opens up new possibilities for data democratization and AI-powered insights.

**Ready to explore your data? Start the servers and dive in!** 🚀

---

_Developed by Data354_
