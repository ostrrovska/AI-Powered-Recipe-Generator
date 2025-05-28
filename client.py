import gradio as gr
from smolagents import InferenceClientModel, CodeAgent
from smolagents.mcp_client import MCPClient
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Connect to the MCP server
mcp_client = MCPClient(
    {"url": os.getenv("MCP_SERVER_URL", "http://localhost:7860/gradio_api/mcp/sse")}
)

try:
    # Get available tools from the server
    tools = mcp_client.get_tools()

    # Create an agent that will use the tools
    model = InferenceClientModel()
    agent = CodeAgent(tools=[*tools], model=model)

    # Create a chat interface
    demo = gr.ChatInterface(
        fn=lambda message, history: str(agent.run(message)),
        title="Recipe Assistant",
        description="Ask me about recipes! I can help you find recipes based on ingredients.",
        examples=[
            "Find me recipes with chicken and rice",
            "What can I make with tomatoes, garlic, and pasta?",
            "I have chocolate, flour, and eggs. What desserts can I make?",
            "Suggest recipes with potatoes and cheese"
        ]
    )

    # Launch the interface
    if __name__ == "__main__":
        demo.launch()

finally:
    # Always disconnect the client when done
    mcp_client.disconnect() 