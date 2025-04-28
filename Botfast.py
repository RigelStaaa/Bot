import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_groq import ChatGroq
import gspread
from gspread_dataframe import get_as_dataframe
from google.oauth2 import service_account
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
import os
import json
from pydantic import BaseModel

# Set up environment variables
# Note: Set GROQ_API_KEY and GOOGLE_CREDENTIALS in Render Dashboard

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>Chatbot</title>
            <style>
                body { font-family: Arial, sans-serif; background: #f2f2f2; padding: 20px; }
                #chat { margin-top: 20px; }
                .message { margin: 10px 0; }
                .user { font-weight: bold; color: blue; }
                .bot { font-weight: bold; color: green; }
            </style>
        </head>
        <body>
            <h1>Welcome to OSV Chatbot by Anurag G.</h1>
            <input type="text" id="userInput" placeholder="Type a message..." style="width:300px;" />
            <button onclick="sendMessage()">Send</button>
            <div id="chat"></div>

            <script>
                async function sendMessage() {
                    const userInput = document.getElementById('userInput').value;
                    if (!userInput) return;
                    
                    const chatDiv = document.getElementById('chat');
                    chatDiv.innerHTML += `<div class='message user'>You: ${userInput}</div>`;

                    const response = await fetch('/ask', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ message: userInput })
                    });
                    const data = await response.json();

                    chatDiv.innerHTML += `<div class='message bot'>Bot: ${data.response}</div>`;
                    document.getElementById('userInput').value = '';
                    chatDiv.scrollTop = chatDiv.scrollHeight;
                }
            </script>
        </body>
    </html>
    """

# Initialize Groq model
llm = ChatGroq(
    temperature=0,
    model="llama3-70b-8192",
    api_key=os.environ["GROQ_API_KEY"]
)

# Load Google Sheet into DataFrame
def load_data():
    SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    credentials_info = json.loads(os.environ["GOOGLE_CREDENTIALS"])
    creds = service_account.Credentials.from_service_account_info(credentials_info, scopes=SCOPES)
    gc = gspread.authorize(creds)

    sheet = gc.open_by_url("https://docs.google.com/spreadsheets/d/1s_uCbs8vxwKQGABKCjEpZCNrxn6omqcJG1DK3HklOF8/edit?usp=sharing")
    worksheet = sheet.worksheet("Sheet1")
    return get_as_dataframe(worksheet).dropna(how='all')

# Function to refresh the bot with new data
def refresh_bot():
    global df, agent  # Use global variables to update them
    df = load_data()  # Reload the data
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        agent_type="openai-tools",
        allow_dangerous_code=True
    )
    print("Bot refreshed with the latest data.")
    return agent

# Initial data load and agent creation
df = load_data()
agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=True,
    agent_type="openai-tools",
    allow_dangerous_code=True
)

# Request model
class Query(BaseModel):
    message: str

@app.post("/ask")
async def ask_question(query: Query):
    try:
        # Pass the user message to the LangChain agent
        response = agent.invoke({"input": query.message})
        # Returning the bot's response
        return JSONResponse(content={"response": response["output"]})
    except Exception as e:
        # In case of error, return a message
        return JSONResponse(content={"error": str(e)}, status_code=400)

# Refresh endpoint
@app.get("/refresh")
async def refresh():
    global agent  # Access the global agent
    agent = refresh_bot()  # Refresh the bot
    return JSONResponse(content={"message": "Bot refreshed successfully!"})
