import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_groq import ChatGroq
import gspread
from gspread_dataframe import get_as_dataframe
from google.oauth2.service_account import Credentials
import os
import json

from google.oauth2 import service_account
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# Set up environment variables
os.environ["GROQ_API_KEY"] = "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"

# Initialize FastAPI app
app = FastAPI()

# Initialize Groq model
llm = ChatGroq(
    temperature=0,
    model="llama3-70b-8192",
    api_key=os.environ["GROQ_API_KEY"]
)

# Load Google Sheet into DataFrame
SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
credentials_info = json.loads(os.environ["GOOGLE_CREDENTIALS"])
creds = service_account.Credentials.from_service_account_info(credentials_info, scopes=SCOPES)
gc = gspread.authorize(creds)

sheet = gc.open_by_url("https://docs.google.com/spreadsheets/d/1s_uCbs8vxwKQGABKCjEpZCNrxn6omqcJG1DK3HklOF8/edit?usp=sharing")
worksheet = sheet.worksheet("Sheet1")
df = get_as_dataframe(worksheet).dropna(how='all')

# LangChain Agent
agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=True,
    agent_type="openai-tools",
    handle_parsing_errors=True,
    allow_dangerous_code=True
)

# Request model
class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(query: Query):
    try:
        response = agent.invoke({"input": query.question})
        return {"response": response["output"]}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("botfast:app", host="0.0.0.0", port=8000, reload=True)

