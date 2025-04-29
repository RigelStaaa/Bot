import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_groq import ChatGroq
import gspread
from gspread_dataframe import get_as_dataframe
from google.oauth2 import service_account
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from difflib import get_close_matches
import os
import json

# Set up environment variables
# Note: Set GROQ_API_KEY and GOOGLE_CREDENTIALS in Render Dashboard

app = FastAPI()

# Serve static files (HTML, JS, CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def get_chat_widget():
    return FileResponse("static/chatbot.html")  # Returns your chatbot HTML file

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
    
    df = get_as_dataframe(worksheet).dropna(how='all')
    print(f"Loaded {len(df)} entries from Google Sheets.")
    return df

# Function to refresh the bot with new data
def refresh_bot():
    global df, agent
    df = load_data()
    
    if df.empty:
        print("Warning: No data loaded. The bot will not function properly.")
    else:
        print(f"Data refreshed with {len(df)} entries.")

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

# Request model for asking questions
class Query(BaseModel):
    message: str

@app.post("/ask")
async def ask_question(query: Query):
    try:
        user_input = query.message.strip()
        print(f"User query: {user_input}")

        if not user_input:
            return JSONResponse(content={"response": "Please enter a message."})

        # Step 1: Fast string match with OSV/FTWZ dataset
        questions_list = [q.lower() for q in df['questions'].dropna().tolist()]
        match = get_close_matches(user_input.lower(), questions_list, n=1, cutoff=0.7)

        if match:
            best_question = match[0]
            original_question_index = questions_list.index(best_question)
            best_answer = df.iloc[original_question_index]['answers']
            print(f"[Fast Match] {df.iloc[original_question_index]['questions']} -> {best_answer}")
            return JSONResponse(content={"response": best_answer})

        # Step 2: If no string match, use semantic search within OSV/FTWZ dataset
        all_qna = "\n".join(f"Q: {q}\nA: {a}" for q, a in zip(df['questions'], df['answers']))

        prompt = f"""You are a helpful assistant for OSV FTWZ FAQs.
Below are the stored Q&A:

{all_qna}

A user asked: "{user_input}"

Find the most relevant answer. If not found, reply politely that you don't have an answer.
Respond ONLY with the answer, not the question.
"""

        response = llm.invoke(prompt)
        answer_text = response.content.strip()

        if answer_text:
            print(f"[Semantic Answer] {answer_text}")
            return JSONResponse(content={"response": answer_text})

        # Step 3: Fallback - If no relevant FAQ match, answer using the model's general knowledge
        # This is the fallback section where the bot uses its own intelligence
        fallback_prompt = f"""You are a general knowledge assistant. Please respond to the following question with your own intelligence:

Question: {user_input}

Provide a helpful and informative answer.
"""

        general_response = llm.invoke(fallback_prompt)
        general_answer_text = general_response.content.strip()

        if general_answer_text:
            print(f"[General Knowledge Answer] {general_answer_text}")
            return JSONResponse(content={"response": general_answer_text})
        else:
            return JSONResponse(content={"response": "I'm sorry, I don't have an answer for that."})

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=400)

# Request model for storing client info
class ClientInfo(BaseModel):
    name: str
    email: str
    phone: str

@app.post("/start")
async def start_chat(client: ClientInfo):
    try:
        SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
        credentials_info = json.loads(os.environ["GOOGLE_CREDENTIALS"])
        creds = service_account.Credentials.from_service_account_info(credentials_info, scopes=SCOPES)
        gc = gspread.authorize(creds)

        sheet = gc.open_by_url("https://docs.google.com/spreadsheets/d/1s_uCbs8vxwKQGABKCjEpZCNrxn6omqcJG1DK3HklOF8/edit?usp=sharing")
        
        # Assuming there is a 'Clients' worksheet to store client info
        try:
            worksheet = sheet.worksheet("Clients")
        except gspread.exceptions.WorksheetNotFound:
            worksheet = sheet.add_worksheet(title="Clients", rows="1000", cols="3")
            worksheet.append_row(["Name", "Email", "Phone"])

        # Append client data
        worksheet.append_row([client.name, client.email, client.phone])

        return JSONResponse(content={"message": "Client information stored successfully."})

    except Exception as e:
        print(f"Error occurred while storing client info: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=400)

# Refresh endpoint
@app.get("/refresh")
async def refresh():
    global agent
    agent = refresh_bot()
    return JSONResponse(content={"message": "Bot refreshed successfully!"})
