import os
from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
from pymongo import MongoClient
import datetime
import io
import uvicorn
import json
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.docstore.document import Document
from pydantic import BaseModel
from dotenv import load_dotenv 

# ✅ Load environment variables
load_dotenv()

app = FastAPI()

# MongoDB Connection
client = MongoClient("mongodb://localhost:27017/")
db = client["reward_system"]
collection = db["tenant_rewards"]

# Define OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# -------------------- Embeddings Processing --------------------

# Fetch reward data from MongoDB
def fetch_data():
    records = collection.find({}, {"_id": 0})  # Exclude MongoDB's _id
    return list(records)

# Convert data into text format for embeddings
def prepare_text_data(records):
    return [
        Document(
            page_content=json.dumps(record, default=str),  # Convert datetime to string
            metadata={"tenant_name": record["tenant_name"], "date": str(record["date"])}  # Ensure date is a string
        )
        for record in records
    ]

# Generate and store embeddings
def generate_and_store_embeddings():
    records = fetch_data()
    if not records:
        print("No data available for embeddings.")
        return

    documents = prepare_text_data(records)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local("faiss_index")

    print("✅ Embeddings generated and stored successfully.")

# Load FAISS vector store for querying
def load_vector_store():
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# -------------------- CSV Upload API --------------------

@app.post("/upload_csv/")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

    # Debugging: print CSV contents
    print(df.head())

    # Ensure required columns exist
    required_columns = {"tenant_name", "date", "new_accounts", "account_points", "total_account_points"}
    if not required_columns.issubset(df.columns):
        missing_columns = required_columns - set(df.columns)
        raise HTTPException(status_code=400, detail=f"CSV missing required columns: {missing_columns}")

    # Convert 'date' to datetime
    df["date"] = pd.to_datetime(df["date"])

    # Convert numeric columns, handling errors
    int_columns = ["new_accounts", "account_points", "total_account_points"]
    for col in int_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with invalid values
    invalid_rows = df[df.isnull().any(axis=1)]
    if not invalid_rows.empty:
        print("Invalid data found:", invalid_rows)  # Debugging
        raise HTTPException(status_code=400, detail="CSV contains invalid numeric values.")

    df[int_columns] = df[int_columns].astype(int)  # Convert back to int after validation

    # Insert into MongoDB
    records = df.to_dict(orient="records")
    df = df.dropna(how="all")
    collection.insert_many(records)

    # Generate embeddings after inserting data
    generate_and_store_embeddings()
    
    return {"message": "✅ CSV uploaded and embeddings generated successfully", "rows_inserted": len(records)}

# -------------------- Query Processing API --------------------

class QueryRequest(BaseModel):
    query: str

@app.post("/query/")
async def query_insights(request: QueryRequest):
    try:
        query = request.query
        vector_store = load_vector_store()
        results = vector_store.similarity_search(query, k=3)  # Retrieve top 3 results

        response = [doc.page_content for doc in results]
        return {"query": query, "results": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------- Run FastAPI Server --------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)