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
import os
from dotenv import load_dotenv

# ✅ Load environment variables
load_dotenv()

app = FastAPI()

# MongoDB Connection
client = MongoClient("mongodb://localhost:27017/")
db = client["reward_system"]
collection = db["tenant_rewards"]

# ✅ Set OpenAI API key from .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Fetch reward data from MongoDB
def fetch_data():
    records = collection.find({}, {"_id": 0})  # Exclude MongoDB's default _id
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
    documents = prepare_text_data(records)
    
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local("faiss_index")
    
    print("Embeddings generated and stored successfully.")

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
    
    return {"message": "CSV uploaded and embeddings generated successfully", "rows_inserted": len(records)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)