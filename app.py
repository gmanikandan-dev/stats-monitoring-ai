import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
import pandas as pd
from pymongo import MongoClient
from datetime import datetime
import io
import uvicorn
import json
from langchain.chains import RetrievalQA
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from dotenv import load_dotenv  # ✅ Import dotenv
from langchain_core.prompts import ChatPromptTemplate

# ✅ Load environment variables
load_dotenv()

app = FastAPI()

# MongoDB Connection
client = MongoClient("mongodb://localhost:27017/")
db = client["reward_system"]
collection = db["tenant_rewards"]

# ✅ Set OpenAI API key from .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("MODEL")

# Define the Chat Model
chat_model = ChatOpenAI(
    model=MODEL,  # or "gpt-3.5-turbo"
    openai_api_key=OPENAI_API_KEY  # Load API Key from .env
)

# -------------------- Embeddings Processing --------------------

# Load LLM & Embeddings
llm = ChatOpenAI(model=MODEL, temperature=0)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

try:
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)    
except:
    vector_store = FAISS.from_documents([], embeddings)  # Create an empty FAISS store if not found

retriever = vector_store.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

# # Convert MongoDB records to text format
# def prepare_text_data(records):
#     return "\n".join([
#         f"Tenant: {record['tenant_name']}, Date: {record['date']}, Total Points: {record['total_account_points']}"
#         for record in records
#     ])

# Generate and store embeddings
def generate_and_store_embeddings():
    records = list(collection.find({}, {"_id": 0}))
    if not records:
        print("No records found for embeddings.")
        return

    documents = [
        Document(
            page_content=json.dumps(record, default=str),
            metadata={"tenant_name": record["tenant_name"], "date": str(record["date"])}
        )
        for record in records
    ]

    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local("faiss_index")
    print("Embeddings updated successfully.")


# -------------------- CSV Upload API --------------------

# Upload CSV API
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
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")

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

# Function to serialize datetime objects
def serialize_datetime(obj):
    if isinstance(obj, datetime):
        return obj.strftime("%Y-%m-%d")  # Convert to string
    return obj

# Query LLM API
@app.get("/query_llm/")
async def query_llm(start_date: str, end_date: str):
    try:
        start_dt = datetime.strptime(start_date, "%d-%m-%Y")
        end_dt = datetime.strptime(end_date, "%d-%m-%Y")

        records = list(collection.find({"date": {"$gte": start_dt, "$lte": end_dt}}, {"_id": 0}))

        if not records:
            return {"message": "No data found for the given date range"}

        # Prepare context for LLM
        context = "\n".join([json.dumps(record, default=serialize_datetime) for record in records])
        # query = f"Which tenant has the highest total account points between {start_date} and {end_date}?"
        query = f"What is the sum of total_account_points for CCC store between {start_date} and {end_date}?"

        # Correct Chat Message Format
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that can analyze CSV data.You can perform aggregation tasks like counting, summing, averaging, and finding the maximum, minimum, and average values of columns. The response should be a number."),
            ("user", "{query}\n\n{context}")
        ])

        formatted_prompt = prompt.format(query=query, context=context)
        response = chat_model.invoke(formatted_prompt)  # ✅ Correct format

        return {"response": response}

    except Exception as e:
        return {"error": str(e)}

# -------------------- Run FastAPI Server --------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)