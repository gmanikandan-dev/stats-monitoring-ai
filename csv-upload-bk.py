from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
from pymongo import MongoClient
import datetime
import io
import uvicorn

app = FastAPI()

# MongoDB Connection
client = MongoClient("mongodb://localhost:27017/")  # Update if needed
db = client["reward_system"]
collection = db["tenant_rewards"]

@app.post("/upload_csv/")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    # Ensure required columns exist
    required_columns = {"tenant_name", "date", "new_accounts", "account_points", "total_account_points"}
    
    if not required_columns.issubset(df.columns):
        raise HTTPException(status_code=400, detail="CSV missing required columns")
    
    # Convert 'date' to datetime for querying
    df["date"] = pd.to_datetime(df["date"])
    
    # Insert data into MongoDB
    records = df.to_dict(orient="records")
    collection.insert_many(records)
    
    return {"message": "CSV uploaded successfully", "rows_inserted": len(records)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
