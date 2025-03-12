# Python stats monitoring AI.

### _Requirements_  
```
python3 --version => 3.10.12
```

## Installation & Setup

#### Clone repository

````
git clone https://github.com/gmanikandan-dev/stats-monitoring-ai.git

````
#### create .env file

````
cp .env.example .env
````

#### Add Groq key in .env file

#### Install virtual environment

````
python3 -m venv venv
````
#### Activate virtual environment

````
source venv/bin/activate
````
#### Install requirements
````
pip install -r requirements.txt
````

>  Starting local development server using below command: http://localhost:8000

````
python3 app.py
````
> To upload your csv using below curl request
````
curl -X 'POST' \
  'http://localhost:8000/upload_csv/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@/path/to/your/file.csv'

````

> To ask query using below curl request
````
curl -X 'GET' "http://localhost:8000/query_llm/?start_date=01-03-2025&end_date=03-03-2025" -H "accept: application/json"
````
> 
## _Follow my git profile and Thank you for visiting_ 
