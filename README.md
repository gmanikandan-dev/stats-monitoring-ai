# Python stats monitoring AI.

### _Requirements_  
```
python3 --version => 3.10.12
```


>  _Step 1 :Starting local development server using below command: http://localhost:8000

````
python3 app.py
````
> _Step 2 : To upload your csv using below curl request
````
curl -X 'POST' \
  'http://localhost:8000/upload_csv/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@/path/to/your/file.csv'

````

> _Step 3 : To ask query using below curl request
````
curl -X 'GET' "http://localhost:8000/query_llm/?start_date=01-03-2025&end_date=03-03-2025" -H "accept: application/json"
````
> 
## _Follow my git profile and Thank you for visiting_ 
