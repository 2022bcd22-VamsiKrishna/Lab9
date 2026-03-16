from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "California Housing ML Model - DVC Lab 8"}