import uvicorn

if __name__ == "__main__":
    uvicorn.run("fast:app", host="192.168.43.125", port=4000,reload=True)


