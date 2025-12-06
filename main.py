from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import chat

app = FastAPI(title="Acquaviva Chatbot API", description="API para interactuar con el bot RAG de Acquaviva")

# Request Model
class ChatRequest(BaseModel):
    message: str

# Response Model
class ChatResponse(BaseModel):
    response: str

@app.on_event("startup")
async def startup_event():
    """Load resources on startup to avoid delay on first request"""
    print("Application startup: Pre-loading resources...")
    chat.init_resources()

@app.get("/")
def health_check():
    return {"status": "Acquaviva Bot Active"}

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    if not request.message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    try:
        # Import and use the refactored function
        response_text = chat.get_acquaviva_response(request.message)
        return ChatResponse(response=response_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Host 0.0.0.0 allows external access, port is dynamic for Render
    import os
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
