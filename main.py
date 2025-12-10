from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
import chat

app = FastAPI(title="Acquaviva Chatbot API", description="API para interactuar con el bot RAG de Acquaviva")

# Request Model
class ChatRequest(BaseModel):
    message: str

# Response Model
class ChatResponse(BaseModel):
    results: List[Dict[str, Any]]

@app.on_event("startup")
async def startup_event():
    """Load resources on startup to avoid delay on first request"""
    print("Application startup: Pre-loading resources...")
    chat.init_resources()

@app.get("/")
def health_check():
    return {"status": "Acquaviva Bot Active (Raw Mode)"}

@app.get("/privacy", response_class=HTMLResponse)
def privacy_policy():
    return """
    <html>
        <head>
            <title>Política de Privacidad</title>
        </head>
        <body>
            <h1>Política de Privacidad</h1>
            <p>El bot 'Acquaviva Expert' usa datos públicos para fines informativos y no guarda datos del usuario.</p>
            <p>Contacto: contacto@ejemplo.com</p>
        </body>
    </html>
    """

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    if not request.message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    try:
        # Import and use the refactored function
        # Now returns a list of dictionaries
        raw_results = chat.get_acquaviva_response(request.message)
        return ChatResponse(results=raw_results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Host 0.0.0.0 allows external access, port is dynamic for Render
    import os
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
