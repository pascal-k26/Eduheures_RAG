import sys
import os
import re
import uuid
import random
import logging
import secrets
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from src.generation import repondre
    from langchain_core.messages import HumanMessage, AIMessage
    RAG_DISPONIBLE = True
except ImportError as e:
    logging.warning(f"Module RAG non disponible : {e}")
    RAG_DISPONIBLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("eduheures_api")

#  Constantes 
MAX_HISTORIQUE       = 10
TTL_SESSION_MINUTES  = 60
MAX_SESSIONS_TOTALES = 500
MAX_MESSAGE_CHARS    = 2000
RATE_LIMIT_CHAT      = "10/minute"
RATE_LIMIT_SESSION   = "20/minute"
RAG_TIMEOUT_SECONDES = 180.0

_executor = ThreadPoolExecutor(max_workers=2)

#  CORS 
_raw_origins = os.getenv("ALLOWED_ORIGINS", "")
ALLOWED_ORIGINS: list[str] = (
    [o.strip() for o in _raw_origins.split(",") if o.strip()]
    if _raw_origins
    else [
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "https://eduheures.netlify.app",   # à adapter 
    ]
)

#  Auth API Key 
API_SECRET_KEY = os.getenv("API_SECRET_KEY", "")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verifier_api_key(key: Optional[str] = Depends(api_key_header)) -> bool:
    if not API_SECRET_KEY:
        return True
    if not key:
        raise HTTPException(status_code=401, detail="Clé API manquante.")
    if not secrets.compare_digest(key, API_SECRET_KEY):
        raise HTTPException(status_code=403, detail="Clé API invalide.")
    return True


# Application 
limiter = Limiter(key_func=get_remote_address, default_limits=["200/hour"])

app = FastAPI(
    title="Eduheures API",
    description="Backend RAG agentique pour l'assistant universitaire UVCI",
    version="1.0.0",
    docs_url="/docs" if os.getenv("ENV", "production") == "development" else None,
    redoc_url=None,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "X-API-Key"],
)

sessions: dict[str, dict] = {}
_UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def nettoyer_sessions_expirées():
    expiration = datetime.utcnow() - timedelta(minutes=TTL_SESSION_MINUTES)
    ids = [sid for sid, d in sessions.items() if d["dernière_activité"] < expiration]
    for sid in ids:
        del sessions[sid]
    if ids:
        logger.info(f"{len(ids)} session(s) expirée(s) supprimée(s)")


@app.middleware("http")
async def nettoyage_periodique(request: Request, call_next):
    if random.randint(1, 50) == 1:
        nettoyer_sessions_expirées()
    return await call_next(request)


async def appeler_rag(question: str, historique: list) -> dict:
    loop = asyncio.get_event_loop()
    return await asyncio.wait_for(
        loop.run_in_executor(_executor, lambda: repondre(question=question, historique=historique)),
        timeout=RAG_TIMEOUT_SECONDES,
    )


@app.on_event("startup")
async def warmup():
    if not RAG_DISPONIBLE:
        return
    logger.info("Warm-up Eduheures RAG...")
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(_executor, lambda: repondre(question="bonjour", historique=[]))
        logger.info("Warm-up terminé.")
    except Exception as e:
        logger.warning(f"Warm-up échoué (non bloquant) : {e}")


# Modèles Pydantic 

class SessionResponse(BaseModel):
    session_id: str
    message: str


class ChatRequest(BaseModel):
    session_id: str
    message: str

    @field_validator("session_id")
    @classmethod
    def valider_uuid(cls, v: str) -> str:
        if not _UUID_PATTERN.match(v):
            raise ValueError("session_id invalide.")
        return v

    @field_validator("message")
    @classmethod
    def valider_message(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Le message ne peut pas être vide.")
        if len(v) > MAX_MESSAGE_CHARS:
            raise ValueError(f"Message trop long ({len(v)} chars, max {MAX_MESSAGE_CHARS}).")
        return v


class SourceInfo(BaseModel):
    fichier: str
    score: float
    type: Optional[str] = "document"
    titre: Optional[str] = None
    page: Optional[str] = None


class ChatResponse(BaseModel):
    reponse: str
    nb_sources: int
    sources: list[SourceInfo]
    session_id: str


class HealthResponse(BaseModel):
    status: str
    rag_disponible: bool
    sessions_actives: int
    version: str


#  Endpoints

@app.get("/health", response_model=HealthResponse, tags=["Système"],
         dependencies=[Depends(verifier_api_key)])
async def health():
    nettoyer_sessions_expirées()
    return HealthResponse(
        status="ok",
        rag_disponible=RAG_DISPONIBLE,
        sessions_actives=len(sessions),
        version=app.version,
    )


@app.post("/session/nouvelle", response_model=SessionResponse, tags=["Sessions"],
          dependencies=[Depends(verifier_api_key)])
@limiter.limit(RATE_LIMIT_SESSION)
async def creer_session(request: Request):
    nettoyer_sessions_expirées()
    if len(sessions) >= MAX_SESSIONS_TOTALES:
        raise HTTPException(status_code=503, detail="Trop de sessions actives. Réessayez.")
    session_id = str(uuid.uuid4())
    sessions[session_id] = {"historique": [], "dernière_activité": datetime.utcnow(), "nb_échanges": 0}
    return SessionResponse(session_id=session_id, message="Session créée")


@app.delete("/session/{session_id}", tags=["Sessions"],
            dependencies=[Depends(verifier_api_key)])
async def supprimer_session(session_id: str):
    if not _UUID_PATTERN.match(session_id):
        raise HTTPException(status_code=400, detail="session_id invalide.")
    sessions.pop(session_id, None)
    return {"message": "Session supprimée"}


@app.post("/session/{session_id}/fermer", tags=["Sessions"],
          dependencies=[Depends(verifier_api_key)])
async def fermer_session(session_id: str):
    if not _UUID_PATTERN.match(session_id):
        raise HTTPException(status_code=400, detail="session_id invalide.")
    sessions.pop(session_id, None)
    return {"message": "Session fermée"}


@app.post("/chat", response_model=ChatResponse, tags=["Chat"],
          dependencies=[Depends(verifier_api_key)])
@limiter.limit(RATE_LIMIT_CHAT)
async def chat(request: Request, data: ChatRequest):
    if not RAG_DISPONIBLE:
        raise HTTPException(status_code=503, detail="Pipeline RAG non disponible.")

    if data.session_id not in sessions:
        raise HTTPException(status_code=404,
                            detail="Session introuvable. Créez une session via POST /session/nouvelle.")

    session = sessions[data.session_id]
    session["dernière_activité"] = datetime.utcnow()
    logger.info(f"[{data.session_id[:8]}] Question reçue ({len(data.message)} chars)")

    try:
        resultat = await appeler_rag(data.message, session["historique"])
    except asyncio.TimeoutError:
        raise HTTPException(status_code=503, detail="Timeout — réessayez dans 30 secondes.")
    except Exception as e:
        logger.error(f"Erreur RAG : {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Erreur interne.")

    reponse      = resultat.get("reponse", "")
    sources_brutes = resultat.get("sources", [])
    nb_sources   = resultat.get("nb_sources", 0)

    session["historique"].append(HumanMessage(content=data.message))
    session["historique"].append(AIMessage(content=reponse))
    session["nb_échanges"] += 1

    if len(session["historique"]) > MAX_HISTORIQUE * 2:
        brut = session["historique"][-(MAX_HISTORIQUE * 2):]
        if brut and not isinstance(brut[0], HumanMessage):
            brut = brut[1:]
        session["historique"] = brut

    sources = [
        SourceInfo(
            fichier = src.get("fichier", "Inconnu"),
            score   = round(float(src.get("score", 0)), 1),
            type    = src.get("type", "document"),
            titre   = src.get("titre"),
            page    = str(src.get("page", "")) if src.get("page") is not None else None,
        )
        for src in sources_brutes
    ]

    return ChatResponse(reponse=reponse, nb_sources=nb_sources, sources=sources, session_id=data.session_id)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server_fastapi:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("ENV", "production") == "development",
    )
