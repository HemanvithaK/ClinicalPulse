from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from dotenv import load_dotenv
from agent.graph import build_graph
from voice.stt import SpeechToText
from voice.tts import TextToSpeech
from ingestion.scheduler import get_background_scheduler
import uvicorn
import asyncio

load_dotenv()

app = FastAPI(
    title="ClinicalPulse API",
    description="Real-time clinical trial intelligence agent",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

graph = build_graph()
stt = SpeechToText()
tts = TextToSpeech()


class QueryRequest(BaseModel):
    query: str
    voice_response: bool = False


class QueryResponse(BaseModel):
    answer: str
    citations: list[str]
    confidence_score: float
    hallucination_flags: list[str]


def invoke_graph(query: str, voice: bool = False) -> dict:
    return graph.invoke({
        "query": query,
        "voice_input": voice,
        "retrieval_results": [],
        "graph_context": [],
        "answer": "",
        "citations": [],
        "confidence_score": 0.0,
        "hallucination_flags": [],
        "regeneration_count": 0,
        "conversation_history": [],
        "error": None,
    })


@app.on_event("startup")
async def startup():
    scheduler = get_background_scheduler()
    scheduler.start()
    print("Background ingestion scheduler started")


@app.on_event("shutdown")
async def shutdown():
    print("Shutting down")


@app.get("/health", description="Simple health check to confirm server is running")
async def health():
    return {"status": "ok", "service": "ClinicalPulse"}


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Text query returns JSON answer with citations."""
    try:
        result = invoke_graph(request.query)
        return QueryResponse(
            answer=result["answer"],
            citations=result["citations"],
            confidence_score=result["confidence_score"],
            hallucination_flags=result["hallucination_flags"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/voice")
async def query_voice_endpoint(request: QueryRequest):
    """Text query with voice metadata response."""
    try:
        result = invoke_graph(request.query)
        answer = result["answer"]
        audio_bytes = tts.synthesize(answer)
        return {
            "answer": answer,
            "citations": result["citations"],
            "confidence_score": result["confidence_score"],
            "audio_size_bytes": len(audio_bytes),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/audio")
async def query_audio_endpoint(request: QueryRequest):
    """Returns raw MP3 audio bytes for browser playback."""
    try:
        result = invoke_graph(request.query)
        answer = result["answer"]
        audio_bytes = tts.synthesize(answer)
        return Response(
            content=audio_bytes,
            media_type="audio/mpeg",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/voice")
async def voice_websocket(websocket: WebSocket):
    """WebSocket for real-time voice interaction."""
    await websocket.accept()
    print("Voice WebSocket connected")

    try:
        while True:
            audio_bytes = await websocket.receive_bytes()
            await websocket.send_json({"status": "transcribing"})

            loop = asyncio.get_event_loop()
            query = await loop.run_in_executor(
                None, stt.transcribe_bytes, audio_bytes
            )
            await websocket.send_json({
                "status": "transcribed",
                "query": query,
            })

            await websocket.send_json({"status": "thinking"})
            result = await loop.run_in_executor(
                None,
                lambda: invoke_graph(query, voice=True)
            )

            answer = result["answer"]

            await websocket.send_json({
                "status": "answer",
                "answer": answer,
                "citations": result["citations"],
                "confidence_score": result["confidence_score"],
            })

            await websocket.send_json({"status": "synthesizing"})
            audio_out = await loop.run_in_executor(
                None, tts.synthesize, answer
            )
            await websocket.send_bytes(audio_out)
            await websocket.send_json({"status": "done"})

    except WebSocketDisconnect:
        print("Voice WebSocket disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.send_json({"status": "error", "message": str(e)})


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
