"""
Ticket Support AI API
FastAPI service exposing the TicketSupportAI for suggesting solutions to support tickets.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
from datetime import datetime
import json
from urllib.parse import parse_qs

from ticket_support_ai import TicketSupportAI

app = FastAPI(
    title="Ticket Support AI API",
    description="AI-powered support ticket solution suggestion service",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

KNOWLEDGE_BASE_PATH = os.path.join(os.path.dirname(__file__), "data", "ticket_knowledge_base.json")
ticket_ai = TicketSupportAI(knowledge_base_path=KNOWLEDGE_BASE_PATH)


class TicketRequest(BaseModel):
    subject: str
    message: str
    category: Optional[str] = None


class TicketSolutionResponse(BaseModel):
    suggested_solution: Optional[str]
    confidence_score: float
    escalate_to_admin: bool
    similar_ticket: Optional[dict] = None
    processed_at: str


class UpdateKnowledgeBaseRequest(BaseModel):
    subject: str
    message: str
    resolution: str
    category: Optional[str] = None


@app.get("/")
async def root():
    return {
        "service": "Ticket Support AI",
        "status": "running",
        "version": "1.0.0"
    }


@app.post("/ai/solve", response_model=TicketSolutionResponse)
async def solve_ticket(request: Request):
    try:
        content_type = request.headers.get("content-type", "")
        content_length = request.headers.get("content-length", "")
        if content_type or content_length:
            print(f"[AI] Headers: content-type={content_type} content-length={content_length}")

        raw_body = await request.body()
        _log_raw_body(raw_body)
        payload = _parse_payload_from_raw(raw_body, request)
        subject = _pick_text(payload, "subject", "title", "summary")
        message = _pick_text(payload, "message", "description", "body", "content")

        _log_request_payload(payload, subject, message)

        result = ticket_ai.suggest_solution(
            subject=subject,
            message=message,
            confidence_threshold=0.75
        )
        result["processed_at"] = datetime.now().isoformat()
        return result
    except Exception as e:
        print(f"Error solving ticket: {str(e)}")
        return {
            "suggested_solution": None,
            "confidence_score": 0.0,
            "escalate_to_admin": True,
            "similar_ticket": None,
            "processed_at": datetime.now().isoformat()
        }


@app.post("/ai/knowledge-base/update")
async def update_knowledge_base(request: Request):
    try:
        content_type = request.headers.get("content-type", "")
        content_length = request.headers.get("content-length", "")
        if content_type or content_length:
            print(f"[AI] Headers: content-type={content_type} content-length={content_length}")

        raw_body = await request.body()
        _log_raw_body(raw_body)
        payload = _parse_payload_from_raw(raw_body, request)
        subject = _pick_text(payload, "subject", "title", "summary")
        message = _pick_text(payload, "message", "description", "body", "content")
        resolution = _pick_text(payload, "resolution", "solution", "reply", "answer")
        category = _pick_text(payload, "category", "type")

        _log_request_payload(payload, subject, message)

        if not subject and not message:
            print("AI update failed but ticket closed normally: missing fields")
            raise HTTPException(
                status_code=422,
                detail="Missing required fields: subject/message and resolution"
            )

        if not resolution:
            print("AI update failed but ticket closed normally: missing fields")
            raise HTTPException(
                status_code=422,
                detail="Missing required fields: resolution"
            )

        # Fallback if message is empty but subject is present
        if not message:
            message = subject
        
        if not subject:
            subject = message

        added = ticket_ai.add_ticket_if_new(
            subject=subject,
            message=message,
            resolution=resolution,
            category=category or None
        )
        if added:
            ticket_ai.save_knowledge_base(KNOWLEDGE_BASE_PATH)
            print("AI knowledge updated successfully")
        else:
            print("AI knowledge update skipped (duplicate)")
        return {
            "status": "success" if added else "duplicate",
            "message": "Knowledge base updated successfully" if added else "Duplicate ticket ignored",
            "updated_at": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error updating knowledge base: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ai/knowledge-base/retrain")
async def retrain_model():
    try:
        ticket_ai.update_model()
        return {
            "status": "success",
            "message": "Model retrained successfully",
            "retrained_at": datetime.now().isoformat(),
            "total_tickets": len(ticket_ai.tickets)
        }
    except Exception as e:
        print(f"Error retraining model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ai/knowledge-base/stats")
async def get_knowledge_base_stats():
    return {
        "total_tickets": len(ticket_ai.tickets),
        "categories": _get_category_stats(),
        "last_updated": datetime.now().isoformat()
    }


def _get_category_stats():
    categories = {}
    for ticket in ticket_ai.tickets:
        category = ticket.get("category", "Uncategorized")
        categories[category] = categories.get(category, 0) + 1
    return categories


def _parse_payload_from_raw(raw_body: bytes, request: Request) -> dict:
    if not raw_body:
        query_payload = dict(request.query_params)
        return query_payload if query_payload else {}

    text = raw_body.decode("utf-8", errors="ignore").strip()
    if not text:
        return {}

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            return parsed[0]
        if isinstance(parsed, str):
            try:
                nested = json.loads(parsed)
                if isinstance(nested, dict):
                    return nested
            except Exception:
                return {"message": parsed}
    except Exception:
        pass

    if "=" in text and "{" not in text:
        form_payload = {
            key: values[0]
            for key, values in parse_qs(text, keep_blank_values=True).items()
            if values
        }
        if form_payload:
            return form_payload

    return {"message": text}


def _pick_text(payload: dict, *keys: str) -> str:
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _log_request_payload(payload: dict, subject: str, message: str) -> None:
    try:
        keys = ", ".join(sorted(payload.keys())) if isinstance(payload, dict) else "(non-dict)"
        print(f"[AI] Payload keys: {keys}")
        print(f"[AI] Subject length: {len(subject)} | Message length: {len(message)}")
    except Exception:
        pass


def _log_raw_body(raw_body: bytes) -> None:
    try:
        raw_len = len(raw_body) if raw_body else 0
        preview_bytes = raw_body[:200] if raw_body else b""
        preview_text = preview_bytes.decode("utf-8", errors="replace")
        preview_text = preview_text.replace("\n", "\\n").replace("\r", "\\r")
        print(f"[AI] Raw body len: {raw_len} preview: {preview_text}")
    except Exception:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
