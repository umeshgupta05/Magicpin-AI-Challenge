from typing import Any

from fastapi import FastAPI, Response
from pydantic import BaseModel, Field

from composer import compose
from engine import VeraEngine


app = FastAPI(title="Vera Challenge Bot", version="1.0.0")
engine = VeraEngine()


class ContextRequest(BaseModel):
    scope: str
    context_id: str
    version: int = Field(ge=0)
    payload: dict[str, Any]
    delivered_at: str | None = None


class TickRequest(BaseModel):
    now: str
    available_triggers: list[str] = Field(default_factory=list)


class ReplyRequest(BaseModel):
    conversation_id: str
    merchant_id: str | None = None
    customer_id: str | None = None
    from_role: str = "merchant"
    message: str
    received_at: str
    turn_number: int = 1


@app.get("/")
async def root() -> dict:
    return {
        "status": "ok",
        "message": "Vera Challenge Bot is running. Use the /v1 endpoints for judging.",
        "endpoints": {
            "healthz": "/v1/healthz",
            "metadata": "/v1/metadata",
            "context": "/v1/context",
            "tick": "/v1/tick",
            "reply": "/v1/reply",
        },
    }


@app.get("/v1/healthz")
async def healthz() -> dict:
    return engine.healthz()


@app.get("/v1/metadata")
async def metadata() -> dict:
    return engine.metadata()


@app.post("/v1/context")
async def context(body: ContextRequest, response: Response) -> dict:
    status, payload = engine.push_context(
        body.scope,
        body.context_id,
        body.version,
        body.payload,
        body.delivered_at,
    )
    response.status_code = status
    return payload


@app.post("/v1/tick")
async def tick(body: TickRequest) -> dict:
    return await engine.tick(body.now, body.available_triggers)


@app.post("/v1/reply")
async def reply(body: ReplyRequest) -> dict:
    return engine.reply(
        conversation_id=body.conversation_id,
        merchant_id=body.merchant_id,
        customer_id=body.customer_id,
        from_role=body.from_role,
        message=body.message,
        received_at=body.received_at,
        turn_number=body.turn_number,
    )


@app.post("/v1/teardown")
async def teardown() -> dict:
    return engine.teardown()
