import re
from dataclasses import dataclass, field
from typing import Any

from composer import CTA_BINARY, CTA_CONFIRM, CTA_NONE, CTA_OPEN, clean_text, first_active_offer, merchant_name


AUTO_REPLY_PATTERNS = [
    r"thank you for contacting",
    r"team will respond",
    r"respond shortly",
    r"automated assistant",
    r"business hours",
    r"away message",
    r"we will get back",
]

HOSTILE_PATTERNS = [
    r"\bstop\b",
    r"not interested",
    r"do not message",
    r"dont message",
    r"don't message",
    r"useless",
    r"spam",
    r"bothering me",
    r"abuse",
]

INTENT_PATTERNS = [
    r"\byes\b",
    r"\bok\b",
    r"go ahead",
    r"let'?s do",
    r"confirm",
    r"send it",
    r"please send",
    r"do it",
    r"start",
    r"proceed",
    r"what'?s next",
]

OFF_TOPIC_PATTERNS = [
    r"\bgst\b",
    r"income tax",
    r"file.*tax",
    r"loan",
    r"legal notice",
]


@dataclass
class ConversationState:
    conversation_id: str
    merchant_id: str | None = None
    customer_id: str | None = None
    trigger_id: str | None = None
    original_action: dict[str, Any] | None = None
    sent_bodies: list[str] = field(default_factory=list)
    turns: list[dict[str, Any]] = field(default_factory=list)
    auto_reply_count: int = 0
    ended: bool = False


class ReplyHandler:
    def respond(
        self,
        state: ConversationState,
        merchant: dict | None,
        message: str,
        merchant_auto_count: int,
    ) -> dict:
        normalized = normalize(message)
        state.turns.append({"from": "merchant", "body": message})

        if is_hostile_or_stop(normalized):
            state.ended = True
            return {
                "action": "end",
                "rationale": "Merchant explicitly opted out or showed frustration; ending and suppressing further turns.",
            }

        if is_auto_reply(normalized):
            state.auto_reply_count += 1
            if merchant_auto_count >= 3 or state.auto_reply_count >= 3:
                state.ended = True
                return {
                    "action": "end",
                    "rationale": "Repeated WhatsApp Business auto-reply detected; no real owner engagement.",
                }
            if merchant_auto_count >= 2 or state.auto_reply_count >= 2:
                return {
                    "action": "wait",
                    "wait_seconds": 86400,
                    "rationale": "Same canned auto-reply repeated; waiting 24h before any retry.",
                }
            return {
                "action": "wait",
                "wait_seconds": 14400,
                "rationale": "Canned auto-reply detected; backing off 4h for the owner or manager.",
            }

        if is_off_topic(normalized):
            body = (
                "I should leave that to the right specialist. Coming back to the Vera task: "
                "reply YES and I will send the ready draft/action from this thread."
            )
            return send_once(state, body, CTA_BINARY, "Off-topic ask declined politely; redirected once to the original business task.")

        if is_intent(normalized):
            body = action_mode_body(state, merchant)
            return send_once(state, body, CTA_CONFIRM, "Merchant committed; switching from persuasion to execution.")

        if "later" in normalized or "busy" in normalized or "tomorrow" in normalized:
            return {
                "action": "wait",
                "wait_seconds": 1800,
                "rationale": "Merchant asked implicitly for time; backing off 30 minutes.",
            }

        body = (
            "Got it. I will keep it tight: I can prepare the draft/action from this trigger and you can approve it before anything goes out. "
            "Reply YES to see the draft."
        )
        return send_once(state, body, CTA_BINARY, "Ambiguous reply acknowledged; next step kept binary and low-friction.")


def send_once(state: ConversationState, body: str, cta: str, rationale: str) -> dict:
    body = clean_text(body)
    if body in state.sent_bodies:
        body = clean_text("Quick version: reply YES and I will send the ready draft for approval.")
    state.sent_bodies.append(body)
    return {"action": "send", "body": body, "cta": cta, "rationale": rationale}


def action_mode_body(state: ConversationState, merchant: dict | None) -> str:
    original = state.original_action or {}
    merchant = merchant or {}
    offer = first_active_offer(merchant)
    name = merchant_name(merchant) if merchant else "your business"
    body = original.get("body", "")
    if "patient" in body.lower() or "customer" in body.lower():
        return (
            f"Great. I am drafting it for {name} now. "
            "Reply CONFIRM and I will format the final WhatsApp note for approval before send."
        )
    if offer:
        return (
            f"Done. I will build the next draft around {offer} and keep it approval-only. "
            "Reply CONFIRM to see the final copy."
        )
    return "Done. I will prepare the next draft now and keep it approval-only. Reply CONFIRM to see it."


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", clean_text(text).lower()).strip()


def is_auto_reply(text: str) -> bool:
    return any(re.search(pattern, text) for pattern in AUTO_REPLY_PATTERNS)


def is_hostile_or_stop(text: str) -> bool:
    return any(re.search(pattern, text) for pattern in HOSTILE_PATTERNS)


def is_intent(text: str) -> bool:
    return any(re.search(pattern, text) for pattern in INTENT_PATTERNS)


def is_off_topic(text: str) -> bool:
    return any(re.search(pattern, text) for pattern in OFF_TOPIC_PATTERNS)
