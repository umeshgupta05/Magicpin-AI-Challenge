import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any

from composer import MessageComposer, compose as compose_rules
from reply_handlers import ConversationState, ReplyHandler, is_auto_reply, normalize


VALID_SCOPES = {"category", "merchant", "customer", "trigger"}


@dataclass
class ContextRecord:
    version: int
    payload: dict[str, Any]
    delivered_at: str | None = None
    stored_at: str | None = None


class DatasetFallback:
    def __init__(self, root: str = ".") -> None:
        self.root = Path(root)
        self.loaded = False
        self.categories: dict[str, dict] = {}
        self.merchants: dict[str, dict] = {}
        self.customers: dict[str, dict] = {}
        self.triggers: dict[str, dict] = {}
        if os.getenv("ENABLE_DATASET_FALLBACK", "1").lower() not in {"0", "false", "no"}:
            self._load()

    def _load(self) -> None:
        try:
            self._load_seed_dir(self.root / "dataset")
            self._load_expanded_dir(self.root / "expanded")
            self.loaded = True
        except Exception:
            self.loaded = False

    def _load_seed_dir(self, dataset_dir: Path) -> None:
        if not dataset_dir.exists():
            return
        cat_dir = dataset_dir / "categories"
        for path in cat_dir.glob("*.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            self.categories[data.get("slug", path.stem)] = data
        seed_specs = [
            ("merchants_seed.json", "merchants", "merchant_id", self.merchants),
            ("customers_seed.json", "customers", "customer_id", self.customers),
            ("triggers_seed.json", "triggers", "id", self.triggers),
        ]
        for filename, container, key, target in seed_specs:
            path = dataset_dir / filename
            if not path.exists():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            for item in data.get(container, []):
                target[item[key]] = item

    def _load_expanded_dir(self, expanded_dir: Path) -> None:
        specs = [
            ("categories", "slug", self.categories),
            ("merchants", "merchant_id", self.merchants),
            ("customers", "customer_id", self.customers),
            ("triggers", "id", self.triggers),
        ]
        for dirname, key, target in specs:
            folder = expanded_dir / dirname
            if not folder.exists():
                continue
            for path in folder.glob("*.json"):
                data = json.loads(path.read_text(encoding="utf-8"))
                target[data.get(key, path.stem)] = data

    def get(self, scope: str, context_id: str | None) -> dict | None:
        if not context_id:
            return None
        mapping = {
            "category": self.categories,
            "merchant": self.merchants,
            "customer": self.customers,
            "trigger": self.triggers,
        }.get(scope)
        return mapping.get(context_id) if mapping is not None else None


class VeraEngine:
    def __init__(self) -> None:
        self.started_at = time.time()
        self._lock = RLock()
        self.contexts: dict[tuple[str, str], ContextRecord] = {}
        self.sent_suppression_keys: set[str] = set()
        self.conversations: dict[str, ConversationState] = {}
        self.merchant_auto_counts: dict[tuple[str, str], int] = {}
        self.composer = MessageComposer()
        self.reply_handler = ReplyHandler()
        self.dataset = DatasetFallback(".")
        self.max_actions = min(20, max(1, int(os.getenv("MAX_ACTIONS_PER_TICK", "8"))))

    def healthz(self) -> dict:
        with self._lock:
            counts = {scope: 0 for scope in VALID_SCOPES}
            for scope, _ in self.contexts:
                counts[scope] = counts.get(scope, 0) + 1
        return {
            "status": "ok",
            "uptime_seconds": int(time.time() - self.started_at),
            "contexts_loaded": counts,
        }

    def metadata(self) -> dict:
        return {
            "team_name": os.getenv("TEAM_NAME", "Individual Vera Challenge Submission"),
            "team_members": [x.strip() for x in os.getenv("TEAM_MEMBERS", "Individual participant").split(",") if x.strip()],
            "model": os.getenv("MODEL_DESCRIPTION", "rules-first composer + optional Groq/Gemini polish"),
            "approach": "deterministic trigger ranking and grounded templates; optional fast LLM polish with rules fallback",
            "contact_email": os.getenv("CONTACT_EMAIL", ""),
            "version": os.getenv("BOT_VERSION", "1.0.0"),
            "submitted_at": os.getenv("SUBMITTED_AT", "2026-05-02T00:00:00Z"),
        }

    def push_context(self, scope: str, context_id: str, version: int, payload: dict, delivered_at: str | None) -> tuple[int, dict]:
        if scope not in VALID_SCOPES:
            return 400, {"accepted": False, "reason": "invalid_scope", "details": f"scope must be one of {sorted(VALID_SCOPES)}"}
        if not context_id:
            return 400, {"accepted": False, "reason": "missing_context_id"}
        if not isinstance(payload, dict):
            return 400, {"accepted": False, "reason": "payload_must_be_object"}
        key = (scope, context_id)
        with self._lock:
            current = self.contexts.get(key)
            if current and current.version >= version:
                return 409, {"accepted": False, "reason": "stale_version", "current_version": current.version}
            stored_at = utc_now()
            self.contexts[key] = ContextRecord(version=version, payload=payload, delivered_at=delivered_at, stored_at=stored_at)
        return 200, {"accepted": True, "ack_id": f"ack_{safe_id(context_id)}_v{version}", "stored_at": stored_at}

    async def tick(self, now: str, available_triggers: list[str]) -> dict:
        trigger_ids = list(dict.fromkeys(available_triggers or []))
        ranked = sorted((self._trigger_by_id(tid) for tid in trigger_ids), key=self._trigger_rank, reverse=True)
        actions: list[dict] = []
        for trigger in ranked:
            if not trigger:
                continue
            suppression_key = trigger.get("suppression_key") or f"trigger:{trigger.get('id')}"
            if suppression_key in self.sent_suppression_keys:
                continue
            bundle = self._bundle_for_trigger(trigger)
            if not bundle:
                continue
            category, merchant, customer = bundle
            if trigger.get("scope") == "customer" and not self._customer_send_allowed(customer):
                continue
            composed = await self.composer.compose(category, merchant, trigger, customer)
            action = self._action_from_composed(now, composed, category, merchant, trigger, customer)
            actions.append(action)
            with self._lock:
                self.sent_suppression_keys.add(action["suppression_key"])
                self.conversations[action["conversation_id"]] = ConversationState(
                    conversation_id=action["conversation_id"],
                    merchant_id=action["merchant_id"],
                    customer_id=action.get("customer_id"),
                    trigger_id=action["trigger_id"],
                    original_action=action,
                    sent_bodies=[action["body"]],
                )
            if len(actions) >= self.max_actions:
                break
        return {"actions": actions}

    def reply(
        self,
        conversation_id: str,
        merchant_id: str | None,
        customer_id: str | None,
        from_role: str,
        message: str,
        received_at: str,
        turn_number: int,
    ) -> dict:
        with self._lock:
            state = self.conversations.get(conversation_id)
            if not state:
                state = ConversationState(conversation_id=conversation_id, merchant_id=merchant_id, customer_id=customer_id)
                self.conversations[conversation_id] = state
            if merchant_id and not state.merchant_id:
                state.merchant_id = merchant_id
            if customer_id and not state.customer_id:
                state.customer_id = customer_id
            state.turns.append(
                {
                    "from": from_role,
                    "body": message,
                    "received_at": received_at,
                    "turn_number": turn_number,
                }
            )
            normalized = normalize(message)
            merchant_key = state.merchant_id or merchant_id or "unknown"
            auto_count = 0
            if is_auto_reply(normalized):
                key = (merchant_key, normalized)
                self.merchant_auto_counts[key] = self.merchant_auto_counts.get(key, 0) + 1
                auto_count = self.merchant_auto_counts[key]
        merchant = self._merchant_by_id(state.merchant_id or merchant_id)
        response = self.reply_handler.respond(state, merchant, message, auto_count)
        if response.get("body"):
            with self._lock:
                state.sent_bodies.append(response["body"])
        return response

    def teardown(self) -> dict:
        with self._lock:
            self.contexts.clear()
            self.sent_suppression_keys.clear()
            self.conversations.clear()
            self.merchant_auto_counts.clear()
        return {"ok": True, "cleared_at": utc_now()}

    def _action_from_composed(
        self,
        now: str,
        composed: dict,
        category: dict,
        merchant: dict,
        trigger: dict,
        customer: dict | None,
    ) -> dict:
        merchant_id = merchant.get("merchant_id") or trigger.get("merchant_id")
        customer_id = customer.get("customer_id") if customer else trigger.get("customer_id")
        trigger_id = trigger.get("id") or stable_hash(trigger)
        conversation_id = conversation_id_for(merchant_id, trigger_id, customer_id)
        template_name = template_for(category, trigger, customer)
        body = composed["body"]
        return {
            "conversation_id": conversation_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": composed["send_as"],
            "trigger_id": trigger_id,
            "template_name": template_name,
            "template_params": template_params(body),
            "body": body,
            "cta": composed["cta"],
            "suppression_key": composed["suppression_key"],
            "rationale": composed["rationale"],
        }

    def _bundle_for_trigger(self, trigger: dict) -> tuple[dict, dict, dict | None] | None:
        merchant = self._merchant_by_id(trigger.get("merchant_id"))
        if not merchant:
            return None
        category = self._category_by_slug(merchant.get("category_slug") or trigger.get("payload", {}).get("category"))
        if not category:
            return None
        customer = None
        if trigger.get("customer_id"):
            customer = self._customer_by_id(trigger.get("customer_id"))
        return category, merchant, customer

    def _trigger_by_id(self, trigger_id: str) -> dict | None:
        return self._get("trigger", trigger_id)

    def _merchant_by_id(self, merchant_id: str | None) -> dict | None:
        return self._get("merchant", merchant_id)

    def _category_by_slug(self, slug: str | None) -> dict | None:
        return self._get("category", slug)

    def _customer_by_id(self, customer_id: str | None) -> dict | None:
        return self._get("customer", customer_id)

    def _get(self, scope: str, context_id: str | None) -> dict | None:
        if not context_id:
            return None
        with self._lock:
            record = self.contexts.get((scope, context_id))
            if record:
                return record.payload
        return self.dataset.get(scope, context_id)

    def _trigger_rank(self, trigger: dict | None) -> int:
        if not trigger:
            return -1
        kind = trigger.get("kind", "")
        weights = {
            "supply_alert": 45,
            "regulation_change": 42,
            "active_planning_intent": 40,
            "chronic_refill_due": 38,
            "recall_due": 36,
            "customer_lapsed_hard": 34,
            "perf_dip": 32,
            "renewal_due": 31,
            "gbp_unverified": 29,
            "review_theme_emerged": 28,
            "ipl_match_today": 27,
            "competitor_opened": 26,
            "winback_eligible": 24,
            "research_digest": 23,
            "category_seasonal": 21,
            "seasonal_perf_dip": 20,
            "perf_spike": 18,
            "milestone_reached": 16,
            "trial_followup": 15,
            "wedding_package_followup": 15,
            "festival_upcoming": 10,
            "curious_ask_due": 8,
            "dormant_with_vera": 7,
        }
        return int(trigger.get("urgency") or 0) * 10 + weights.get(kind, 5)

    def _customer_send_allowed(self, customer: dict | None) -> bool:
        if not customer:
            return False
        prefs = customer.get("preferences", {})
        consent = customer.get("consent", {})
        if prefs.get("reminder_opt_in") is False:
            return False
        if consent.get("scope") == [] and not consent.get("opted_in_at"):
            return False
        return True


def template_for(category: dict, trigger: dict, customer: dict | None) -> str:
    prefix = "merchant" if customer or trigger.get("scope") == "customer" else "vera"
    kind = re.sub(r"[^a-z0-9_]+", "_", str(trigger.get("kind", "message")).lower())
    slug = re.sub(r"[^a-z0-9_]+", "_", str(category.get("slug", "general")).lower())
    return f"{prefix}_{slug}_{kind}_v1"


def template_params(body: str) -> list[str]:
    compact = re.sub(r"\s+", " ", body).strip()
    if len(compact) <= 240:
        return [compact]
    return [compact[:240], compact[240:480]]


def conversation_id_for(merchant_id: str | None, trigger_id: str, customer_id: str | None) -> str:
    base = f"{merchant_id or 'merchant'}:{customer_id or 'merchant'}:{trigger_id}"
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:10]
    merchant_part = safe_id(merchant_id or "merchant")[:24]
    trigger_part = safe_id(trigger_id)[:28]
    return f"conv_{merchant_part}_{trigger_part}_{digest}"


def stable_hash(value: Any) -> str:
    return hashlib.sha1(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def safe_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", str(value)).strip("_") or "id"


def utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"


# Convenience export for scripts that want the deterministic challenge function.
__all__ = ["VeraEngine", "compose_rules"]
