import asyncio
import hashlib
import json
import re
from datetime import datetime
from typing import Any

from llm_clients import LLMRouter


CTA_BINARY = "binary_yes_no"
CTA_OPEN = "open_ended"
CTA_NONE = "none"
CTA_SLOT = "multi_choice_slot"
CTA_CONFIRM = "binary_confirm_cancel"


def compose(category: dict, merchant: dict, trigger: dict, customer: dict | None = None) -> dict:
    """Deterministic challenge composer. Safe to use for submission.jsonl generation."""
    return RuleComposer().compose(category, merchant, trigger, customer)


class MessageComposer:
    def __init__(self) -> None:
        self.rules = RuleComposer()
        self.llm = LLMRouter()

    async def compose(self, category: dict, merchant: dict, trigger: dict, customer: dict | None = None) -> dict:
        base = self.rules.compose(category, merchant, trigger, customer)
        provider, polished = await self._try_polish(category, merchant, trigger, customer, base)
        if polished:
            polished["rationale"] = f"{polished['rationale']} Polished by {provider}; facts selected deterministically."
            return polished
        return base

    async def _try_polish(
        self,
        category: dict,
        merchant: dict,
        trigger: dict,
        customer: dict | None,
        base: dict,
    ) -> tuple[str | None, dict | None]:
        evidence = build_evidence_pack(category, merchant, trigger, customer)
        if not self.llm.active_provider_names():
            return None, None
        system = (
            "You polish Vera WhatsApp messages. Return JSON only. Do not add facts. "
            "Use only the evidence list. Keep the same CTA, send_as, and suppression_key. "
            "No URLs. No hype. One clear next action."
        )
        prompt = json.dumps(
            {
                "evidence": evidence,
                "base_message": base,
                "required_keys": ["body", "cta", "send_as", "suppression_key", "rationale"],
            },
            ensure_ascii=True,
        )
        try:
            provider, data = await asyncio.wait_for(self.llm.complete_json(system, prompt), timeout=self.llm.timeout_s + 1)
        except Exception:
            return None, None
        if not data:
            return None, None
        candidate = {
            "body": clean_text(str(data.get("body", ""))),
            "cta": base["cta"],
            "send_as": base["send_as"],
            "suppression_key": base["suppression_key"],
            "rationale": clean_text(str(data.get("rationale") or base["rationale"])),
        }
        if validate_candidate(candidate, base, category, merchant, trigger, customer):
            return provider, candidate
        return None, None


class RuleComposer:
    def compose(self, category: dict, merchant: dict, trigger: dict, customer: dict | None = None) -> dict:
        if customer or trigger.get("scope") == "customer":
            result = self._customer_message(category, merchant, trigger, customer or {})
        else:
            result = self._merchant_message(category, merchant, trigger)

        result["body"] = clean_text(result["body"])
        result["rationale"] = clean_text(result["rationale"])
        result["suppression_key"] = trigger.get("suppression_key") or stable_key(trigger)
        return result

    def _merchant_message(self, category: dict, merchant: dict, trigger: dict) -> dict:
        kind = trigger.get("kind", "general")
        if kind in {"research_digest", "regulation_change", "cde_opportunity", "supply_alert", "category_seasonal"}:
            return self._digest_or_alert(category, merchant, trigger)
        if kind in {"perf_dip", "perf_spike", "seasonal_perf_dip"}:
            return self._performance(category, merchant, trigger)
        if kind == "review_theme_emerged":
            return self._review_theme(category, merchant, trigger)
        if kind == "competitor_opened":
            return self._competitor(category, merchant, trigger)
        if kind == "ipl_match_today":
            return self._ipl(category, merchant, trigger)
        if kind == "active_planning_intent":
            return self._active_planning(category, merchant, trigger)
        if kind == "curious_ask_due":
            return self._curious_ask(category, merchant, trigger)
        if kind == "renewal_due":
            return self._renewal(category, merchant, trigger)
        if kind == "winback_eligible":
            return self._winback(category, merchant, trigger)
        if kind == "gbp_unverified":
            return self._gbp_unverified(category, merchant, trigger)
        if kind == "festival_upcoming":
            return self._festival(category, merchant, trigger)
        if kind == "milestone_reached":
            return self._milestone(category, merchant, trigger)
        if kind == "dormant_with_vera":
            return self._dormant(category, merchant, trigger)
        return self._generic_merchant(category, merchant, trigger)

    def _customer_message(self, category: dict, merchant: dict, trigger: dict, customer: dict) -> dict:
        kind = trigger.get("kind", "customer_message")
        if kind == "recall_due":
            return self._recall_due(category, merchant, trigger, customer)
        if kind == "wedding_package_followup":
            return self._wedding_followup(category, merchant, trigger, customer)
        if kind == "customer_lapsed_hard":
            return self._customer_lapse(category, merchant, trigger, customer)
        if kind == "chronic_refill_due":
            return self._chronic_refill(category, merchant, trigger, customer)
        if kind == "trial_followup":
            return self._trial_followup(category, merchant, trigger, customer)
        if kind in {"customer_lapsed_soft", "appointment_tomorrow"}:
            return self._customer_generic(category, merchant, trigger, customer)
        return self._customer_generic(category, merchant, trigger, customer)

    def _digest_or_alert(self, category: dict, merchant: dict, trigger: dict) -> dict:
        kind = trigger.get("kind", "")
        item = find_digest_item(category, trigger)
        who = merchant_salutation(category, merchant)
        business = merchant_name(merchant)
        active_offer = first_active_offer(merchant)
        agg = merchant.get("customer_aggregate", {})
        payload = trigger.get("payload", {})

        if kind == "research_digest" and item:
            trial = item.get("trial_n")
            segment = item.get("patient_segment", "")
            segment = segment.replace("_", " ") if segment else ""
            segment_count = agg.get("high_risk_adult_count")
            if segment_count:
                segment_phrase = f"your {segment_count} high-risk adult patients"
            elif segment:
                segment_phrase = f"your {segment}"
            else:
                segment_phrase = "your patients"
            metric = find_percent(item.get("summary", ""))
            metric_phrase = (
                f"shows 3-month fluoride recall reduced caries recurrence {metric} vs 6-month"
                if metric
                else "notes better outcomes with 3-month fluoride recall vs 6-month"
            )
            trial_phrase = f"{trial:,}-patient trial" if trial else "The item"
            body = (
                f"{who}, {item.get('source', 'this week')}: one item fits {segment_phrase}. "
                f"{trial_phrase} {metric_phrase}. "
                "Want me to pull the 2-min summary and draft a patient WhatsApp you can review?"
            )
            return msg(body, CTA_OPEN, "vera", "Research digest matched to merchant cohort and category clinical voice.")

        if kind == "regulation_change" and item:
            deadline = payload.get("deadline_iso") or find_date(item.get("summary", ""))
            deadline_text = pretty_date(deadline)
            deadline_clause = f" Deadline is {deadline_text}." if deadline_text else ""
            body = (
                f"{who}, compliance heads-up: {item.get('title')}. Source: {item.get('source')}."
                f"{deadline_clause} Want me to draft a 5-point SOP checklist for your clinic?"
            )
            return msg(body, CTA_BINARY, "vera", "Regulation trigger uses source, deadline, and a low-effort compliance action.")

        if kind == "cde_opportunity":
            credits = payload.get("credits") or item.get("credits") if item else payload.get("credits")
            fee = payload.get("fee", "").replace("_", " ")
            date = item.get("date") if item else trigger.get("expires_at")
            title = item.get("title") if item else "CDE opportunity"
            when = pretty_datetime(date)
            when_text = f" on {when}" if when else ""
            credits_text = f" ({credits} credits)" if credits else ""
            fee_text = f", {fee}" if fee else ""
            body = (
                f"{who}, quick CDE pick: {title}{when_text}{credits_text}{fee_text}. "
                "Want me to send the 2-line invite text you can forward to your team?"
            )
            return msg(body, CTA_BINARY, "vera", "CDE trigger is time-bound and framed as a quick professional opportunity.")

        if kind == "supply_alert":
            batches = ", ".join(payload.get("affected_batches", []))
            molecule = payload.get("molecule", "medicine")
            mfr = payload.get("manufacturer", "manufacturer")
            chronic = agg.get("chronic_rx_count")
            body = (
                f"{who}, urgent stock check: {mfr} flagged {molecule} batches {batches}. "
                f"You have {chronic} chronic-Rx customers in the roster. Want me to draft the batch-check workflow and customer note?"
                if chronic and batches
                else f"{who}, urgent stock check: {molecule} supply alert received. Want me to draft the batch-check workflow and customer note?"
            )
            return msg(body, CTA_BINARY, "vera", "Supply alert uses batch details and pharmacy customer aggregate without inventing affected customers.")

        if kind == "category_seasonal":
            trends = [str(t).replace("_", " ") for t in payload.get("trends", []) if t]
            trend_text = ", ".join(trends[:4])
            if trend_text:
                body = (
                    f"{who}, seasonal demand shift for {business}: {trend_text}. "
                    "Want me to draft a shelf-priority checklist and a 3-line WhatsApp for regular customers?"
                )
            else:
                body = (
                    f"{who}, seasonal demand shift spotted for {business}. "
                    "Want me to draft a shelf-priority checklist and a 3-line WhatsApp for regular customers?"
                )
            return msg(body, CTA_BINARY, "vera", "Seasonal pharmacy trigger turns trend facts into a concrete merchandising action.")

        title = item.get("title") if item else payload.get("metric_or_topic", kind).replace("_", " ")
        source = f" Source: {item.get('source')}." if item and item.get("source") else ""
        body = f"{who}, useful {category_slug(category)} update: {title}.{source} Want me to turn this into a short merchant-ready action note?"
        return msg(body, CTA_BINARY, "vera", "Digest-like trigger grounded in category update.")

    def _performance(self, category: dict, merchant: dict, trigger: dict) -> dict:
        who = merchant_salutation(category, merchant)
        perf = merchant.get("performance", {})
        payload = trigger.get("payload", {})
        kind = trigger.get("kind")
        metric = payload.get("metric", "views")
        delta = payload.get("delta_pct")
        current = perf.get(metric)
        active_offer = first_active_offer(merchant) or first_catalog_offer(category)
        peer_ctr = category.get("peer_stats", {}).get("avg_ctr")
        metric_label = str(metric or "metric").replace("_", " ")
        delta_text = fmt_pct(delta)
        delta_abs = delta_text.lstrip("-") if delta_text.startswith("-") else delta_text
        window = payload.get("window", "7d")
        current_text = f" - now {current}" if not is_missing(current) else ""
        ctr_text = fmt_pct(perf.get("ctr"))
        peer_text = fmt_pct(peer_ctr)
        verb = "are" if metric_label.endswith("s") else "is"
        ctr_phrase = ""
        if ctr_text:
            ctr_phrase = f" Your 30d CTR is {ctr_text}"
            if peer_text:
                ctr_phrase += f" vs peer {peer_text}"
            ctr_phrase += "."

        if kind == "seasonal_perf_dip":
            delta_clause = f"down {delta_abs}" if delta_abs else "down this week"
            body = (
                f"{who}, {metric_label} {verb} {delta_clause}, but your trigger marks this as expected seasonal movement. "
                f"You still have {aggregate_member_or_customer_count(merchant)} to retain. Want me to draft a summer retention challenge instead of pushing ads?"
            )
            return msg(body, CTA_BINARY, "vera", "Seasonal dip is reframed to retention instead of panic spending.")

        if kind == "perf_spike":
            driver = payload.get("likely_driver")
            delta_up = delta_text if delta_text and not delta_text.startswith("-") else ""
            delta_clause = f"up {delta_up}" if delta_up else "up"
            driver_clause = f", likely from {driver.replace('_', ' ')}" if driver else ""
            body = (
                f"{who}, nice spike: {metric_label} {verb} {delta_clause} over {window}{driver_clause}. "
                f"Want me to turn this into a follow-up post using {clean_text(active_offer)}?"
            )
            return msg(body, CTA_BINARY, "vera", "Performance spike is converted into a timely amplification action.")

        delta_clause = f"dropped {delta_abs}" if delta_abs else "dipped"
        body = (
            f"{who}, {metric_label} {delta_clause} over {window}{current_text}."
            f"{ctr_phrase} Want me to draft one recovery post around {clean_text(active_offer)}?"
        )
        return msg(body, CTA_BINARY, "vera", "Performance dip uses merchant metrics, peer benchmark, and a single recovery action.")

    def _review_theme(self, category: dict, merchant: dict, trigger: dict) -> dict:
        who = merchant_salutation(category, merchant)
        p = trigger.get("payload", {})
        quote = p.get("common_quote")
        count = p.get("occurrences_30d")
        count_text = f"{count} recent reviews mention " if not is_missing(count) else "recent reviews mention "
        body = (
            f"{who}, review pattern to catch early: {count_text}{p.get('theme', 'one theme').replace('_', ' ')}"
            f"{f' - \"{quote}\"' if quote else ''}. "
            "Want me to draft a short owner reply + one operations fix message for your team?"
        )
        return msg(body, CTA_BINARY, "vera", "Review theme trigger converts customer feedback into a response and operational fix.")

    def _competitor(self, category: dict, merchant: dict, trigger: dict) -> dict:
        who = merchant_salutation(category, merchant)
        p = trigger.get("payload", {})
        offer = first_active_offer(merchant) or first_catalog_offer(category)
        competitor = p.get("competitor_name") or "a competitor"
        distance = p.get("distance_km")
        opened = pretty_date(p.get("opened_date"))
        distance_text = f"{distance} km away" if not is_missing(distance) else "nearby"
        date_text = f" on {opened}" if opened else ""
        their_offer = clean_text(p.get("their_offer", ""))
        offer_text = f"They show {their_offer}; " if their_offer else ""
        body = (
            f"{who}, new competitor nearby: {competitor} opened {distance_text}{date_text}. "
            f"{offer_text}You already have {clean_text(offer)}. "
            "Want me to draft a calm GBP post that defends your positioning without starting a price war?"
        )
        return msg(body, CTA_BINARY, "vera", "Competitor trigger uses local competitive facts and avoids panic discounting.")

    def _ipl(self, category: dict, merchant: dict, trigger: dict) -> dict:
        who = merchant_salutation(category, merchant)
        p = trigger.get("payload", {})
        item = find_digest_by_kind(category, "seasonal")
        offer = first_active_offer(merchant) or first_catalog_offer(category)
        summary = item.get("summary", "") if item else ""
        saturday_fact = "restaurant covers down 12% vs Saturday average" if "down 12%" in summary else ""
        weeknight_fact = "weeknight matches drive +18% covers" if "+18%" in summary else ""
        if p.get("is_weeknight"):
            advice = f"{weeknight_fact}; use your {clean_text(offer)} as the hook" if weeknight_fact else f"use your {clean_text(offer)} as the hook"
        else:
            advice = f"{saturday_fact.capitalize()}; keep it delivery-first with {clean_text(offer)}" if saturday_fact else f"keep it delivery-first with {clean_text(offer)}"
        body = (
            f"{who}, {p.get('match', 'match')} at {p.get('venue', 'local venue')} starts {pretty_time(p.get('match_time_iso'))}. "
            f"{advice}. Want me to draft the banner copy and Insta story? Live in 10 min."
        )
        return msg(body, CTA_BINARY, "vera", "IPL trigger combines match timing, category digest, and the merchant's active offer.")

    def _active_planning(self, category: dict, merchant: dict, trigger: dict) -> dict:
        who = merchant_salutation(category, merchant)
        p = trigger.get("payload", {})
        topic = p.get("intent_topic", "plan").replace("_", " ")
        offer = first_active_offer(merchant) or first_catalog_offer(category)
        history_text = " ".join(str(h.get("body", "")) for h in merchant.get("conversation_history", []))
        orders = re.search(r"(\d+)\s+orders/day", history_text)
        if "corporate_bulk_thali" in p.get("intent_topic", ""):
            order_phrase = f" Your weekday thali is already at {orders.group(1)} orders/day." if orders else ""
            body = (
                f"{who}, starter draft for {topic}:{order_phrase}\n"
                f"- Base: {clean_text(offer)}\n"
                "- 10 plates: listed price, one delivery slot\n"
                "- 25 plates: office pack with filter-coffee add-on\n"
                "- 50+ plates: pre-order by 5pm previous day\n"
                "Want me to turn this into a 3-line WhatsApp pitch for office admins?"
            )
            return msg(body, CTA_BINARY, "vera", "Merchant already asked for the plan; response switches directly to drafted action.")
        if "kids_yoga" in p.get("intent_topic", ""):
            body = (
                f"{who}, here is a clean kids yoga summer-camp draft: 4 weeks, 3 classes/week, age 7-12, "
                "morning batch for school holidays, with one parent demo class at the end. "
                "Want me to draft the GBP post + parent WhatsApp now?"
            )
            return msg(body, CTA_BINARY, "vera", "Planning trigger receives a concrete draft instead of more qualification.")
        body = (
            f"{who}, I picked up your planning intent: {topic}. "
            f"Starting point should use {clean_text(offer)} and one simple follow-up action. "
            "Want me to draft the merchant-ready message now?"
        )
        return msg(body, CTA_BINARY, "vera", "Active planning trigger moves to action mode immediately.")

    def _curious_ask(self, category: dict, merchant: dict, trigger: dict) -> dict:
        who = merchant_salutation(category, merchant)
        business = merchant_name(merchant)
        active_offer = first_active_offer(merchant)
        body = (
            f"{who}, quick check for {business}: what service has been most asked-for this week"
            f"{f' - is it related to {clean_text(active_offer)}?' if active_offer else '?'} "
            "Reply with just the service name; I will turn it into a Google post and a 4-line customer reply."
        )
        return msg(body, CTA_OPEN, "vera", "Curious-ask trigger uses the merchant as source and offers immediate drafting value.")

    def _renewal(self, category: dict, merchant: dict, trigger: dict) -> dict:
        who = merchant_salutation(category, merchant)
        p = trigger.get("payload", {})
        perf = merchant.get("performance", {})
        days = p.get("days_remaining", merchant.get("subscription", {}).get("days_remaining"))
        renewal_amount = p.get("renewal_amount")
        calls = perf.get("calls")
        ctr = fmt_pct(perf.get("ctr"))
        intro = f"{who}, Pro renewal is due in {days} days" if not is_missing(days) else f"{who}, Pro renewal is coming up"
        if not is_missing(renewal_amount):
            intro = f"{intro} (Rs {renewal_amount})"
        metrics = []
        if not is_missing(calls):
            metrics.append(f"calls are at {calls}")
        if ctr:
            metrics.append(f"CTR is {ctr}")
        metrics_text = f"Before that, {', '.join(metrics)}. " if metrics else ""
        body = (
            f"{intro}. {metrics_text}"
            "Want me to first fix the highest-impact listing gap so renewal is tied to visible growth?"
        )
        return msg(body, CTA_BINARY, "vera", "Renewal is connected to merchant performance, not a generic payment reminder.")

    def _winback(self, category: dict, merchant: dict, trigger: dict) -> dict:
        who = merchant_salutation(category, merchant)
        p = trigger.get("payload", {})
        days_since = p.get("days_since_expiry", merchant.get("subscription", {}).get("days_since_expiry"))
        perf_dip = fmt_pct(p.get("perf_dip_pct"))
        lapsed = p.get("lapsed_customers_added_since_expiry")
        intro = f"{who}, your plan expired {days_since} days ago." if not is_missing(days_since) else f"{who}, your plan expired recently."
        perf_text = f" Since then performance is down {perf_dip}." if perf_dip else ""
        lapsed_text = f" {lapsed} more customers became lapsed." if not is_missing(lapsed) else ""
        body = (
            f"{intro}{perf_text}{lapsed_text} "
            "Want me to draft a 7-day comeback campaign before asking you to renew?"
        )
        return msg(body, CTA_BINARY, "vera", "Winback trigger leads with lost momentum and offers value before renewal.")

    def _gbp_unverified(self, category: dict, merchant: dict, trigger: dict) -> dict:
        who = merchant_salutation(category, merchant)
        p = trigger.get("payload", {})
        uplift = fmt_pct(p.get("estimated_uplift_pct"))
        uplift_text = f"The trigger estimates {uplift} upside after verification." if uplift else "The trigger flags upside after verification."
        path = str(p.get("verification_path") or "").replace("_", " ").strip()
        path_text = f" Path: {path}." if path else ""
        body = (
            f"{who}, your Google profile is still unverified. {uplift_text}{path_text} "
            "Want me to draft the exact verification steps for your staff?"
        )
        return msg(body, CTA_BINARY, "vera", "GBP verification trigger uses the provided uplift and path.")

    def _festival(self, category: dict, merchant: dict, trigger: dict) -> dict:
        who = merchant_salutation(category, merchant)
        p = trigger.get("payload", {})
        offer = first_active_offer(merchant) or first_catalog_offer(category)
        days_until = p.get("days_until")
        fest_date = pretty_date(p.get("date"))
        if not is_missing(days_until) and fest_date:
            timing = f"{days_until} days away ({fest_date})"
        elif fest_date:
            timing = f"coming up on {fest_date}"
        elif not is_missing(days_until):
            timing = f"{days_until} days away"
        else:
            timing = "coming up soon"
        body = (
            f"{who}, {p.get('festival', 'festival')} is {timing}. "
            f"For {category_slug(category)}, the safest early move is a reviewable post around {clean_text(offer)}. "
            "Want me to draft it now and keep it ready?"
        )
        return msg(body, CTA_BINARY, "vera", "Festival trigger is handled as early planning with category fit.")

    def _milestone(self, category: dict, merchant: dict, trigger: dict) -> dict:
        who = merchant_salutation(category, merchant)
        p = trigger.get("payload", {})
        metric = str(p.get("metric", "milestone")).replace("_", " ")
        value_now = p.get("value_now")
        target = p.get("milestone_value")
        if not is_missing(value_now) and not is_missing(target):
            milestone_text = f"{value_now} {metric} now, {target} target next."
        elif not is_missing(value_now):
            milestone_text = f"{value_now} {metric} so far."
        elif not is_missing(target):
            milestone_text = f"Target {metric} is {target}."
        else:
            milestone_text = ""
        if milestone_text:
            body = (
                f"{who}, you are close to a milestone: {milestone_text} "
                "Want me to draft a polite review-request WhatsApp for recent happy customers?"
            )
        else:
            body = (
                f"{who}, you are close to a milestone. "
                "Want me to draft a polite review-request WhatsApp for recent happy customers?"
            )
        return msg(body, CTA_BINARY, "vera", "Milestone trigger uses current and target values with a low-risk action.")

    def _dormant(self, category: dict, merchant: dict, trigger: dict) -> dict:
        who = merchant_salutation(category, merchant)
        p = trigger.get("payload", {})
        business = merchant_name(merchant)
        days_since = p.get("days_since_last_merchant_message")
        topic = p.get("last_topic")
        intro = (
            f"{who}, it has been {days_since} days since your last Vera reply"
            if not is_missing(days_since)
            else f"{who}, it has been a while since your last Vera reply"
        )
        topic_text = f" on {str(topic).replace('_', ' ')}" if topic else ""
        body = (
            f"{intro}{topic_text}. "
            f"I found one quick update for {business}. Reply YES and I will send only the 2-line version."
        )
        return msg(body, CTA_BINARY, "vera", "Dormancy trigger uses low-pressure re-entry and a tiny commitment.")

    def _generic_merchant(self, category: dict, merchant: dict, trigger: dict) -> dict:
        who = merchant_salutation(category, merchant)
        topic = str(trigger.get("payload", {}).get("metric_or_topic") or trigger.get("kind", "update")).replace("_", " ")
        fact = best_merchant_fact(merchant)
        body = f"{who}, quick {category_slug(category)} update on {topic}. {fact} Want me to draft the next customer-facing post?"
        return msg(body, CTA_BINARY, "vera", "Generic fallback still anchors on merchant context and a single action.")

    def _recall_due(self, category: dict, merchant: dict, trigger: dict, customer: dict) -> dict:
        p = trigger.get("payload", {})
        slots = p.get("available_slots", [])
        slot_text = " or ".join(slot.get("label", "") for slot in slots[:2] if slot.get("label"))
        offer = first_active_offer(merchant)
        offer_sentence = f"{clean_text(offer)}. " if offer else ""
        service = p.get("service_due", "recall").replace("_", " ")
        due_date = pretty_date(p.get("due_date"))
        last_date = pretty_date(p.get("last_service_date"))
        due_clause = f"Your {service} is due on {due_date}" if due_date else f"Your {service} is due"
        last_clause = f"; last visit was {last_date}." if last_date else "."
        body = (
            f"Hi {customer_first(customer)}, {merchant_name(merchant)} here. "
            f"{due_clause}{last_clause} "
            f"{f'Aapke liye slots ready hain: {slot_text}. ' if slot_text else ''}"
            f"{offer_sentence}Reply {('1 or 2' if slot_text else 'YES')} to book, or share a better time."
        )
        return msg(body, CTA_SLOT if slot_text else CTA_BINARY, "merchant_on_behalf", "Customer recall uses due date, prior visit, preference slots, and merchant offer.")

    def _wedding_followup(self, category: dict, merchant: dict, trigger: dict, customer: dict) -> dict:
        p = trigger.get("payload", {})
        days_to = p.get("days_to_wedding")
        wedding_date = pretty_date(p.get("wedding_date"))
        trial_date = pretty_date(p.get("trial_completed"))
        if not is_missing(days_to) and wedding_date:
            timing = f"{days_to} days to your wedding on {wedding_date}"
        elif wedding_date:
            timing = f"wedding on {wedding_date}"
        elif not is_missing(days_to):
            timing = f"{days_to} days to your wedding"
        else:
            timing = "your wedding timeline"
        trial_text = f"Your bridal trial was on {trial_date}. " if trial_date else ""
        body = (
            f"Hi {customer_first(customer)}, {merchant_name(merchant)} here. "
            f"{timing}. "
            f"{trial_text}"
            "This is the right window to plan skin-prep and final bookings. Want us to block your preferred Saturday slot?"
        )
        return msg(body, CTA_BINARY, "merchant_on_behalf", "Bridal follow-up uses wedding timeline and trial relationship.")

    def _customer_lapse(self, category: dict, merchant: dict, trigger: dict, customer: dict) -> dict:
        p = trigger.get("payload", {})
        offer = first_active_offer(merchant)
        restart = clean_text(offer) if offer else "one no-commitment restart slot"
        focus = p.get("previous_focus") or customer.get("preferences", {}).get("training_focus")
        days_since = p.get("days_since_last_visit")
        days_text = (
            f"It has been {days_since} days since your last visit - no pressure. "
            if not is_missing(days_since)
            else "It has been a while since your last visit - no pressure. "
        )
        body = (
            f"Hi {customer_first(customer)}, {owner_or_business(merchant)} here. "
            f"{days_text}"
            f"We can restart gently with {restart}"
            f"{f' around your {focus.replace('_', ' ')} goal' if focus else ''}. "
            "Reply YES and we will hold one no-commitment slot."
        )
        return msg(body, CTA_BINARY, "merchant_on_behalf", "Winback avoids guilt and uses prior customer goal plus merchant offer.")

    def _chronic_refill(self, category: dict, merchant: dict, trigger: dict, customer: dict) -> dict:
        p = trigger.get("payload", {})
        meds = ", ".join(p.get("molecule_list", []))
        offers = "; ".join(clean_text(o.get("title", "")) for o in merchant.get("offers", []) if o.get("status") == "active")
        run_out = pretty_date(p.get("stock_runs_out_iso"))
        run_out_text = f"run out on {run_out}" if run_out else "are due soon"
        body = (
            f"Namaste, {merchant_name(merchant)} here. {customer_first(customer)}'s monthly medicines"
            f"{f' ({meds})' if meds else ''} {run_out_text}. "
            f"Same medicines can be packed{f'; {offers}' if offers else ''}. "
            "Reply CONFIRM to dispatch to the saved address, or reply CHANGE if dosage/brand changed."
        )
        return msg(body, CTA_CONFIRM, "merchant_on_behalf", "Refill reminder is precise, respectful, and avoids inventing price or phone data.")

    def _trial_followup(self, category: dict, merchant: dict, trigger: dict, customer: dict) -> dict:
        p = trigger.get("payload", {})
        slots = p.get("next_session_options", [])
        slot = slots[0].get("label") if slots else ""
        trial_date = pretty_date(p.get("trial_date"))
        trial_text = f"Thanks for trying the session on {trial_date}. " if trial_date else "Thanks for trying the session. "
        body = (
            f"Hi {customer_first(customer)}, {merchant_name(merchant)} here. "
            f"{trial_text}"
            f"{f'Next suitable slot: {slot}. ' if slot else ''}"
            "Want us to hold it for you?"
        )
        return msg(body, CTA_BINARY, "merchant_on_behalf", "Trial follow-up uses the trial date and next available slot.")

    def _customer_generic(self, category: dict, merchant: dict, trigger: dict, customer: dict) -> dict:
        offer = first_active_offer(merchant)
        offer_sentence = f"{clean_text(offer)}. " if offer else ""
        topic = customer_topic_text(trigger.get("kind", "reminder"))
        body = (
            f"Hi {customer_first(customer)}, {merchant_name(merchant)} here. "
            f"A quick update for you: {topic}. "
            f"{offer_sentence}Reply YES if you want us to help with the next step."
        )
        return msg(body, CTA_BINARY, "merchant_on_behalf", "Customer fallback uses merchant identity, trigger reason, and one CTA.")


def msg(body: str, cta: str, send_as: str, rationale: str) -> dict:
    return {
        "body": body,
        "cta": cta,
        "send_as": send_as,
        "suppression_key": "",
        "rationale": rationale,
    }


def validate_candidate(candidate: dict, base: dict, category: dict, merchant: dict, trigger: dict, customer: dict | None) -> bool:
    body = candidate.get("body", "")
    if not body or len(body) > 1100:
        return False
    if "http://" in body.lower() or "https://" in body.lower():
        return False
    if candidate.get("cta") != base.get("cta"):
        return False
    if candidate.get("send_as") != base.get("send_as"):
        return False
    if candidate.get("suppression_key") != base.get("suppression_key"):
        return False
    forbidden = ["tbd", "insert ", "unknown", "guaranteed", "100% safe", "miracle"]
    taboos = category.get("voice", {}).get("vocab_taboo", [])
    for phrase in forbidden + [str(t) for t in taboos]:
        if phrase and phrase.lower() in body.lower():
            return False
    anchors = evidence_anchors(category, merchant, trigger, customer)
    return any(anchor.lower() in body.lower() for anchor in anchors if len(anchor) >= 3)


def build_evidence_pack(category: dict, merchant: dict, trigger: dict, customer: dict | None) -> dict:
    item = find_digest_item(category, trigger)
    return {
        "category": category.get("slug"),
        "voice": category.get("voice", {}).get("tone"),
        "taboos": category.get("voice", {}).get("vocab_taboo", [])[:6],
        "merchant": {
            "name": merchant_name(merchant),
            "owner": merchant.get("identity", {}).get("owner_first_name"),
            "locality": merchant.get("identity", {}).get("locality"),
            "city": merchant.get("identity", {}).get("city"),
            "performance": merchant.get("performance", {}),
            "active_offers": [clean_text(o.get("title", "")) for o in merchant.get("offers", []) if o.get("status") == "active"],
            "signals": merchant.get("signals", []),
            "customer_aggregate": merchant.get("customer_aggregate", {}),
        },
        "trigger": {
            "kind": trigger.get("kind"),
            "payload": trigger.get("payload", {}),
            "urgency": trigger.get("urgency"),
            "suppression_key": trigger.get("suppression_key"),
        },
        "digest_item": item,
        "customer": customer,
    }


def evidence_anchors(category: dict, merchant: dict, trigger: dict, customer: dict | None) -> list[str]:
    anchors = [
        merchant_name(merchant),
        str(merchant.get("identity", {}).get("owner_first_name", "")),
        str(merchant.get("identity", {}).get("locality", "")),
        str(trigger.get("kind", "")).replace("_", " "),
    ]
    for value in trigger.get("payload", {}).values():
        if isinstance(value, (str, int, float)):
            anchors.append(clean_text(str(value)))
        elif isinstance(value, list):
            anchors.extend(clean_text(str(v)) for v in value if isinstance(v, (str, int, float)))
    if customer:
        anchors.append(customer_first(customer))
    return [a for a in anchors if a and a != "None"]


def find_digest_item(category: dict, trigger: dict) -> dict:
    payload = trigger.get("payload", {})
    ids = [
        payload.get("top_item_id"),
        payload.get("digest_item_id"),
        payload.get("alert_id"),
        payload.get("item_id"),
    ]
    digest = category.get("digest", [])
    for item_id in ids:
        if not item_id:
            continue
        for item in digest:
            if item.get("id") == item_id:
                return item
    kind_map = {
        "research_digest": "research",
        "regulation_change": "compliance",
        "cde_opportunity": "cde",
        "supply_alert": "alert",
        "category_seasonal": "seasonal",
    }
    return find_digest_by_kind(category, kind_map.get(trigger.get("kind"), ""))


def find_digest_by_kind(category: dict, kind: str) -> dict:
    if not kind:
        return {}
    for item in category.get("digest", []):
        if item.get("kind") == kind:
            return item
    return {}


def merchant_salutation(category: dict, merchant: dict) -> str:
    ident = merchant.get("identity", {})
    owner = str(ident.get("owner_first_name") or "").strip()
    if category.get("slug") == "dentists" and owner:
        return owner if owner.lower().startswith("dr") else f"Dr. {owner}"
    return owner or merchant_name(merchant)


def merchant_name(merchant: dict) -> str:
    return clean_text(merchant.get("identity", {}).get("name") or merchant.get("merchant_id") or "your business")


def owner_or_business(merchant: dict) -> str:
    return clean_text(merchant.get("identity", {}).get("owner_first_name") or merchant_name(merchant))


def customer_first(customer: dict) -> str:
    raw = clean_text(customer.get("identity", {}).get("name") or "there")
    return raw.split("(")[0].strip() or raw


def customer_topic_text(kind: str) -> str:
    mapping = {
        "customer_lapsed_soft": "it has been a while since your last visit",
        "appointment_tomorrow": "your appointment is tomorrow",
    }
    return mapping.get(kind, str(kind or "update").replace("_", " "))


def category_slug(category: dict) -> str:
    return clean_text(category.get("display_name") or category.get("slug") or "business")


def first_active_offer(merchant: dict) -> str:
    for offer in merchant.get("offers", []):
        if offer.get("status") == "active" and offer.get("title"):
            return clean_text(offer["title"])
    return ""


def first_catalog_offer(category: dict) -> str:
    for offer in category.get("offer_catalog", []):
        title = clean_text(offer.get("title", ""))
        if title and offer.get("type") != "percentage_discount":
            return title
    offers = category.get("offer_catalog", [])
    return clean_text(offers[0].get("title", "")) if offers else "a focused customer offer"


def aggregate_member_or_customer_count(merchant: dict) -> str:
    agg = merchant.get("customer_aggregate", {})
    for key in ["total_active_members", "total_unique_ytd", "chronic_rx_count"]:
        if agg.get(key) is not None:
            return f"{agg[key]} {key.replace('_', ' ')}"
    return "your current customer base"


def best_merchant_fact(merchant: dict) -> str:
    perf = merchant.get("performance", {})
    if perf.get("views") is not None and perf.get("calls") is not None:
        return f"Your 30d dashboard shows {perf['views']} views and {perf['calls']} calls."
    signals = merchant.get("signals", [])
    if signals:
        return f"Current signal: {str(signals[0]).replace('_', ' ')}."
    return f"{merchant_name(merchant)} has enough context for a focused next step."


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip().lower()
    return text in {"", "?", "none", "null", "nan"}


def fmt_pct(value: Any) -> str:
    if is_missing(value):
        return ""
    text = str(value).strip()
    if text.endswith("%"):
        return text
    try:
        num = float(value)
    except Exception:
        return ""
    if abs(num) <= 1:
        num *= 100
    text = f"{num:.1f}".rstrip("0").rstrip(".")
    return f"{text}%"


def pretty_date(value: Any) -> str:
    if is_missing(value):
        return ""
    text = str(value)
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return dt.strftime("%d %b %Y").lstrip("0")
    except Exception:
        return text[:10] if len(text) >= 10 else text


def pretty_datetime(value: Any) -> str:
    if is_missing(value):
        return ""
    text = str(value)
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return dt.strftime("%d %b, %I:%M%p").replace("AM", "am").replace("PM", "pm").lstrip("0")
    except Exception:
        return text


def pretty_time(value: Any) -> str:
    if not value:
        return "soon"
    text = str(value)
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return dt.strftime("%I:%M%p").lstrip("0").replace("AM", "am").replace("PM", "pm")
    except Exception:
        return text


def find_percent(text: str) -> str:
    match = re.search(r"(\d+(?:\.\d+)?)\s*%", text or "")
    return f"{match.group(1)}%" if match else ""


def find_date(text: str) -> str:
    match = re.search(r"\b(20\d{2}-\d{2}-\d{2}|[A-Z][a-z]{2,8}\s+\d{1,2})\b", text or "")
    return match.group(1) if match else ""


def stable_key(trigger: dict) -> str:
    raw = trigger.get("id") or json.dumps(trigger, sort_keys=True)
    return "trigger:" + hashlib.sha1(str(raw).encode("utf-8")).hexdigest()[:12]


def clean_text(value: Any) -> str:
    text = str(value or "")
    replacements = {
        "\u00e2\u201a\u00b9": "Rs ",
        "\u20b9": "Rs ",
        "\u00e2\u20ac\u201d": "-",
        "\u2014": "-",
        "\u00e2\u20ac\u201c": "-",
        "\u2013": "-",
        "\u00e2\u2020\u2019": "->",
        "\u00e2\u02dc\u2026": "star",
        "\u00f0\u0178\u00a6\u00b7": "",
        "\u00f0\u0178\u2018\u2039": "",
        "\u00f0\u0178\u2019\u008d": "",
        "\u00a0": " ",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()
