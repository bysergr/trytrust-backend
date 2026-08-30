"""One module for the API lane to import.

`src/api/` should not need to know that a run is checkpointed in `agent_runs`,
that escalations resume through the gate, or that the outbox needs draining.
Every function here takes a connection and returns plain dicts, ready to be
JSON-encoded by whatever web framework sits in front.

Authentication happens HERE, not in the layers below, because this is the edge
where untrusted input arrives. A caller without a token can read; changing
anything needs one (see auth.py).
"""

from __future__ import annotations

import json
from typing import Any

from . import audit, auth, chat, escalation, graph, kernel, limits, memory, registry, relay, watcher
from . import mandate as mandate_mod
from .config import LLM_MODEL, PRODUCT_DOMAIN, PRODUCT_NAME
from .ports.base import MERCHANTS, TOOLS
from .ports.setup import setup as setup_merchants


# ── lifecycle ────────────────────────────────────────────────────────────────
def bootstrap(
    *, vuelaya_url: str | None = None, mami_url: str | None = None, rappi_url: str | None = None
) -> dict[str, Any]:
    """Call once at process start. Registers merchants and event subscribers."""
    merchants = setup_merchants(vuelaya_url=vuelaya_url, mami_url=mami_url, rappi_url=rappi_url)
    relay.default_subscribers()
    return {
        "product": PRODUCT_NAME,
        "domain": PRODUCT_DOMAIN,
        "model": LLM_MODEL,
        "merchants": merchants,
        "tools_callable": TOOLS.callable_names(),
        "tools_refused": TOOLS.refused,
    }


def health(conn) -> dict[str, Any]:
    chain = audit.verify_all(conn)
    return {
        "ok": chain["valid"],
        "chains": chain["chains"],
        "events": chain["checked"],
        "merchants": sorted(MERCHANTS),
        "outbox": relay.pending(conn),
    }


# ── the buyer's conversation ─────────────────────────────────────────────────
NO_AGENT_REPLY = (
    "No encuentro un agente con mandato activo para eso. "
    "Prueba con un vuelo, un hotel, comida o mercado."
)


def session_binding(conn, session_id: str | None) -> dict[str, str] | None:
    """If this conversation already has a live run, keep that agent/mandate."""
    if not session_id:
        return None
    row = conn.execute(
        "SELECT agent_id, mandate_jti FROM agent_runs "
        "WHERE session_id=? AND status IN ('running','awaiting_human') "
        "ORDER BY created_at DESC LIMIT 1",
        (session_id,),
    ).fetchone()
    if not row:
        return None
    return {"agent_id": row["agent_id"], "mandate_jti": row["mandate_jti"]}


def turn(
    conn,
    *,
    text: str,
    session_id: str | None = None,
    person: str = "buyer",
    channel: str = "chat",
) -> dict[str, Any]:
    """One conversation turn without the caller naming an agent.

    A live run on this session wins (approve / reject / guidance must not
    re-route). Otherwise the dispatcher picks the mandate whose scope matches.
    """
    from . import router as agent_router

    bound = session_binding(conn, session_id)
    if bound:
        result = ask(
            conn,
            text=text,
            agent_id=bound["agent_id"],
            mandate_jti=bound["mandate_jti"],
            session_id=session_id,
            person=person,
            channel=channel,
        )
        return {**result, "dispatch": None, "continued": True}

    picked = agent_router.select_agent(conn, text)
    if picked is None:
        return {
            "session_id": session_id,
            "replies": [NO_AGENT_REPLY],
            "run": None,
            "awaiting_human": False,
            "dispatch": None,
            "continued": False,
        }
    result = ask(
        conn,
        text=text,
        agent_id=picked["agent_id"],
        mandate_jti=picked["mandate_jti"],
        session_id=session_id,
        person=person,
        channel=channel,
    )
    return {**result, "dispatch": picked, "continued": False}


def ask(
    conn,
    *,
    text: str,
    agent_id: str,
    mandate_jti: str,
    session_id: str | None = None,
    person: str = "buyer",
    channel: str = "chat",
) -> dict[str, Any]:
    """A turn. Starts a run, answers an escalation, or redirects one in flight."""
    # Resolve agent_id to existing DB agent if needed
    row = conn.execute(
        "SELECT id FROM agents WHERE id = ? OR name LIKE ? LIMIT 1",
        (agent_id, f"%{agent_id}%"),
    ).fetchone()
    if not row:
        row = conn.execute("SELECT id FROM agents LIMIT 1").fetchone()
    if row:
        agent_id = row["id"]

    # Resolve mandate_jti to active mandate if needed
    m_row = conn.execute(
        "SELECT jti FROM mandates WHERE jti = ? LIMIT 1", (mandate_jti,)
    ).fetchone()
    if not m_row:
        m_row = conn.execute("SELECT jti FROM mandates WHERE status = 'active' LIMIT 1").fetchone()
    if m_row:
        mandate_jti = m_row["jti"]

    session = chat.Session(
        conn,
        agent_id=agent_id,
        mandate_jti=mandate_jti,
        session_id=session_id,
        person=person,
        channel=channel,
    )
    replies = session.send(text)
    run = session.active_run()
    return {
        "session_id": session.session_id,
        "replies": replies,
        "run": _run_view(run) if run else None,
        "awaiting_human": bool(run and run["status"] == "awaiting_human"),
    }


def transcript(conn, session_id: str) -> list[dict]:
    return chat.transcript(conn, session_id)


def _run_view(run: dict) -> dict[str, Any]:
    state = run["state"]
    return {
        "run_id": run["run_id"],
        "status": run["status"],
        "node": run["node"],
        "agent_version": run["agent_version"],
        "escalation_id": run.get("escalation_id"),
        "proposal": state.get("proposal"),
        # What the search actually found, with each merchant's own CDN image
        # URLs, so a UI can show the real product pictures instead of none.
        "offers": (state.get("offers") or [])[:12],
        "result": state.get("result"),
    }


# ── the human in the loop ────────────────────────────────────────────────────
def pending_escalations(conn) -> list[dict]:
    return escalation.pending(conn)


def resolve_escalation(
    conn,
    *,
    escalation_id: str,
    decision: str,
    token: str,
    channel: str = "api",
    sticky: bool = True,
) -> dict[str, Any]:
    row = escalation.get(conn, escalation_id)
    agent_row = conn.execute(
        "SELECT agent_id FROM mandates WHERE jti=?", (row["mandate_jti"],)
    ).fetchone()
    who = auth.require(
        conn, token, "escalation.resolve", agent_row["agent_id"] if agent_row else None
    )
    result = escalation.resolve(
        conn,
        escalation_id,
        decision=decision.upper(),
        approver=who.person_id,
        channel=channel,
        sticky=sticky,
    )
    if row["run_id"]:
        graph.resume(conn, row["run_id"])
    return result


# ── the console ──────────────────────────────────────────────────────────────
def list_agents(conn) -> list[dict]:
    return [dict(r) for r in registry.list_agents(conn)]


def get_agent(conn, agent_id: str) -> dict[str, Any]:
    agent = registry.get_agent(conn, agent_id)
    version = registry.get_version(conn, agent_id)
    return {
        "agent": dict(agent),
        "ontology": json.loads(version["ontology"]),
        "history": [dict(h) for h in registry.history(conn, agent_id)],
    }


def publish_ontology(
    conn, *, agent_id: str, ontology: dict, token: str, reason: str = ""
) -> dict[str, Any]:
    who = auth.require(conn, token, "agent.publish", agent_id)
    version = registry.publish_version(
        conn, agent_id, ontology, changed_by=who.person_id, reason=reason
    )
    return {"agent_id": agent_id, "version": version, "by": who.person_id}


def create_watch(
    conn,
    *,
    agent_id: str,
    mandate_jti: str,
    query: dict,
    max_price: float,
    token: str,
    interval_s: int = 300,
) -> dict[str, Any]:
    who = auth.require(conn, token, "watch.create", agent_id)
    return watcher.create_watch(
        conn,
        agent_id=agent_id,
        mandate_jti=mandate_jti,
        query=query,
        threshold={"<=": [{"var": "offer.price"}, max_price]},
        interval_s=interval_s,
        created_by=who.person_id,
    )


def revoke_mandate(conn, *, mandate_jti: str, token: str) -> dict[str, Any]:
    who = auth.require(conn, token, "mandate.revoke")
    return mandate_mod.revoke(conn, mandate_jti, actor=who.person_id)


# ── the control tower ────────────────────────────────────────────────────────
def trail(
    conn, *, mandate_jti: str | None = None, after_seq: int = 0, limit: int = 100
) -> list[dict]:
    sql = "SELECT * FROM audit_events WHERE seq > ?"
    args: list[Any] = [after_seq]
    if mandate_jti:
        sql += " AND mandate_jti = ?"
        args.append(mandate_jti)
    sql += " ORDER BY seq LIMIT ?"
    args.append(limit)
    return [
        {**dict(r), "payload": json.loads(r["payload"])}
        for r in conn.execute(sql, tuple(args)).fetchall()
    ]


def verify(conn) -> dict[str, Any]:
    return {"chains": audit.verify_all(conn), "checkpoint": audit.verify_checkpoint(conn)}


def mandate_view(conn, mandate_jti: str) -> dict[str, Any]:
    row = mandate_mod.get(conn, mandate_jti)
    return {
        "jti": row["jti"],
        "status": row["status"],
        "spent": row["spent_total"],
        "reserved": row["reserved_amount"],
        "txn_count": row["txn_count"],
        "claims": json.loads(row["claims"]),
        "memory": memory.summarise(conn, mandate_jti),
    }


def events_since(event_id: str | None = None) -> list[dict]:
    """For SSE. The relay keeps a bounded tail; this reads it."""
    buffer = relay.SUBSCRIBERS.get("sse")
    return buffer.since(event_id) if buffer else []


# ── background work ──────────────────────────────────────────────────────────
def tick(conn) -> dict[str, Any]:
    """One scheduler pass: expire escalations, poll watches, drain the outbox."""
    return watcher.tick(conn)


def guardrails(conn) -> dict[str, Any]:
    return limits.snapshot(conn)


# ── traceability: which mandates paid for a transaction ──────────────────────
def _claims_summary(claims: dict[str, Any]) -> dict[str, Any]:
    """The part of a mandate a person reads when asking 'was this allowed?'."""
    limits_ = claims.get("limits") or {}
    return {
        "max_per_txn": limits_.get("max_per_txn"),
        "total_budget": limits_.get("total_budget"),
        "max_txn": limits_.get("max_txn"),
        "currency": claims.get("currency"),
        "scope": claims.get("scope") or {},
        "signed_with": claims.get("signed_with"),
    }


def purchases(conn, *, mandate_jti: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    """Transactions, newest first.

    `mandate_jti` filters to the mandate NAMED on the intent. A purchase debits
    that mandate's whole ancestry, so filtering by an ancestor deliberately does
    not match here -- use `purchase_trace` to see the full set a given
    transaction touched.
    """
    sql = "SELECT * FROM purchases"
    args: list[Any] = []
    if mandate_jti:
        sql += " WHERE mandate_jti = ?"
        args.append(mandate_jti)
    sql += " ORDER BY created_at DESC LIMIT ?"
    args.append(limit)
    rows = conn.execute(sql, tuple(args)).fetchall()

    out: list[dict[str, Any]] = []
    for row in rows:
        mandate = conn.execute(
            "SELECT claims FROM mandates WHERE jti=?", (row["mandate_jti"],)
        ).fetchone()
        claims = json.loads(mandate["claims"]) if mandate else {}
        out.append(
            {
                "purchase_id": row["id"],
                "status": row["status"],
                "reason_code": row["reason_code"],
                "amount": row["amount"],
                "currency": claims.get("currency"),
                "mandate_jti": row["mandate_jti"],
                "intent_jti": row["intent_jti"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "mandate_depth": len(kernel.chain(conn, row["mandate_jti"])),
                "receipt": json.loads(row["receipt"]) if row["receipt"] else None,
            }
        )
    return out


def purchase_trace(conn, purchase_id: str) -> dict[str, Any]:
    """One transaction and every mandate it was transacted against.

    A purchase does not debit a single mandate. A sticky approval issues a child
    mandate carrying `parent_jti` (H6), and `kernel.reserve_chain`, `settle` and
    `release` all walk the whole ancestry -- a child can never spend what its
    parent cannot. So the honest answer to "which mandate paid for this?" is a
    list, child first, and every entry carries the limits that were in force and
    the amount this transaction took out of it.

    Read-only. It reconstructs from the same rows the gate wrote; it never
    recomputes a decision, because a decision recomputed later is not evidence
    of what was decided then.
    """
    row = conn.execute("SELECT * FROM purchases WHERE id=?", (purchase_id,)).fetchone()
    if row is None:
        raise KeyError(f"no such purchase: {purchase_id}")

    amount = row["amount"]
    ancestry = kernel.chain(conn, row["mandate_jti"])
    mandates: list[dict[str, Any]] = []
    for depth, jti in enumerate(ancestry):
        m = conn.execute("SELECT * FROM mandates WHERE jti=?", (jti,)).fetchone()
        if m is None:
            # An ancestor named by parent_jti that no longer exists is worth
            # showing as a hole rather than skipping: a gap in the ancestry
            # changes what the numbers below it mean.
            mandates.append({"jti": jti, "depth": depth, "missing": True})
            continue
        claims = json.loads(m["claims"])
        mandates.append(
            {
                "jti": jti,
                "depth": depth,
                "role": "authorising" if depth == 0 else "ancestor",
                "status": m["status"],
                "parent_jti": m["parent_jti"],
                "user_id": m["user_id"],
                "agent_id": m["agent_id"],
                "limits": _claims_summary(claims),
                "debited": amount,
                "spent_total": m["spent_total"],
                "reserved_amount": m["reserved_amount"],
                "txn_count": m["txn_count"],
            }
        )

    intent_row = conn.execute(
        "SELECT * FROM purchase_intents WHERE jti=?", (row["intent_jti"],)
    ).fetchone()
    intent = None
    if intent_row is not None:
        intent = {
            "jti": intent_row["jti"],
            "agent_id": intent_row["agent_id"],
            "nonce": intent_row["nonce"],
            "status": intent_row["status"],
            "signature": intent_row["signature"],
            "intent": json.loads(intent_row["intent"]) if intent_row["intent"] else None,
        }

    # The chain events for this purchase. Neither id lives in a column, so this
    # matches on the payload text -- narrowed by the ancestry first, so it scans
    # one mandate's chain rather than the whole log. Both ids are needed: the
    # saga events carry `purchase_id`, but the gate's own verdict
    # (`purchase.gated`, `purchase.verified`, `purchase.refused`) carries only
    # `intent_jti`, and the verdict is the half a person actually came to read.
    placeholders = ",".join("?" for _ in ancestry) or "''"
    events = [
        {**dict(e), "payload": json.loads(e["payload"])}
        for e in conn.execute(
            f"SELECT * FROM audit_events WHERE mandate_jti IN ({placeholders}) "
            "AND (payload LIKE ? OR payload LIKE ?) ORDER BY seq",
            (*ancestry, f"%{purchase_id}%", f"%{row['intent_jti']}%"),
        ).fetchall()
    ]

    return {
        "purchase_id": row["id"],
        "status": row["status"],
        "reason_code": row["reason_code"],
        "amount": amount,
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "reservation_id": row["reservation_id"],
        "receipt": json.loads(row["receipt"]) if row["receipt"] else None,
        "mandates": mandates,
        "intent": intent,
        "events": events,
    }
