"""Admin panel: a small server-rendered dashboard for tickets.

Plain FastAPI + Jinja2, no JS framework. KB editor and settings are added on
top of this in later steps.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app import kb, storage
from app.agent import SupportAgent
from app.retriever import InMemoryRetriever

_TEMPLATES = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent / "templates"))

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("", response_class=HTMLResponse)
def dashboard(request: Request) -> HTMLResponse:
    return _TEMPLATES.TemplateResponse(
        "admin/dashboard.html",
        {
            "request": request,
            "active": "dashboard",
            "counts": storage.dashboard_counts(),
            "recent": storage.list_tickets()[:5],
        },
    )


@router.get("/tickets", response_class=HTMLResponse)
def tickets(request: Request, status: str | None = None) -> HTMLResponse:
    return _TEMPLATES.TemplateResponse(
        "admin/tickets.html",
        {
            "request": request,
            "active": "tickets",
            "tickets": storage.list_tickets(status),
            "status": status,
        },
    )


@router.get("/tickets/{ticket_id}", response_class=HTMLResponse)
def ticket_detail(request: Request, ticket_id: int) -> HTMLResponse:
    ticket = storage.get_ticket(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return _TEMPLATES.TemplateResponse(
        "admin/ticket_detail.html",
        {"request": request, "active": "tickets", "t": ticket},
    )


@router.post("/tickets/{ticket_id}")
def update_ticket(
    ticket_id: int, action: str = Form(...), note: str = Form("")
) -> RedirectResponse:
    if action == "close":
        storage.update_ticket(ticket_id, status="closed", note=note)
    elif action == "reopen":
        storage.update_ticket(ticket_id, status="open", note=note)
    elif action == "save":
        storage.update_ticket(ticket_id, note=note)
    return RedirectResponse(url=f"/admin/tickets/{ticket_id}", status_code=303)


# ---- Chats ----


@router.get("/chats", response_class=HTMLResponse)
def chats(request: Request, ip: str | None = None) -> HTMLResponse:
    return _TEMPLATES.TemplateResponse(
        "admin/chats.html",
        {
            "request": request,
            "active": "chats",
            "messages": storage.list_messages(ip=ip),
            "ips": storage.list_ips(),
            "tickets": storage.ticket_by_message(),
            "ip": ip,
        },
    )


# ---- Knowledge base ----


def _rebuild_index(request: Request) -> None:
    """Re-embed the KB and hot-swap the agent so edits take effect immediately."""
    settings = request.app.state.settings
    embedder = request.app.state.embedder
    retriever = InMemoryRetriever.from_kb(settings.kb_path, embedder)
    retriever.save(settings.index_path)
    request.app.state.agent = SupportAgent(embedder, retriever, request.app.state.llm, settings)


@router.get("/kb", response_class=HTMLResponse)
def kb_list(request: Request) -> HTMLResponse:
    return _TEMPLATES.TemplateResponse(
        "admin/kb.html",
        {"request": request, "active": "kb", "entries": kb.load()},
    )


@router.get("/kb/new", response_class=HTMLResponse)
def kb_new(request: Request) -> HTMLResponse:
    return _TEMPLATES.TemplateResponse(
        "admin/kb_form.html",
        {"request": request, "active": "kb", "entry": None},
    )


@router.get("/kb/{entry_id}", response_class=HTMLResponse)
def kb_edit(request: Request, entry_id: str) -> HTMLResponse:
    entry = kb.get(entry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="KB entry not found")
    return _TEMPLATES.TemplateResponse(
        "admin/kb_form.html",
        {"request": request, "active": "kb", "entry": entry},
    )


@router.post("/kb/save")
def kb_save(
    request: Request,
    id: str = Form(...),
    category: str = Form(...),
    question: str = Form(...),
    answer: str = Form(...),
) -> RedirectResponse:
    kb.upsert(
        {
            "id": id.strip(),
            "category": category.strip(),
            "question": question.strip(),
            "answer": answer.strip(),
        }
    )
    _rebuild_index(request)
    return RedirectResponse(url="/admin/kb", status_code=303)


@router.post("/kb/{entry_id}/delete")
def kb_delete(request: Request, entry_id: str) -> RedirectResponse:
    if kb.count() <= 1:
        # Never leave the KB empty — the retriever needs at least one entry.
        return RedirectResponse(url="/admin/kb", status_code=303)
    kb.delete(entry_id)
    _rebuild_index(request)
    return RedirectResponse(url="/admin/kb", status_code=303)
