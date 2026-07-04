"""Admin panel: a small server-rendered dashboard for tickets.

Plain FastAPI + Jinja2, no JS framework. KB editor and settings are added on
top of this in later steps.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app import storage

_TEMPLATES = Jinja2Templates(
    directory=str(Path(__file__).resolve().parent.parent / "templates")
)

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
