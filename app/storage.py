"""SQLite storage for chat history and support tickets.

Every /chat turn is logged. When the assistant declines (out of scope or low
confidence), we open a ticket so a human can pick it up from the admin panel.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import DateTime, Float, String, Text, create_engine, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

DB_PATH = Path("data/app.db")

_engine = create_engine(
    f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False}
)
_Session = sessionmaker(bind=_engine, expire_on_commit=False)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    channel: Mapped[str] = mapped_column(String(20), default="web")
    session_id: Mapped[str | None] = mapped_column(String(64), default=None)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    decision: Mapped[str] = mapped_column(String(16))
    top1: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Ticket(Base):
    __tablename__ = "tickets"

    id: Mapped[int] = mapped_column(primary_key=True)
    message_id: Mapped[int | None] = mapped_column(default=None)
    channel: Mapped[str] = mapped_column(String(20), default="web")
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    decision: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(16), default="open")
    note: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    resolved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None
    )


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(_engine)


def record_chat(
    question: str,
    answer: str,
    decision: str,
    top1: float,
    *,
    channel: str = "web",
    session_id: str | None = None,
) -> int | None:
    """Log a turn and, on a decline, open a ticket. Returns the ticket id if one opened."""
    with _Session() as s:
        msg = Message(
            channel=channel,
            session_id=session_id,
            question=question,
            answer=answer,
            decision=decision,
            top1=top1,
        )
        s.add(msg)
        s.flush()

        ticket_id: int | None = None
        if decision == "decline":
            ticket = Ticket(
                message_id=msg.id,
                channel=channel,
                question=question,
                answer=answer,
                decision=decision,
            )
            s.add(ticket)
            s.flush()
            ticket_id = ticket.id

        s.commit()
        return ticket_id


def list_tickets(status: str | None = None) -> list[Ticket]:
    with _Session() as s:
        stmt = select(Ticket).order_by(Ticket.created_at.desc())
        if status:
            stmt = stmt.where(Ticket.status == status)
        return list(s.scalars(stmt))


def get_ticket(ticket_id: int) -> Ticket | None:
    with _Session() as s:
        return s.get(Ticket, ticket_id)


def update_ticket(ticket_id: int, *, status: str | None = None, note: str | None = None) -> Ticket | None:
    with _Session() as s:
        ticket = s.get(Ticket, ticket_id)
        if ticket is None:
            return None
        if status is not None:
            ticket.status = status
            ticket.resolved_at = _now() if status == "closed" else None
        if note is not None:
            ticket.note = note
        s.commit()
        return ticket


def dashboard_counts() -> dict[str, int]:
    with _Session() as s:
        return {
            "messages": s.scalar(select(func.count()).select_from(Message)) or 0,
            "tickets_open": s.scalar(
                select(func.count()).select_from(Ticket).where(Ticket.status == "open")
            )
            or 0,
            "tickets_total": s.scalar(select(func.count()).select_from(Ticket)) or 0,
        }
