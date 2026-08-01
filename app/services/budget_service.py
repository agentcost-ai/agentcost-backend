"""
Project budget guardrails service.

Provides monthly budget evaluation, threshold alert persistence, and
fan-out of crossed-threshold notifications (in-app + email) to the
project's owner and admin members.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import select, func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.db_models import BudgetThresholdAlert, Event, Project
from ..models.user_models import ProjectMember, User, UserRole
from .currency_service import CurrencyService
from .notification_service import NotificationService


logger = logging.getLogger(__name__)


class BudgetService:
    DEFAULT_THRESHOLDS = [50.0, 80.0, 100.0]

    def __init__(self, db: AsyncSession):
        self.db = db

    @staticmethod
    def _month_window(now: datetime) -> tuple[datetime, datetime]:
        start = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
        if now.month == 12:
            end = datetime(now.year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            end = datetime(now.year, now.month + 1, 1, tzinfo=timezone.utc)
        return start, end

    @staticmethod
    def _period_key(now: datetime) -> str:
        return f"{now.year:04d}-{now.month:02d}"

    @staticmethod
    def has_budget(project: Project) -> bool:
        """Whether this project has a budget worth evaluating.

        Lets the ingestion path skip ``evaluate`` entirely — that call costs a
        month-to-date ``SUM(cost)`` plus an FX lookup on *every* batch, and for
        the overwhelming majority of projects (no budget set) the answer is
        always "not enforced".
        """
        return float(project.monthly_budget_usd or 0.0) > 0

    @staticmethod
    def normalize_thresholds(raw: Any) -> list[float]:
        values: list[float] = []
        if isinstance(raw, list):
            for item in raw:
                try:
                    val = float(item)
                except (TypeError, ValueError):
                    continue
                if 1 <= val <= 100:
                    values.append(val)

        if not values:
            return BudgetService.DEFAULT_THRESHOLDS.copy()

        # Deduplicate and sort
        return sorted(set(round(v, 2) for v in values))

    async def get_monthly_spend(self, project_id: str, now: datetime | None = None) -> float:
        current = now or datetime.now(timezone.utc)
        start, end = self._month_window(current)

        result = await self.db.execute(
            select(func.coalesce(func.sum(Event.cost), 0.0)).where(
                Event.project_id == project_id,
                Event.timestamp >= start,
                Event.timestamp < end,
            )
        )
        return float(result.scalar() or 0.0)

    async def evaluate(
        self,
        project: Project,
        additional_cost: float = 0.0,
        *,
        hot_path: bool = False,
    ) -> dict[str, Any]:
        """
        Evaluate the project's budget against month-to-date spend.

        ``additional_cost`` is in USD (the unit cost events use). The budget
        and the returned spend figures are expressed in the project's
        ``budget_currency``; ``CurrencyService`` converts USD spend into that
        currency before the comparison.

        ``hot_path=True`` (event ingestion) takes the cached FX rate instead of
        one that may block on the FX provider — see CurrencyService.
        """
        budget = float(project.monthly_budget_usd or 0.0)
        mode = (project.budget_enforcement_mode or "off").lower()
        thresholds = self.normalize_thresholds(project.budget_alert_thresholds)
        currency = CurrencyService.normalize(getattr(project, "budget_currency", "USD"))
        fx_rate = (
            CurrencyService.cached_usd_to(currency)
            if hot_path
            else await CurrencyService.usd_to(currency)
        )

        now = datetime.now(timezone.utc)
        current_spend_usd = await self.get_monthly_spend(project.id, now)
        projected_spend_usd = current_spend_usd + max(additional_cost, 0.0)

        current_spend = current_spend_usd * fx_rate
        projected_spend = projected_spend_usd * fx_rate

        if budget <= 0:
            return {
                "enabled": False,
                "mode": "off",
                "budget": None,
                "current_spend": round(current_spend, 6),
                "current_spend_usd": round(current_spend_usd, 6),
                "projected_spend": round(projected_spend, 6),
                "utilization_percent": None,
                "crossed_thresholds": [],
                "should_block": False,
                "period_key": self._period_key(now),
                "thresholds": thresholds,
                "currency": currency,
                "fx_rate": fx_rate,
            }

        utilization = (projected_spend / budget) * 100 if budget > 0 else 0.0
        crossed = [t for t in thresholds if utilization >= t]

        return {
            "enabled": True,
            "mode": mode,
            "budget": round(budget, 6),
            "current_spend": round(current_spend, 6),
            "current_spend_usd": round(current_spend_usd, 6),
            "projected_spend": round(projected_spend, 6),
            "utilization_percent": round(utilization, 2),
            "crossed_thresholds": crossed,
            "should_block": mode == "hard_cap" and projected_spend >= budget,
            "period_key": self._period_key(now),
            "thresholds": thresholds,
            "currency": currency,
            "fx_rate": fx_rate,
        }

    async def _existing_thresholds(self, project_id: str, period_key: str) -> set[float]:
        """Thresholds already alerted on for this project/month.

        Only a fast path: a concurrent writer can insert between this read and
        our own insert, which is why the insert itself is savepoint-guarded.
        """
        existing_rows = await self.db.execute(
            select(BudgetThresholdAlert.threshold_percent).where(
                BudgetThresholdAlert.project_id == project_id,
                BudgetThresholdAlert.period_key == period_key,
            )
        )
        return {round(float(v), 2) for v in existing_rows.scalars().all()}

    async def record_threshold_crossings(
        self,
        project_id: str,
        period_key: str,
        crossed_thresholds: list[float],
        spent_amount: float,
        budget_amount: float,
        utilization_percent: float,
        *,
        project: Optional[Project] = None,
        dispatch_notifications: bool = True,
    ) -> list[float]:
        if not crossed_thresholds:
            return []

        existing = await self._existing_thresholds(project_id, period_key)

        inserted: list[float] = []
        for threshold in crossed_thresholds:
            normalized = round(float(threshold), 2)
            if normalized in existing:
                continue

            row = BudgetThresholdAlert(
                project_id=project_id,
                period_key=period_key,
                threshold_percent=normalized,
                spent_amount=round(float(spent_amount), 6),
                budget_amount=round(float(budget_amount), 6),
                utilization_percent=round(float(utilization_percent), 2),
            )
            # SAVEPOINT per row. The SELECT above is check-then-insert against a
            # UNIQUE index, so two flushes crossing the same threshold at once
            # both pass the check and one hits an IntegrityError. Without the
            # savepoint that error poisons the request's transaction and rolls
            # back the *events* that were just ingested — alerting must never
            # be able to destroy telemetry.
            try:
                async with self.db.begin_nested():
                    self.db.add(row)
                    await self.db.flush()
            except IntegrityError:
                # The other writer recorded this crossing; it owns the alert.
                logger.debug(
                    "Budget threshold %s already recorded for %s/%s",
                    normalized,
                    project_id,
                    period_key,
                )
                continue

            inserted.append(normalized)

        if inserted and dispatch_notifications:
            try:
                await self._fanout_alerts(
                    project_id=project_id,
                    project=project,
                    period_key=period_key,
                    spent_amount=spent_amount,
                    budget_amount=budget_amount,
                    utilization_percent=utilization_percent,
                    crossed_thresholds=inserted,
                )
            except Exception as exc:  # noqa: BLE001 — never let alerts break ingestion
                logger.warning(
                    "Budget alert fan-out failed for project %s: %s",
                    project_id,
                    exc,
                )

        return inserted

    async def _resolve_recipients(
        self, project_id: str
    ) -> list[User]:
        """Return owner + admin members of the project (deduplicated, active, with email)."""
        # Owner
        owner_stmt = (
            select(User)
            .join(Project, Project.owner_id == User.id)
            .where(Project.id == project_id)
        )
        # Admin members
        admins_stmt = (
            select(User)
            .join(ProjectMember, ProjectMember.user_id == User.id)
            .where(
                ProjectMember.project_id == project_id,
                ProjectMember.role == UserRole.ADMIN.value,
                ProjectMember.accepted_at.isnot(None),
            )
        )

        owner_rows = (await self.db.execute(owner_stmt)).scalars().all()
        admin_rows = (await self.db.execute(admins_stmt)).scalars().all()

        seen: dict[str, User] = {}
        for user in list(owner_rows) + list(admin_rows):
            if not user or not user.id or not user.email:
                continue
            if not getattr(user, "is_active", True):
                continue
            if getattr(user, "is_deleted", False):
                continue
            seen[user.id] = user
        return list(seen.values())

    async def _fanout_alerts(
        self,
        *,
        project_id: str,
        project: Optional[Project],
        period_key: str,
        spent_amount: float,
        budget_amount: float,
        utilization_percent: float,
        crossed_thresholds: list[float],
    ) -> None:
        recipients = await self._resolve_recipients(project_id)
        if not recipients:
            return

        if project is None:
            project = (
                await self.db.execute(select(Project).where(Project.id == project_id))
            ).scalar_one_or_none()
            if project is None:
                return

        mode = (project.budget_enforcement_mode or "off").lower()
        currency = CurrencyService.normalize(getattr(project, "budget_currency", "USD"))
        highest_threshold = max(crossed_thresholds)
        is_hard_cap_breach = mode == "hard_cap" and utilization_percent >= 100

        # In-app notification — one per recipient, summarizing the highest crossed threshold.
        notif_service = NotificationService(self.db)
        title = (
            f"Budget hard cap reached for {project.name}"
            if is_hard_cap_breach
            else f"{int(round(highest_threshold))}% of monthly budget used"
        )
        body = (
            f"Project {project.name} has used {utilization_percent:.1f}% of its "
            f"{CurrencyService.format_amount(budget_amount, currency)} budget for {period_key}."
        )
        severity = (
            "critical"
            if is_hard_cap_breach
            else ("warning" if highest_threshold >= 80 else "info")
        )
        payload = {
            "thresholds_crossed": crossed_thresholds,
            "spent_amount": round(spent_amount, 6),
            "budget_amount": round(budget_amount, 6),
            "utilization_percent": round(utilization_percent, 2),
            "period_key": period_key,
            "enforcement_mode": mode,
            "currency": currency,
        }
        for user in recipients:
            try:
                await notif_service.create(
                    user_id=user.id,
                    type=("budget_hard_cap" if is_hard_cap_breach else "budget_threshold"),
                    severity=severity,
                    title=title,
                    body=body,
                    link="/settings",
                    project_id=project_id,
                    payload=payload,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to create in-app notification for %s: %s", user.id, exc)

        # Email — fire once per recipient, per highest threshold (avoids spam during a batch).
        try:
            from .email_service import send_budget_alert_email  # local import to avoid cycle
        except Exception as exc:  # noqa: BLE001
            logger.warning("Email service unavailable for budget alert: %s", exc)
            return

        for user in recipients:
            try:
                await send_budget_alert_email(
                    user.email,
                    project_name=project.name,
                    threshold_percent=highest_threshold,
                    utilization_percent=utilization_percent,
                    spent_amount=spent_amount,
                    budget_amount=budget_amount,
                    period_key=period_key,
                    enforcement_mode=mode,
                    recipient_name=user.name,
                    currency=currency,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to send budget alert email to %s: %s", user.email, exc)
