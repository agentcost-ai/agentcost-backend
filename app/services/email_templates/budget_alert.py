"""Budget threshold / hard-cap notification email."""

from typing import Optional

from ._base import base_wrapper, cta_button, esc


_CURRENCY_SYMBOLS = {"USD": "$", "INR": "₹"}


def _format_currency(amount: float, currency: str = "USD") -> str:
    sym = _CURRENCY_SYMBOLS.get((currency or "USD").upper(), "$")
    try:
        return f"{sym}{amount:,.2f}"
    except (TypeError, ValueError):
        return f"{sym}0.00"


def get_budget_alert_email_html(
    *,
    project_name: str,
    threshold_percent: float,
    utilization_percent: float,
    spent_amount: float,
    budget_amount: float,
    period_key: str,
    enforcement_mode: str,
    dashboard_link: str,
    recipient_name: Optional[str] = None,
    currency: str = "USD",
) -> str:
    """Render the budget alert HTML email."""
    safe_project = esc(project_name)
    safe_period = esc(period_key)
    safe_name = esc(recipient_name) if recipient_name else None
    mode = (enforcement_mode or "warn").lower()
    is_cap = mode == "hard_cap" and utilization_percent >= 100

    headline = (
        "Budget hard cap reached"
        if is_cap
        else f"You've crossed {threshold_percent:.0f}% of your monthly budget"
    )
    intro = (
        "Event ingestion for this project has been paused for the remainder of the month "
        "because the hard-cap enforcement mode is enabled."
        if is_cap
        else "This is an early warning so you can react before spend escalates."
    )

    accent = "#f87171" if is_cap or utilization_percent >= 100 else "#fbbf24"
    salutation = f"Hi {safe_name}," if safe_name else "Hi,"

    body = f"""\
    <tr>
        <td style="padding: 32px 40px;">
            <p style="margin: 0 0 16px 0; font-size: 15px; color: #d4d4d8;">{salutation}</p>
            <h2 style="margin: 0 0 12px 0; font-size: 20px; font-weight: 600; color: {accent};">
                {esc(headline)}
            </h2>
            <p style="margin: 0 0 20px 0; font-size: 14px; line-height: 1.6; color: #a1a1aa;">
                Project <strong style="color: #ffffff;">{safe_project}</strong> &middot; {safe_period} (UTC)
            </p>
            <p style="margin: 0 0 20px 0; font-size: 14px; line-height: 1.6; color: #a1a1aa;">
                {intro}
            </p>
            <div style="padding: 16px; border-radius: 10px; background-color: #101014; border: 1px solid #27272a;">
                <table role="presentation" style="width: 100%; border-collapse: collapse; font-size: 14px;">
                    <tr>
                        <td style="padding: 6px 0; color: #71717a;">Month-to-date spend</td>
                        <td style="padding: 6px 0; text-align: right; color: #ffffff; font-weight: 600;">{_format_currency(spent_amount, currency)}</td>
                    </tr>
                    <tr>
                        <td style="padding: 6px 0; color: #71717a;">Monthly budget</td>
                        <td style="padding: 6px 0; text-align: right; color: #ffffff; font-weight: 600;">{_format_currency(budget_amount, currency)}</td>
                    </tr>
                    <tr>
                        <td style="padding: 6px 0; color: #71717a;">Utilization</td>
                        <td style="padding: 6px 0; text-align: right; color: {accent}; font-weight: 600;">{utilization_percent:.1f}%</td>
                    </tr>
                    <tr>
                        <td style="padding: 6px 0; color: #71717a;">Enforcement mode</td>
                        <td style="padding: 6px 0; text-align: right; color: #ffffff;">{esc(mode)}</td>
                    </tr>
                </table>
            </div>
            <div style="margin-top: 24px;">
                {cta_button(dashboard_link, "Review project budget")}
            </div>
            <p style="margin: 24px 0 0 0; font-size: 12px; line-height: 1.5; color: #71717a;">
                You're receiving this because you own or administer this project on AgentCost.
                Update thresholds or change enforcement mode in <em>Settings &rarr; Budget</em>.
            </p>
        </td>
    </tr>"""

    return base_wrapper(body)
