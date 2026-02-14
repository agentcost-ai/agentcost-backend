"""Welcome email template for new users — sent after successful registration."""

from typing import Optional
from ._base import base_wrapper, cta_button, esc


# Badge label mapping
_BADGE_LABELS = {
    "top_20": "Top 20",
    "top_50": "Top 50",
    "top_100": "Top 100",
    "top_1000": "Top 1,000",
}


def _badge_section(user_number: int, badge: str) -> str:
    """Render the early-adopter badge callout."""
    label = _BADGE_LABELS.get(badge, f"#{user_number}")
    ordinal = _ordinal(user_number)

    return f"""\
            <table role="presentation" style="width: 100%; margin: 24px 0; border-collapse: collapse;">
                <tr>
                    <td style="background: linear-gradient(135deg, #18181b 0%, #27272a 100%); border: 1px solid #3f3f46; border-radius: 10px; padding: 28px; text-align: center;">
                        <p style="margin: 0 0 4px 0; font-size: 13px; text-transform: uppercase; letter-spacing: 0.1em; color: #a1a1aa;">Early Adopter</p>
                        <p style="margin: 0 0 8px 0; font-size: 36px; font-weight: 700; color: #ffffff;">{ordinal}</p>
                        <p style="margin: 0; font-size: 14px; color: #a1a1aa;">user to join AgentCost</p>
                        <table role="presentation" style="margin: 16px auto 0 auto;">
                            <tr>
                                <td style="background-color: #ffffff; color: #18181b; font-size: 12px; font-weight: 700; padding: 6px 16px; border-radius: 100px; text-transform: uppercase; letter-spacing: 0.05em;">{esc(label)} Member</td>
                            </tr>
                        </table>
                    </td>
                </tr>
            </table>"""


def _ordinal(n: int) -> str:
    """Return the ordinal string for *n* (e.g. 1st, 2nd, 3rd, 11th)."""
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def get_welcome_email_html(
    name: Optional[str],
    user_number: int,
    milestone_badge: Optional[str],
    dashboard_link: str,
) -> str:
    display_name = esc(name) if name else "there"

    badge_html = ""
    if milestone_badge and user_number:
        badge_html = _badge_section(user_number, milestone_badge)

    body = f"""\
    <tr>
        <td style="padding: 40px;">
            <h2 style="margin: 0 0 16px 0; font-size: 22px; font-weight: 600; color: #ffffff;">Welcome to AgentCost!</h2>
            <p style="margin: 0 0 24px 0; font-size: 15px; line-height: 1.6; color: #a1a1aa;">Hey {display_name},</p>
            <p style="margin: 0 0 8px 0; font-size: 15px; line-height: 1.6; color: #a1a1aa;">
                Thank you for joining AgentCost &mdash; the developer-first platform for tracking, analysing, and optimising your AI&nbsp;agent costs.
            </p>
{badge_html}
            <p style="margin: 0 0 8px 0; font-size: 15px; line-height: 1.6; color: #a1a1aa;">
                As an early member, your feedback directly shapes the product. We read every suggestion and prioritise features based on what our community needs most.
            </p>

            <!-- What you get -->
            <table role="presentation" style="width: 100%; margin: 24px 0; border-collapse: collapse;">
                <tr>
                    <td style="padding: 20px; background-color: #1c1c1f; border-radius: 8px; border: 1px solid #27272a;">
                        <p style="margin: 0 0 12px 0; font-size: 14px; font-weight: 600; color: #ffffff;">Here&rsquo;s what you can do right away:</p>
                        <table role="presentation" style="width: 100%; border-collapse: collapse;">
                            <tr>
                                <td style="padding: 6px 0; font-size: 14px; color: #a1a1aa;">&#x2022;&ensp;Install the SDK and start tracking LLM calls in under 2 minutes</td>
                            </tr>
                            <tr>
                                <td style="padding: 6px 0; font-size: 14px; color: #a1a1aa;">&#x2022;&ensp;View per-agent and per-model cost breakdowns on your dashboard</td>
                            </tr>
                            <tr>
                                <td style="padding: 6px 0; font-size: 14px; color: #a1a1aa;">&#x2022;&ensp;Get automated optimisation recommendations to cut costs</td>
                            </tr>
                            <tr>
                                <td style="padding: 6px 0; font-size: 14px; color: #a1a1aa;">&#x2022;&ensp;Detect caching opportunities and duplicate queries</td>
                            </tr>
                        </table>
                    </td>
                </tr>
            </table>

            {cta_button(dashboard_link, "Open Dashboard")}

            <p style="margin: 32px 0 0 0; font-size: 13px; line-height: 1.6; color: #71717a;">
                Questions or ideas? Reply to this email or use the feedback button in the dashboard &mdash; we&rsquo;d love to hear from you.
            </p>
        </td>
    </tr>"""

    return base_wrapper(body)
