"""Email verification template."""

from typing import Optional
from ._base import base_wrapper, cta_button, esc


def get_verification_email_html(name: Optional[str], verification_link: str) -> str:
    display_name = esc(name) if name else "there"

    body = f"""\
    <tr>
        <td style="padding: 40px;">
            <h2 style="margin: 0 0 16px 0; font-size: 22px; font-weight: 600; color: #ffffff;">Verify your email address</h2>
            <p style="margin: 0 0 24px 0; font-size: 15px; line-height: 1.6; color: #a1a1aa;">Hey {display_name},</p>
            <p style="margin: 0 0 24px 0; font-size: 15px; line-height: 1.6; color: #a1a1aa;">Thanks for signing up for AgentCost. Before you can start tracking your AI agent costs, we need to verify your email address.</p>
            <p style="margin: 0 0 32px 0; font-size: 15px; line-height: 1.6; color: #a1a1aa;">Click the button below to confirm your email:</p>
            {cta_button(verification_link, "Verify Email Address")}
            <p style="margin: 32px 0 0 0; font-size: 13px; line-height: 1.6; color: #71717a;">If the button doesn&rsquo;t work, copy and paste this link into your browser:</p>
            <p style="margin: 8px 0 0 0; font-size: 13px; word-break: break-all; color: #a1a1aa;">{esc(verification_link)}</p>
            <p style="margin: 32px 0 0 0; font-size: 13px; line-height: 1.6; color: #71717a;">This link expires in 24 hours. If you didn&rsquo;t create an AgentCost account, you can safely ignore this email.</p>
        </td>
    </tr>"""

    return base_wrapper(body)
