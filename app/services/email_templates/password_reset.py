"""Password reset template."""

from typing import Optional
from ._base import base_wrapper, cta_button, esc


def get_password_reset_email_html(name: Optional[str], reset_link: str) -> str:
    display_name = esc(name) if name else "there"

    body = f"""\
    <tr>
        <td style="padding: 40px;">
            <h2 style="margin: 0 0 16px 0; font-size: 22px; font-weight: 600; color: #ffffff;">Reset your password</h2>
            <p style="margin: 0 0 24px 0; font-size: 15px; line-height: 1.6; color: #a1a1aa;">Hey {display_name},</p>
            <p style="margin: 0 0 24px 0; font-size: 15px; line-height: 1.6; color: #a1a1aa;">We received a request to reset the password for your AgentCost account. If you made this request, click the button below to set a new password:</p>
            {cta_button(reset_link, "Reset Password")}
            <p style="margin: 32px 0 0 0; font-size: 13px; line-height: 1.6; color: #71717a;">If the button doesn&rsquo;t work, copy and paste this link into your browser:</p>
            <p style="margin: 8px 0 0 0; font-size: 13px; word-break: break-all; color: #a1a1aa;">{esc(reset_link)}</p>
            <div style="margin: 32px 0 0 0; padding: 16px; background-color: #27272a; border-radius: 8px;">
                <p style="margin: 0; font-size: 13px; line-height: 1.6; color: #ffffff;">Important: This link expires in 24 hours.</p>
            </div>
            <p style="margin: 24px 0 0 0; font-size: 13px; line-height: 1.6; color: #71717a;">If you didn&rsquo;t request a password reset, you can ignore this email. Your password will remain unchanged.</p>
        </td>
    </tr>"""

    return base_wrapper(body)
