"""User notification for feedback status updates."""

from typing import Optional
from ._base import base_wrapper, cta_button, esc

STATUS_LABELS = {
    "open": "Open",
    "in_progress": "In Progress",
    "completed": "Completed",
    "rejected": "Rejected",
    "duplicate": "Duplicate",
}


def get_feedback_update_email_html(
    name: Optional[str],
    title: str,
    status: str,
    admin_response: Optional[str],
    link: str,
) -> str:
    display_name = esc(name) if name else "there"
    safe_title = esc(title)
    safe_status = esc(STATUS_LABELS.get(status, status))

    response_block = ""
    if admin_response:
        safe_response = esc(admin_response)
        response_block = f"""\
            <div style="margin-top: 20px; padding: 16px; border-radius: 10px; background-color: #101014; border: 1px solid #27272a;">
                <p style="margin: 0 0 8px 0; font-size: 13px; color: #a1a1aa;">Admin response</p>
                <p style="margin: 0; font-size: 14px; line-height: 1.6; color: #ffffff;">{safe_response}</p>
            </div>"""

    body = f"""\
    <tr>
        <td style="padding: 32px 40px;">
            <h2 style="margin: 0 0 12px 0; font-size: 20px; font-weight: 600; color: #ffffff;">Your feedback has an update</h2>
            <p style="margin: 0 0 16px 0; font-size: 14px; line-height: 1.6; color: #a1a1aa;">Hey {display_name},</p>
            <p style="margin: 0 0 12px 0; font-size: 14px; line-height: 1.6; color: #a1a1aa;">Status update for:</p>
            <p style="margin: 0 0 16px 0; font-size: 15px; font-weight: 600; color: #ffffff;">{safe_title}</p>
            <p style="margin: 0 0 8px 0; font-size: 14px; line-height: 1.6; color: #a1a1aa;">Current status: <strong style="color: #ffffff;">{safe_status}</strong></p>
            {response_block}
            <div style="margin-top: 24px;">
                {cta_button(link, "View request")}
            </div>
        </td>
    </tr>"""

    return base_wrapper(body)
