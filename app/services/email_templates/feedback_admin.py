"""Admin notification for new feedback submissions."""

from ._base import base_wrapper, cta_button, esc


def get_feedback_admin_email_html(
    feedback_type: str,
    title: str,
    description: str,
    submitted_by: str,
    link: str,
) -> str:
    safe_type = esc(feedback_type)
    safe_title = esc(title)
    safe_desc = esc(description)
    safe_by = esc(submitted_by)

    body = f"""\
    <tr>
        <td style="padding: 32px 40px;">
            <h2 style="margin: 0 0 12px 0; font-size: 20px; font-weight: 600; color: #ffffff;">New feedback submitted</h2>
            <p style="margin: 0 0 20px 0; font-size: 14px; line-height: 1.6; color: #a1a1aa;">
                Type: <strong style="color: #ffffff;">{safe_type}</strong><br>
                Submitted by: <strong style="color: #ffffff;">{safe_by}</strong>
            </p>
            <div style="padding: 16px; border-radius: 10px; background-color: #101014; border: 1px solid #27272a;">
                <p style="margin: 0 0 10px 0; font-size: 15px; font-weight: 600; color: #ffffff;">{safe_title}</p>
                <p style="margin: 0; font-size: 14px; line-height: 1.6; color: #a1a1aa;">{safe_desc}</p>
            </div>
            <div style="margin-top: 24px;">
                {cta_button(link, "Review feedback")}
            </div>
        </td>
    </tr>"""

    return base_wrapper(body)
