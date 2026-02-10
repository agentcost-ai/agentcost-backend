"""
Admin direct email template.

Used when an admin sends a direct message to a user from the dashboard.
"""

from ._base import base_wrapper, esc


def get_admin_direct_email_html(body: str) -> str:
    """Render the admin direct email template."""
    # Convert newlines to <br> for plain-text bodies
    formatted = esc(body).replace("\n", "<br>")

    content = f"""
    <div style="font-size: 14px; line-height: 1.7; color: #d4d4d8;">
        {formatted}
    </div>
    """
    return base_wrapper(content)
