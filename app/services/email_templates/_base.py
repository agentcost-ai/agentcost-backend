"""Shared helpers for email templates."""

from datetime import datetime
from html import escape as html_escape


def get_current_year() -> int:
    return datetime.now().year


def esc(value: str | None) -> str:
    """Escape untrusted values for safe HTML embedding."""
    if value is None:
        return ""
    return html_escape(str(value), quote=True)


def base_wrapper(body_html: str, *, year: int | None = None) -> str:
    """Wrap *body_html* in the common email shell (background, outer table, footer)."""
    current_year = year or get_current_year()

    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; background-color: #0d0d0f; -webkit-font-smoothing: antialiased;">
    <table role="presentation" style="width: 100%; border-collapse: collapse;">
        <tr>
            <td style="padding: 40px 20px;">
                <table role="presentation" style="max-width: 560px; margin: 0 auto; background-color: #18181b; border-radius: 12px; border: 1px solid #27272a;">
                    <!-- Logo header -->
                    <tr>
                        <td style="padding: 40px 40px 30px 40px; text-align: center; border-bottom: 1px solid #27272a;">
                            <h1 style="margin: 0; font-size: 28px; font-weight: 700; color: #ffffff; letter-spacing: -0.025em;">AgentCost</h1>
                        </td>
                    </tr>
                    {body_html}
                    <!-- Footer -->
                    <tr>
                        <td style="padding: 24px 40px; border-top: 1px solid #27272a; text-align: center;">
                            <p style="margin: 0; font-size: 12px; color: #52525b;">&copy; {current_year} AgentCost. All rights reserved.</p>
                        </td>
                    </tr>
                </table>
            </td>
        </tr>
    </table>
</body>
</html>"""


def cta_button(href: str, label: str) -> str:
    """Render a centred call-to-action button."""
    return f"""\
<table role="presentation" style="width: 100%;">
    <tr>
        <td style="text-align: center;">
            <a href="{esc(href)}" style="display: inline-block; padding: 14px 32px; background-color: #ffffff; color: #18181b; text-decoration: none; font-size: 15px; font-weight: 600; border-radius: 8px; mso-padding-alt: 0; text-align: center;">{esc(label)}</a>
        </td>
    </tr>
</table>"""
