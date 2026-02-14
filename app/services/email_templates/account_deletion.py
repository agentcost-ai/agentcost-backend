"""User notification for account deletion schedule."""

from typing import Optional
from ._base import base_wrapper, cta_button, esc


def get_account_deletion_email_html(
    email: str,
    name: Optional[str],
    grace_days: int,
    expiry_date: str,
    login_link: str,
) -> str:
    """
    Generate HTML for account deletion notification.
    """
    display_name = esc(name) if name else "there"
    
    body = f"""\
    <tr>
        <td style="padding: 32px 40px;">
            <h2 style="margin: 0 0 12px 0; font-size: 20px; font-weight: 600; color: #ef4444;">Action Required: Account Deletion Scheduled</h2>
            <p style="margin: 0 0 16px 0; font-size: 14px; line-height: 1.6; color: #a1a1aa;">Hey {display_name},</p>
            <p style="margin: 0 0 12px 0; font-size: 14px; line-height: 1.6; color: #a1a1aa;">
                Your account <strong>{esc(email)}</strong> has been scheduled for permanent deletion by an administrator.
            </p>
            
            <div style="margin: 24px 0; padding: 16px; border-radius: 10px; background-color: #101014; border: 1px solid #27272a;">
                <p style="margin: 0 0 8px 0; font-size: 13px; color: #a1a1aa;">Scheduled Deletion Date</p>
                <p style="margin: 0; font-size: 16px; font-weight: 600; color: #ffffff;">{esc(expiry_date)}</p>
                <p style="margin: 8px 0 0 0; font-size: 13px; color: #fbbf24;">(in {grace_days} days)</p>
            </div>
            
            <p style="margin: 0 0 24px 0; font-size: 14px; line-height: 1.6; color: #a1a1aa;">
                If you wish to keep your account, simply log in within the next {grace_days} days to reactivate it immediately. 
                After this period, all your data (projects, events, settings) will be <strong>permanently lost</strong> and cannot be recovered.
            </p>
            
            <div style="margin-top: 24px;">
                {cta_button(login_link, "Log in to Reactivate Account")}
            </div>
        </td>
    </tr>"""

    return base_wrapper(body)
