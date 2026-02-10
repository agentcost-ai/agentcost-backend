"""
AgentCost - Email Templates

Centralised HTML templates for all outgoing transactional emails.
Each template lives in its own module so the email_service stays lean.
"""

from .verification import get_verification_email_html
from .password_reset import get_password_reset_email_html
from .invitation import get_invitation_email_html
from .invitation_new_user import get_new_user_invitation_email_html
from .feedback_admin import get_feedback_admin_email_html
from .feedback_update import get_feedback_update_email_html
from .admin_direct import get_admin_direct_email_html

__all__ = [
    "get_verification_email_html",
    "get_password_reset_email_html",
    "get_invitation_email_html",
    "get_new_user_invitation_email_html",
    "get_feedback_admin_email_html",
    "get_feedback_update_email_html",
    "get_admin_direct_email_html",
]
