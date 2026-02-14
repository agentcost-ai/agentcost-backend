"""
Email Service using Resend

Handles all outgoing emails: verification, password reset, invitations, feedback.
Templates live in the ``email_templates`` package to keep this module lean.
"""

import resend
import asyncio
from typing import Optional

from ..config import get_settings
from .email_templates import (
    get_verification_email_html,
    get_password_reset_email_html,
    get_invitation_email_html,
    get_new_user_invitation_email_html,
    get_feedback_admin_email_html,
    get_feedback_update_email_html,
    get_admin_direct_email_html,
    get_welcome_email_html,
    get_account_deletion_email_html,
)

settings = get_settings()

# Configure Resend with settings from config
resend.api_key = settings.resend_api_key

SENDER_EMAIL = settings.resend_sender_email
SENDER_NAME = settings.resend_sender_name
FRONTEND_URL = settings.frontend_url


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _send(to: str | list[str], subject: str, html: str) -> bool:
    """Low-level send wrapper used by every public function."""
    if not resend.api_key:
        print(f"[EMAIL] RESEND_API_KEY not set - skipping email to {to}")
        return False

    try:
        recipients = [to] if isinstance(to, str) else to
        params = {
            "from": f"{SENDER_NAME} <{SENDER_EMAIL}>",
            "to": recipients,
            "subject": subject,
            "html": html,
        }

        response = resend.Emails.send(params)

        if response and response.get("id"):
            print(f"[EMAIL] Email sent to {to} (id: {response['id']})")
            return True

        print(f"[EMAIL] Failed to send email to {to}: {response}")
        return False
    except Exception as e:
        print(f"[EMAIL] Error sending email to {to}: {e}")
        return False


async def _send_async(to: str | list[str], subject: str, html: str) -> bool:
    """Async wrapper that runs the blocking _send in a thread executor."""
    return await asyncio.to_thread(_send, to, subject, html)


# ---------------------------------------------------------------------------
# Public email senders
# ---------------------------------------------------------------------------


async def send_verification_email(
    email: str,
    token: str,
    name: Optional[str] = None,
) -> bool:
    """Send email verification link to user."""
    verification_link = f"{FRONTEND_URL}/auth/verify-email?token={token}"
    html = get_verification_email_html(name, verification_link)
    return await _send_async(email, "Verify your AgentCost account", html)


async def send_password_reset_email(
    email: str,
    token: str,
    name: Optional[str] = None,
) -> bool:
    """Send password reset link to user."""
    reset_link = f"{FRONTEND_URL}/auth/reset-password?token={token}"
    html = get_password_reset_email_html(name, reset_link)
    return await _send_async(email, "Reset your AgentCost password", html)


async def send_invitation_email(
    email: str,
    project_name: str,
    inviter_name: str,
    role: str,
    invitee_name: Optional[str] = None,
) -> bool:
    """Send project invitation email to an existing user."""
    dashboard_link = FRONTEND_URL
    html = get_invitation_email_html(
        invitee_name, project_name, inviter_name, role, dashboard_link
    )
    return await _send_async(
        email,
        f"You've been invited to {project_name} on AgentCost",
        html,
    )


async def send_new_user_invitation_email(
    email: str,
    project_name: str,
    inviter_name: str,
    role: str,
) -> bool:
    """Send project invitation email to an unregistered user."""
    register_link = f"{FRONTEND_URL}/auth/register"
    html = get_new_user_invitation_email_html(
        email, project_name, inviter_name, role, register_link
    )
    return await _send_async(
        email,
        f"You've been invited to {project_name} on AgentCost",
        html,
    )


async def send_feedback_admin_notification(
    feedback_id: str,
    feedback_type: str,
    title: str,
    description: str,
    submitted_by: str,
) -> bool:
    """Notify admins that new feedback has been submitted."""
    admin_email = settings.feedback_admin_email
    if not admin_email:
        print("[EMAIL] Feedback admin email not configured - skipping notification")
        return False

    link = f"{FRONTEND_URL}/feedback?feedback_id={feedback_id}"
    html = get_feedback_admin_email_html(
        feedback_type=feedback_type,
        title=title,
        description=description,
        submitted_by=submitted_by,
        link=link,
    )
    return await _send_async(admin_email, "New feedback submitted", html)


async def send_feedback_update_email(
    email: str,
    title: str,
    status: str,
    admin_response: Optional[str],
    name: Optional[str] = None,
    feedback_id: Optional[str] = None,
) -> bool:
    """Notify a user about a feedback status update."""
    link_suffix = f"?feedback_id={feedback_id}" if feedback_id else ""
    link = f"{FRONTEND_URL}/feedback{link_suffix}"
    html = get_feedback_update_email_html(
        name=name,
        title=title,
        status=status,
        admin_response=admin_response,
        link=link,
    )
    return await _send_async(email, "Your feedback has been updated", html)


async def send_welcome_email(
    email: str,
    name: Optional[str] = None,
    user_number: int = 0,
    milestone_badge: Optional[str] = None,
) -> bool:
    """Send welcome email with early-adopter badge to a new user."""
    dashboard_link = FRONTEND_URL
    html = get_welcome_email_html(
        name=name,
        user_number=user_number,
        milestone_badge=milestone_badge,
        dashboard_link=dashboard_link,
    )
    subject = "Welcome to AgentCost"
    if milestone_badge:
        subject = f"Welcome to AgentCost — You're #{user_number}!"
    return await _send_async(email, subject, html)


async def send_account_deletion_email(
    email: str,
    name: Optional[str],
    grace_expiry_date: str,
) -> bool:
    """
    Notify user that their account is scheduled for deletion.
    
    Includes grace period expiry date and reactivation link.
    """
    login_link = f"{FRONTEND_URL}/auth/login"
    grace_days = settings.deletion_grace_days
    
    html = get_account_deletion_email_html(
        email=email,
        name=name,
        grace_days=grace_days,
        expiry_date=grace_expiry_date,
        login_link=login_link,
    )
    
    return await _send_async(
        email, 
        "Action Required: Account Deletion Scheduled", 
        html
    )


def send_admin_email(to: str, subject: str, body: str) -> bool:
    """
    Send a direct email from the admin panel.

    Public wrapper around ``_send`` that applies the standard admin
    email template so callers do not need to construct raw HTML.
    """
    html = get_admin_direct_email_html(body)
    return _send(to, subject, html)
