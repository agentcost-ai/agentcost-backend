"""
Email Service using Resend

Handles all outgoing emails: verification, password reset, etc.
"""

import resend
from typing import Optional
from datetime import datetime

from ..config import get_settings

settings = get_settings()

# Configure Resend with settings from config
resend.api_key = settings.resend_api_key

SENDER_EMAIL = settings.resend_sender_email
SENDER_NAME = settings.resend_sender_name
FRONTEND_URL = settings.frontend_url


def _get_verification_email_html(name: Optional[str], verification_link: str) -> str:
    """Build the HTML template for email verification"""
    
    display_name = name if name else "there"
    current_year = datetime.now().year
    
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Verify Your Email</title>
</head>
<body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; background-color: #0d0d0f;">
    <table role="presentation" style="width: 100%; border-collapse: collapse;">
        <tr>
            <td style="padding: 40px 20px;">
                <table role="presentation" style="max-width: 560px; margin: 0 auto; background-color: #18181b; border-radius: 12px; border: 1px solid #27272a;">
                    <!-- Header -->
                    <tr>
                        <td style="padding: 40px 40px 30px 40px; text-align: center; border-bottom: 1px solid #27272a;">
                            <h1 style="margin: 0; font-size: 28px; font-weight: 700; color: #ffffff;">
                                AgentCost
                            </h1>
                        </td>
                    </tr>
                    
                    <!-- Body -->
                    <tr>
                        <td style="padding: 40px;">
                            <h2 style="margin: 0 0 16px 0; font-size: 22px; font-weight: 600; color: #ffffff;">
                                Verify your email address
                            </h2>
                            <p style="margin: 0 0 24px 0; font-size: 15px; line-height: 1.6; color: #a1a1aa;">
                                Hey {display_name},
                            </p>
                            <p style="margin: 0 0 24px 0; font-size: 15px; line-height: 1.6; color: #a1a1aa;">
                                Thanks for signing up for AgentCost. Before you can start tracking your AI agent costs, we need to verify your email address.
                            </p>
                            <p style="margin: 0 0 32px 0; font-size: 15px; line-height: 1.6; color: #a1a1aa;">
                                Click the button below to confirm your email:
                            </p>
                            
                            <!-- CTA Button -->
                            <table role="presentation" style="width: 100%;">
                                <tr>
                                    <td style="text-align: center;">
                                        <a href="{verification_link}" style="display: inline-block; padding: 14px 32px; background-color: #ffffff; color: #18181b; text-decoration: none; font-size: 15px; font-weight: 600; border-radius: 8px;">
                                            Verify Email Address
                                        </a>
                                    </td>
                                </tr>
                            </table>
                            
                            <p style="margin: 32px 0 0 0; font-size: 13px; line-height: 1.6; color: #71717a;">
                                If the button doesn't work, copy and paste this link into your browser:
                            </p>
                            <p style="margin: 8px 0 0 0; font-size: 13px; word-break: break-all; color: #a1a1aa;">
                                {verification_link}
                            </p>
                            
                            <p style="margin: 32px 0 0 0; font-size: 13px; line-height: 1.6; color: #71717a;">
                                This link expires in 24 hours. If you didn't create an AgentCost account, you can safely ignore this email.
                            </p>
                        </td>
                    </tr>
                    
                    <!-- Footer -->
                    <tr>
                        <td style="padding: 24px 40px; border-top: 1px solid #27272a; text-align: center;">
                            <p style="margin: 0; font-size: 12px; color: #52525b;">
                                © {current_year} AgentCost. All rights reserved.
                            </p>
                            <p style="margin: 8px 0 0 0; font-size: 12px; color: #52525b;">
                                You're receiving this email because you signed up for AgentCost.
                            </p>
                        </td>
                    </tr>
                </table>
            </td>
        </tr>
    </table>
</body>
</html>
"""


def _get_password_reset_email_html(name: Optional[str], reset_link: str) -> str:
    """Build the HTML template for password reset"""
    
    display_name = name if name else "there"
    current_year = datetime.now().year
    
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reset Your Password</title>
</head>
<body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; background-color: #0d0d0f;">
    <table role="presentation" style="width: 100%; border-collapse: collapse;">
        <tr>
            <td style="padding: 40px 20px;">
                <table role="presentation" style="max-width: 560px; margin: 0 auto; background-color: #18181b; border-radius: 12px; border: 1px solid #27272a;">
                    <!-- Header -->
                    <tr>
                        <td style="padding: 40px 40px 30px 40px; text-align: center; border-bottom: 1px solid #27272a;">
                            <h1 style="margin: 0; font-size: 28px; font-weight: 700; color: #ffffff;">
                                AgentCost
                            </h1>
                        </td>
                    </tr>
                    
                    <!-- Body -->
                    <tr>
                        <td style="padding: 40px;">
                            <h2 style="margin: 0 0 16px 0; font-size: 22px; font-weight: 600; color: #ffffff;">
                                Reset your password
                            </h2>
                            <p style="margin: 0 0 24px 0; font-size: 15px; line-height: 1.6; color: #a1a1aa;">
                                Hey {display_name},
                            </p>
                            <p style="margin: 0 0 24px 0; font-size: 15px; line-height: 1.6; color: #a1a1aa;">
                                We received a request to reset the password for your AgentCost account. If you made this request, click the button below to set a new password:
                            </p>
                            
                            <!-- CTA Button -->
                            <table role="presentation" style="width: 100%;">
                                <tr>
                                    <td style="text-align: center;">
                                        <a href="{reset_link}" style="display: inline-block; padding: 14px 32px; background-color: #ffffff; color: #18181b; text-decoration: none; font-size: 15px; font-weight: 600; border-radius: 8px;">
                                            Reset Password
                                        </a>
                                    </td>
                                </tr>
                            </table>
                            
                            <p style="margin: 32px 0 0 0; font-size: 13px; line-height: 1.6; color: #71717a;">
                                If the button doesn't work, copy and paste this link into your browser:
                            </p>
                            <p style="margin: 8px 0 0 0; font-size: 13px; word-break: break-all; color: #a1a1aa;">
                                {reset_link}
                            </p>
                            
                            <div style="margin: 32px 0 0 0; padding: 16px; background-color: #27272a; border-radius: 8px;">
                                <p style="margin: 0; font-size: 13px; line-height: 1.6; color: #ffffff;">
                                    Important: This link expires in 24 hours.
                                </p>
                            </div>
                            
                            <p style="margin: 24px 0 0 0; font-size: 13px; line-height: 1.6; color: #71717a;">
                                If you didn't request a password reset, you can ignore this email. Your password will remain unchanged.
                            </p>
                        </td>
                    </tr>
                    
                    <!-- Footer -->
                    <tr>
                        <td style="padding: 24px 40px; border-top: 1px solid #27272a; text-align: center;">
                            <p style="margin: 0; font-size: 12px; color: #52525b;">
                                © {current_year} AgentCost. All rights reserved.
                            </p>
                            <p style="margin: 8px 0 0 0; font-size: 12px; color: #52525b;">
                                This is an automated security email from AgentCost.
                            </p>
                        </td>
                    </tr>
                </table>
            </td>
        </tr>
    </table>
</body>
</html>
"""


async def send_verification_email(email: str, token: str, name: Optional[str] = None) -> bool:
    """
    Send email verification link to user.
    
    Returns True if sent successfully, False otherwise.
    """
    if not resend.api_key:
        # Log warning in dev, but don't crash
        print("[EMAIL] RESEND_API_KEY not set - skipping verification email")
        return False
    
    verification_link = f"{FRONTEND_URL}/auth/verify-email?token={token}"
    
    try:
        params = {
            "from": f"{SENDER_NAME} <{SENDER_EMAIL}>",
            "to": [email],
            "subject": "Verify your AgentCost account",
            "html": _get_verification_email_html(name, verification_link),
        }
        
        response = resend.Emails.send(params)
        
        # resend returns dict with 'id' on success
        if response and response.get("id"):
            print(f"[EMAIL] Verification email sent to {email} (id: {response['id']})")
            return True
        
        print(f"[EMAIL] Failed to send verification email to {email}: {response}")
        return False
        
    except Exception as e:
        print(f"[EMAIL] Error sending verification email to {email}: {e}")
        return False


async def send_password_reset_email(email: str, token: str, name: Optional[str] = None) -> bool:
    """
    Send password reset link to user.
    
    Returns True if sent successfully, False otherwise.
    """
    if not resend.api_key:
        print("[EMAIL] RESEND_API_KEY not set - skipping password reset email")
        return False
    
    reset_link = f"{FRONTEND_URL}/auth/reset-password?token={token}"
    
    try:
        params = {
            "from": f"{SENDER_NAME} <{SENDER_EMAIL}>",
            "to": [email],
            "subject": "Reset your AgentCost password",
            "html": _get_password_reset_email_html(name, reset_link),
        }
        
        response = resend.Emails.send(params)
        
        if response and response.get("id"):
            print(f"[EMAIL] Password reset email sent to {email} (id: {response['id']})")
            return True
        
        print(f"[EMAIL] Failed to send password reset email to {email}: {response}")
        return False
        
    except Exception as e:
        print(f"[EMAIL] Error sending password reset email to {email}: {e}")
        return False


def _get_invitation_email_html(
    invitee_name: Optional[str],
    project_name: str,
    inviter_name: str,
    role: str,
) -> str:
    """Build the HTML template for project invitation"""
    
    display_name = invitee_name if invitee_name else "there"
    current_year = datetime.now().year
    dashboard_link = f"{FRONTEND_URL}"
    
    role_description = {
        "admin": "full access to manage the project, invite members, and view all data",
        "member": "access to view analytics, create events, and export data",
        "viewer": "read-only access to view project analytics and events",
    }.get(role.lower(), "access to the project")
    
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Project Invitation</title>
</head>
<body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; background-color: #0d0d0f;">
    <table role="presentation" style="width: 100%; border-collapse: collapse;">
        <tr>
            <td style="padding: 40px 20px;">
                <table role="presentation" style="max-width: 560px; margin: 0 auto; background-color: #18181b; border-radius: 12px; border: 1px solid #27272a;">
                    <!-- Header -->
                    <tr>
                        <td style="padding: 40px 40px 30px 40px; text-align: center; border-bottom: 1px solid #27272a;">
                            <h1 style="margin: 0; font-size: 28px; font-weight: 700; color: #ffffff;">
                                AgentCost
                            </h1>
                        </td>
                    </tr>
                    
                    <!-- Body -->
                    <tr>
                        <td style="padding: 40px;">
                            <h2 style="margin: 0 0 16px 0; font-size: 22px; font-weight: 600; color: #ffffff;">
                                You've been invited to a project
                            </h2>
                            <p style="margin: 0 0 24px 0; font-size: 15px; line-height: 1.6; color: #a1a1aa;">
                                Hey {display_name},
                            </p>
                            <p style="margin: 0 0 24px 0; font-size: 15px; line-height: 1.6; color: #a1a1aa;">
                                <strong style="color: #ffffff;">{inviter_name}</strong> has invited you to join 
                                <strong style="color: #ffffff;">{project_name}</strong> on AgentCost.
                            </p>
                            
                            <div style="margin: 24px 0; padding: 20px; background-color: #27272a; border-radius: 8px;">
                                <p style="margin: 0 0 8px 0; font-size: 13px; color: #71717a; text-transform: uppercase; letter-spacing: 0.5px;">
                                    Your Role
                                </p>
                                <p style="margin: 0; font-size: 18px; font-weight: 600; color: #ffffff; text-transform: capitalize;">
                                    {role}
                                </p>
                                <p style="margin: 8px 0 0 0; font-size: 13px; color: #a1a1aa;">
                                    As a {role}, you'll have {role_description}.
                                </p>
                            </div>
                            
                            <p style="margin: 0 0 32px 0; font-size: 15px; line-height: 1.6; color: #a1a1aa;">
                                Log in to your AgentCost dashboard to accept this invitation:
                            </p>
                            
                            <!-- CTA Button -->
                            <table role="presentation" style="width: 100%;">
                                <tr>
                                    <td style="text-align: center;">
                                        <a href="{dashboard_link}" style="display: inline-block; padding: 14px 32px; background-color: #ffffff; color: #18181b; text-decoration: none; font-size: 15px; font-weight: 600; border-radius: 8px;">
                                            View Invitation
                                        </a>
                                    </td>
                                </tr>
                            </table>
                            
                            <p style="margin: 32px 0 0 0; font-size: 13px; line-height: 1.6; color: #71717a;">
                                If you don't want to join this project, you can ignore this email or decline the invitation from your dashboard.
                            </p>
                        </td>
                    </tr>
                    
                    <!-- Footer -->
                    <tr>
                        <td style="padding: 24px 40px; border-top: 1px solid #27272a; text-align: center;">
                            <p style="margin: 0; font-size: 12px; color: #52525b;">
                                © {current_year} AgentCost. All rights reserved.
                            </p>
                            <p style="margin: 8px 0 0 0; font-size: 12px; color: #52525b;">
                                You received this email because {inviter_name} invited you to a project.
                            </p>
                        </td>
                    </tr>
                </table>
            </td>
        </tr>
    </table>
</body>
</html>
"""


async def send_invitation_email(
    email: str,
    project_name: str,
    inviter_name: str,
    role: str,
    invitee_name: Optional[str] = None,
) -> bool:
    """
    Send project invitation email to user.
    
    Returns True if sent successfully, False otherwise.
    """
    if not resend.api_key:
        print("[EMAIL] RESEND_API_KEY not set - skipping invitation email")
        return False
    
    try:
        params = {
            "from": f"{SENDER_NAME} <{SENDER_EMAIL}>",
            "to": [email],
            "subject": f"You've been invited to {project_name} on AgentCost",
            "html": _get_invitation_email_html(invitee_name, project_name, inviter_name, role),
        }
        
        response = resend.Emails.send(params)
        
        if response and response.get("id"):
            print(f"[EMAIL] Invitation email sent to {email} (id: {response['id']})")
            return True
        
        print(f"[EMAIL] Failed to send invitation email to {email}: {response}")
        return False
        
    except Exception as e:
        print(f"[EMAIL] Error sending invitation email to {email}: {e}")
        return False


def _get_new_user_invitation_email_html(
    email: str,
    project_name: str,
    inviter_name: str,
    role: str,
) -> str:
    """Build the HTML template for inviting a user who hasn't registered yet"""
    
    current_year = datetime.now().year
    register_link = f"{FRONTEND_URL}/auth/register"
    
    role_description = {
        "admin": "full access to manage the project, invite members, and view all data",
        "member": "access to view analytics, create events, and export data",
        "viewer": "read-only access to view project analytics and events",
    }.get(role.lower(), "access to the project")
    
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Project Invitation</title>
</head>
<body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; background-color: #0d0d0f;">
    <table role="presentation" style="width: 100%; border-collapse: collapse;">
        <tr>
            <td style="padding: 40px 20px;">
                <table role="presentation" style="max-width: 560px; margin: 0 auto; background-color: #18181b; border-radius: 12px; border: 1px solid #27272a;">
                    <!-- Header -->
                    <tr>
                        <td style="padding: 40px 40px 30px 40px; text-align: center; border-bottom: 1px solid #27272a;">
                            <h1 style="margin: 0; font-size: 28px; font-weight: 700; color: #ffffff;">
                                AgentCost
                            </h1>
                        </td>
                    </tr>
                    
                    <!-- Body -->
                    <tr>
                        <td style="padding: 40px;">
                            <h2 style="margin: 0 0 16px 0; font-size: 22px; font-weight: 600; color: #ffffff;">
                                You've been invited to join a project
                            </h2>
                            <p style="margin: 0 0 24px 0; font-size: 15px; line-height: 1.6; color: #a1a1aa;">
                                Hello,
                            </p>
                            <p style="margin: 0 0 24px 0; font-size: 15px; line-height: 1.6; color: #a1a1aa;">
                                <strong style="color: #ffffff;">{inviter_name}</strong> has invited you to join 
                                <strong style="color: #ffffff;">{project_name}</strong> on AgentCost, an LLM cost tracking platform.
                            </p>
                            
                            <div style="margin: 24px 0; padding: 20px; background-color: #27272a; border-radius: 8px;">
                                <p style="margin: 0 0 8px 0; font-size: 13px; color: #71717a; text-transform: uppercase; letter-spacing: 0.5px;">
                                    Your Role
                                </p>
                                <p style="margin: 0; font-size: 18px; font-weight: 600; color: #ffffff; text-transform: capitalize;">
                                    {role}
                                </p>
                                <p style="margin: 8px 0 0 0; font-size: 13px; color: #a1a1aa;">
                                    As a {role}, you'll have {role_description}.
                                </p>
                            </div>
                            
                            <p style="margin: 0 0 16px 0; font-size: 15px; line-height: 1.6; color: #a1a1aa;">
                                To accept this invitation, you'll need to create an AgentCost account using this email address 
                                (<strong style="color: #ffffff;">{email}</strong>).
                            </p>
                            
                            <p style="margin: 0 0 32px 0; font-size: 15px; line-height: 1.6; color: #a1a1aa;">
                                Once you've registered and verified your email, the invitation will be waiting for you in your dashboard.
                            </p>
                            
                            <!-- CTA Button -->
                            <table role="presentation" style="width: 100%;">
                                <tr>
                                    <td style="text-align: center;">
                                        <a href="{register_link}" style="display: inline-block; padding: 14px 32px; background-color: #ffffff; color: #18181b; text-decoration: none; font-size: 15px; font-weight: 600; border-radius: 8px;">
                                            Create Account
                                        </a>
                                    </td>
                                </tr>
                            </table>
                            
                            <p style="margin: 32px 0 0 0; font-size: 13px; line-height: 1.6; color: #71717a;">
                                If you don't want to join this project, you can ignore this email. The invitation will remain 
                                pending until you register.
                            </p>
                        </td>
                    </tr>
                    
                    <!-- Footer -->
                    <tr>
                        <td style="padding: 24px 40px; border-top: 1px solid #27272a; text-align: center;">
                            <p style="margin: 0; font-size: 12px; color: #52525b;">
                                {current_year} AgentCost. All rights reserved.
                            </p>
                            <p style="margin: 8px 0 0 0; font-size: 12px; color: #52525b;">
                                You received this email because {inviter_name} invited you to a project.
                            </p>
                        </td>
                    </tr>
                </table>
            </td>
        </tr>
    </table>
</body>
</html>
"""


async def send_new_user_invitation_email(
    email: str,
    project_name: str,
    inviter_name: str,
    role: str,
) -> bool:
    """
    Send project invitation email to a user who hasn't registered yet.
    Includes instructions to create an account.
    
    Returns True if sent successfully, False otherwise.
    """
    if not resend.api_key:
        print("[EMAIL] RESEND_API_KEY not set - skipping new user invitation email")
        return False
    
    try:
        params = {
            "from": f"{SENDER_NAME} <{SENDER_EMAIL}>",
            "to": [email],
            "subject": f"You've been invited to {project_name} on AgentCost",
            "html": _get_new_user_invitation_email_html(email, project_name, inviter_name, role),
        }
        
        response = resend.Emails.send(params)
        
        if response and response.get("id"):
            print(f"[EMAIL] New user invitation email sent to {email} (id: {response['id']})")
            return True
        
        print(f"[EMAIL] Failed to send new user invitation email to {email}: {response}")
        return False
        
    except Exception as e:
        print(f"[EMAIL] Error sending new user invitation email to {email}: {e}")
        return False
