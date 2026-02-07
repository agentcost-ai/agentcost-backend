"""Project invitation template (existing users)."""

from typing import Optional
from ._base import base_wrapper, cta_button, esc

ROLE_DESCRIPTIONS = {
    "admin": "full access to manage the project, invite members, and view all data",
    "member": "access to view analytics, create events, and export data",
    "viewer": "read-only access to view project analytics and events",
}


def get_invitation_email_html(
    invitee_name: Optional[str],
    project_name: str,
    inviter_name: str,
    role: str,
    dashboard_link: str,
) -> str:
    display_name = esc(invitee_name) if invitee_name else "there"
    safe_project = esc(project_name)
    safe_inviter = esc(inviter_name)
    safe_role = esc(role)
    role_desc = ROLE_DESCRIPTIONS.get(role.lower(), "access to the project")

    body = f"""\
    <tr>
        <td style="padding: 40px;">
            <h2 style="margin: 0 0 16px 0; font-size: 22px; font-weight: 600; color: #ffffff;">You&rsquo;ve been invited to a project</h2>
            <p style="margin: 0 0 24px 0; font-size: 15px; line-height: 1.6; color: #a1a1aa;">Hey {display_name},</p>
            <p style="margin: 0 0 24px 0; font-size: 15px; line-height: 1.6; color: #a1a1aa;"><strong style="color: #ffffff;">{safe_inviter}</strong> has invited you to join <strong style="color: #ffffff;">{safe_project}</strong> on AgentCost.</p>
            <div style="margin: 24px 0; padding: 20px; background-color: #27272a; border-radius: 8px;">
                <p style="margin: 0 0 8px 0; font-size: 13px; color: #71717a; text-transform: uppercase; letter-spacing: 0.5px;">Your Role</p>
                <p style="margin: 0; font-size: 18px; font-weight: 600; color: #ffffff; text-transform: capitalize;">{safe_role}</p>
                <p style="margin: 8px 0 0 0; font-size: 13px; color: #a1a1aa;">As a {safe_role}, you&rsquo;ll have {role_desc}.</p>
            </div>
            <p style="margin: 0 0 32px 0; font-size: 15px; line-height: 1.6; color: #a1a1aa;">Log in to your AgentCost dashboard to accept this invitation:</p>
            {cta_button(dashboard_link, "View Invitation")}
            <p style="margin: 32px 0 0 0; font-size: 13px; line-height: 1.6; color: #71717a;">If you don&rsquo;t want to join this project, you can ignore this email or decline the invitation from your dashboard.</p>
        </td>
    </tr>"""

    return base_wrapper(body)
