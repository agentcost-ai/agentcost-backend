"""Project invitation template (new / unregistered users)."""

from ._base import base_wrapper, cta_button, esc
from .invitation import ROLE_DESCRIPTIONS


def get_new_user_invitation_email_html(
    email: str,
    project_name: str,
    inviter_name: str,
    role: str,
    register_link: str,
) -> str:
    safe_email = esc(email)
    safe_project = esc(project_name)
    safe_inviter = esc(inviter_name)
    safe_role = esc(role)
    role_desc = ROLE_DESCRIPTIONS.get(role.lower(), "access to the project")

    body = f"""\
    <tr>
        <td style="padding: 40px;">
            <h2 style="margin: 0 0 16px 0; font-size: 22px; font-weight: 600; color: #ffffff;">You&rsquo;ve been invited to join a project</h2>
            <p style="margin: 0 0 24px 0; font-size: 15px; line-height: 1.6; color: #a1a1aa;">Hello,</p>
            <p style="margin: 0 0 24px 0; font-size: 15px; line-height: 1.6; color: #a1a1aa;"><strong style="color: #ffffff;">{safe_inviter}</strong> has invited you to join <strong style="color: #ffffff;">{safe_project}</strong> on AgentCost, an LLM cost tracking platform.</p>
            <div style="margin: 24px 0; padding: 20px; background-color: #27272a; border-radius: 8px;">
                <p style="margin: 0 0 8px 0; font-size: 13px; color: #71717a; text-transform: uppercase; letter-spacing: 0.5px;">Your Role</p>
                <p style="margin: 0; font-size: 18px; font-weight: 600; color: #ffffff; text-transform: capitalize;">{safe_role}</p>
                <p style="margin: 8px 0 0 0; font-size: 13px; color: #a1a1aa;">As a {safe_role}, you&rsquo;ll have {role_desc}.</p>
            </div>
            <p style="margin: 0 0 16px 0; font-size: 15px; line-height: 1.6; color: #a1a1aa;">To accept this invitation, create an AgentCost account using this email address (<strong style="color: #ffffff;">{safe_email}</strong>).</p>
            <p style="margin: 0 0 32px 0; font-size: 15px; line-height: 1.6; color: #a1a1aa;">Once you&rsquo;ve registered and verified your email, the invitation will be waiting for you in your dashboard.</p>
            {cta_button(register_link, "Create Account")}
            <p style="margin: 32px 0 0 0; font-size: 13px; line-height: 1.6; color: #71717a;">If you don&rsquo;t want to join this project, you can ignore this email. The invitation will remain pending until you register.</p>
        </td>
    </tr>"""

    return base_wrapper(body)
