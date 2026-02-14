"""
AgentCost - Admin User Management CLI

Usage:
    # Create the initial superuser (interactive)
    python -m scripts.manage_admin create

    # Create with inline args (CI/scripting)
    python -m scripts.manage_admin create --email admin@yourdomain.com --password '<your-password>'

    # Promote an existing user to superuser
    python -m scripts.manage_admin promote --email user@example.com

    # Demote a superuser back to normal user
    python -m scripts.manage_admin demote --email admin@agentcost.tech

    # List all superusers
    python -m scripts.manage_admin list
"""

import argparse
import asyncio
import getpass
import sys
import os
import re

# Add project root to path so we can import app modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import select, update
from app.database import async_session_maker, create_tables
from app.models.user_models import User
from app.services.auth_service import hash_password


def validate_email(email: str) -> bool:
    return bool(re.match(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$", email))


def validate_password(password: str) -> list[str]:
    """Return list of password policy violations."""
    issues = []
    if len(password) < 10:
        issues.append("Must be at least 10 characters")
    if not re.search(r"[A-Z]", password):
        issues.append("Must contain at least one uppercase letter")
    if not re.search(r"[a-z]", password):
        issues.append("Must contain at least one lowercase letter")
    if not re.search(r"[0-9]", password):
        issues.append("Must contain at least one digit")
    if not re.search(r"[^A-Za-z0-9]", password):
        issues.append("Must contain at least one special character")
    return issues


async def create_superuser(email: str | None, password: str | None, name: str | None):
    """Create a new superuser account or promote existing."""
    await create_tables()

    if not email:
        email = input("Email: ").strip()
    if not validate_email(email):
        print(f"ERROR: Invalid email format: {email}")
        sys.exit(1)

    async with async_session_maker() as db:
        # Check if user already exists
        existing = (await db.execute(
            select(User).where(User.email == email.lower())
        )).scalar_one_or_none()

        if existing:
            if existing.is_superuser:
                print(f"User {email} is already a superuser.")
                sys.exit(0)
            # Promote existing user
            existing.is_superuser = True
            existing.is_active = True
            await db.commit()
            print(f"Existing user {email} promoted to superuser.")
            return

        # New user — get password
        if not password:
            print("Set password for the new superuser account.")
            print("Requirements: 10+ chars, uppercase, lowercase, digit, special character")
            password = getpass.getpass("Password: ")
            confirm = getpass.getpass("Confirm password: ")
            if password != confirm:
                print("ERROR: Passwords do not match.")
                sys.exit(1)

        issues = validate_password(password)
        if issues:
            print("ERROR: Password does not meet requirements:")
            for issue in issues:
                print(f"  - {issue}")
            sys.exit(1)

        if not name:
            name = input("Display name (optional, press Enter to skip): ").strip() or None

        user = User(
            email=email.lower().strip(),
            password_hash=hash_password(password),
            name=name,
            is_superuser=True,
            is_active=True,
            email_verified=True,
        )
        db.add(user)
        await db.commit()

        print("")
        print("=" * 50)
        print("  SUPERUSER CREATED SUCCESSFULLY")
        print("=" * 50)
        print(f"  Email:    {email}")
        print(f"  Name:     {name or '(not set)'}")
        print(f"  Admin:    Yes")
        print(f"  Active:   Yes")
        print("=" * 50)
        print("")
        print("You can now log in at the admin panel.")


async def promote_user(email: str | None):
    """Promote an existing user to superuser."""
    await create_tables()

    if not email:
        email = input("Email of user to promote: ").strip()

    async with async_session_maker() as db:
        user = (await db.execute(
            select(User).where(User.email == email.lower())
        )).scalar_one_or_none()

        if not user:
            print(f"ERROR: No user found with email {email}")
            sys.exit(1)

        if user.is_superuser:
            print(f"User {email} is already a superuser.")
            sys.exit(0)

        user.is_superuser = True
        await db.commit()
        print(f"User {email} has been promoted to superuser.")


async def demote_user(email: str | None):
    """Remove superuser privileges from a user."""
    await create_tables()

    if not email:
        email = input("Email of user to demote: ").strip()

    async with async_session_maker() as db:
        user = (await db.execute(
            select(User).where(User.email == email.lower())
        )).scalar_one_or_none()

        if not user:
            print(f"ERROR: No user found with email {email}")
            sys.exit(1)

        if not user.is_superuser:
            print(f"User {email} is not a superuser.")
            sys.exit(0)

        # Safety: check we're not demoting the last superuser
        total_admins = (await db.execute(
            select(User).where(User.is_superuser == True, User.is_active == True)
        )).scalars().all()

        if len(total_admins) <= 1:
            print("ERROR: Cannot demote the last remaining superuser.")
            print("Create another superuser first.")
            sys.exit(1)

        user.is_superuser = False
        await db.commit()
        print(f"User {email} has been demoted from superuser.")


async def list_superusers():
    """List all superuser accounts."""
    await create_tables()

    async with async_session_maker() as db:
        superusers = (await db.execute(
            select(User).where(User.is_superuser == True)
        )).scalars().all()

        if not superusers:
            print("No superuser accounts found.")
            print("")
            print("Create one with:")
            print("  python -m scripts.manage_admin create")
            return

        print(f"\n{'Email':<35} {'Name':<20} {'Active':<8} {'Created'}")
        print("-" * 90)
        for u in superusers:
            created = u.created_at.strftime("%Y-%m-%d %H:%M") if u.created_at else "N/A"
            print(f"{u.email:<35} {(u.name or '-'):<20} {'Yes' if u.is_active else 'No':<8} {created}")
        print(f"\nTotal: {len(superusers)} superuser(s)")


def main():
    parser = argparse.ArgumentParser(
        description="AgentCost Admin User Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m scripts.manage_admin create
  python -m scripts.manage_admin create --email admin@yourdomain.com --password '<your-password>'
  python -m scripts.manage_admin promote --email user@example.com
  python -m scripts.manage_admin demote --email admin@old.com
  python -m scripts.manage_admin list
        """,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # create
    create_parser = subparsers.add_parser("create", help="Create a new superuser account")
    create_parser.add_argument("--email", type=str, help="Superuser email address")
    create_parser.add_argument("--password", type=str, help="Password (omit for interactive prompt)")
    create_parser.add_argument("--name", type=str, help="Display name")

    # promote
    promote_parser = subparsers.add_parser("promote", help="Promote existing user to superuser")
    promote_parser.add_argument("--email", type=str, help="Email of user to promote")

    # demote
    demote_parser = subparsers.add_parser("demote", help="Remove superuser from existing admin")
    demote_parser.add_argument("--email", type=str, help="Email of user to demote")

    # list
    subparsers.add_parser("list", help="List all superuser accounts")

    args = parser.parse_args()

    if args.command == "create":
        asyncio.run(create_superuser(args.email, args.password, getattr(args, "name", None)))
    elif args.command == "promote":
        asyncio.run(promote_user(args.email))
    elif args.command == "demote":
        asyncio.run(demote_user(args.email))
    elif args.command == "list":
        asyncio.run(list_superusers())


if __name__ == "__main__":
    main()
