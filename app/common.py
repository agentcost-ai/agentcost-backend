"""Shared helpers (UUID generation, password rules).

Lives at app/ level to avoid circular imports between utils/ and models/.
"""

import re
import uuid


def generate_uuid() -> str:
    """Generate a UUID4 string for use as primary keys."""
    return str(uuid.uuid4())


def validate_password_strength(v: str) -> str:
    """
    Validate that a password meets minimum security requirements.

    Rules:
    - At least 8 characters
    - At least one uppercase letter
    - At least one lowercase letter
    - At least one digit

    Returns the password unchanged if valid.
    Raises ValueError with a descriptive message otherwise.
    """
    if len(v) < 8:
        raise ValueError('Password must be at least 8 characters')
    if not re.search(r'[A-Z]', v):
        raise ValueError('Password must contain at least one uppercase letter')
    if not re.search(r'[a-z]', v):
        raise ValueError('Password must contain at least one lowercase letter')
    if not re.search(r'[0-9]', v):
        raise ValueError('Password must contain at least one number')
    return v
