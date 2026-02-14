DO $$
DECLARE
    v_admin_email TEXT := current_setting('app.admin_email', true);
    v_admin_hash  TEXT := current_setting('app.admin_password_hash', true);
    v_admin_name  TEXT := COALESCE(current_setting('app.admin_name', true), 'Admin');
BEGIN
    -- Fail fast if required variables are not provided
    IF v_admin_email IS NULL THEN
        RAISE EXCEPTION 'admin_email is required. Pass via: -v admin_email="''your@email.com''"';
    END IF;
    IF v_admin_hash IS NULL THEN
        RAISE EXCEPTION 'admin_password_hash is required. Generate one with: python -c "from passlib.context import CryptContext; print(CryptContext(schemes=[''bcrypt'']).hash(''YourPassword''))"';
    END IF;

    INSERT INTO users (
        id,
        email,
        password_hash,
        name,
        auth_provider,
        email_verified,
        is_active,
        is_superuser,
        is_deleted,
        -- Explicitly NULL: no badge, no user_number, no milestone for admins
        user_number,
        milestone_badge,
        created_at,
        updated_at
    ) VALUES (
        gen_random_uuid()::text,
        lower(trim(v_admin_email)),
        v_admin_hash,
        v_admin_name,
        'email',
        true,       -- Admin email is pre-verified
        true,       -- Active immediately
        true,       -- Superuser flag
        false,      -- Not deleted
        NULL,       -- No user_number for admins
        NULL,       -- No milestone badge for admins
        NOW(),
        NOW()
    )
    ON CONFLICT (email) DO UPDATE SET
        is_superuser = true,
        is_active = true,
        is_deleted = false,
        updated_at = NOW()
    WHERE users.is_superuser = false;  -- Only update if not already a superuser

    -- Report what happened
    IF FOUND THEN
        RAISE NOTICE 'Admin account created or updated: %', v_admin_email;
    ELSE
        RAISE NOTICE 'Admin account already exists and is already a superuser: %', v_admin_email;
    END IF;
END $$;
