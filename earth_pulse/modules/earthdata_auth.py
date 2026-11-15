"""Earthdata authentication. Credentials come from environment variables or
Streamlit secrets only. Nothing is hardcoded, logged or written to disk.

Env vars (token preferred):  EARTHDATA_TOKEN  or  EARTHDATA_USERNAME + EARTHDATA_PASSWORD
"""
import os


class AuthError(Exception):
    pass


def redact(text, secrets=()):
    out = str(text)
    for s in secrets:
        if s:
            out = out.replace(s, "***REDACTED***")
    return out


_KEYS = (("token", "EARTHDATA_TOKEN"), ("username", "EARTHDATA_USERNAME"),
         ("password", "EARTHDATA_PASSWORD"))


def get_credentials():
    """Return dict(token, username, password) from env, then Streamlit secrets."""
    creds = {k: os.environ.get(v) for k, v in _KEYS}
    if not (creds["token"] or (creds["username"] and creds["password"])):
        try:  # only inside a Streamlit app that defines secrets
            import streamlit as st
            for k, v in _KEYS:
                if v in st.secrets:
                    creds[k] = st.secrets[v]
        except Exception:
            pass
    return creds


def credentials_available():
    c = get_credentials()
    return bool(c["token"] or (c["username"] and c["password"]))


def build_asf_session():
    """Authenticated ASFSession (needed for downloads; public search does not need it)."""
    c = get_credentials()
    if not credentials_available():
        raise AuthError(
            "No Earthdata credentials found. Set EARTHDATA_TOKEN (or EARTHDATA_USERNAME and "
            "EARTHDATA_PASSWORD) as environment variables or in .streamlit/secrets.toml. "
            "Create a free account/token at https://urs.earthdata.nasa.gov/.")
    try:
        import asf_search as asf
    except ImportError as e:
        raise AuthError("asf_search is not installed: pip install -r requirements.txt") from e
    secrets = (c["token"], c["password"])
    try:
        session = asf.ASFSession()
        if c["token"]:
            session.auth_with_token(c["token"])
        else:
            session.auth_with_creds(c["username"], c["password"])
        return session
    except Exception as e:  # never leak credentials through the message
        raise AuthError("Earthdata authentication failed: "
                        + redact(f"{type(e).__name__}: {e}", secrets)) from None


_EA = {"auth": None}


def earthaccess_login():
    """Log in with earthaccess using credentials from env/Streamlit secrets. Credentials are only held
    in this process; nothing is written to disk (strategy='environment', no netrc persistence)."""
    if _EA["auth"] is not None:
        return _EA["auth"]
    c = get_credentials()
    if not credentials_available():
        raise AuthError("No Earthdata credentials found. Set EARTHDATA_TOKEN (or EARTHDATA_USERNAME and "
                        "EARTHDATA_PASSWORD) in the environment or .streamlit/secrets.toml.")
    try:
        import earthaccess
    except ImportError as e:
        raise AuthError("earthaccess is not installed: pip install -r requirements.txt") from e
    for k, env in (("token", "EARTHDATA_TOKEN"), ("username", "EARTHDATA_USERNAME"), ("password", "EARTHDATA_PASSWORD")):
        if c[k]:
            os.environ[env] = c[k]           # this process only
    secrets = (c["token"], c["password"])
    try:
        auth = earthaccess.login(strategy="environment")
    except Exception as e:
        raise AuthError("Earthdata login failed: " + redact(f"{type(e).__name__}: {e}", secrets)) from None
    if not getattr(auth, "authenticated", True):
        raise AuthError("Earthdata login was not accepted. Check your token/username and password.")
    _EA["auth"] = auth
    return auth
