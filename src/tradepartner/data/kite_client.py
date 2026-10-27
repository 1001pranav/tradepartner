"""Builds an authenticated KiteConnect client from Keychain credentials."""

from kiteconnect import KiteConnect

from tradepartner.data import kite_auth

# Index tokens used for relative/regime features. [VERIFY] against the instrument master.
INDEX_TOKENS = {"NIFTY 50": 256265, "INDIA VIX": 264969}


class NotLoggedInError(Exception):
    pass


def login_client() -> KiteConnect:
    """Client without an access token, used only to start the login flow."""
    creds = kite_auth.load_credentials()
    if creds is None:
        raise NotLoggedInError("no Kite api key/secret in Keychain; run `tradepartner login`")
    return KiteConnect(api_key=creds[0])


def get_kite() -> KiteConnect:
    kite = login_client()
    token = kite_auth.load_access_token()
    if token is None:
        raise NotLoggedInError("Kite access token missing or expired; run `tradepartner login`")
    kite.set_access_token(token)
    return kite
