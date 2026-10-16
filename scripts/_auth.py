import os


def auth_headers():
    """Authorization header only when API_KEY is set; an empty 'Bearer ' is an illegal value."""
    key = os.getenv("API_KEY", "").strip()
    return {"Authorization": "Bearer " + key} if key else {}
