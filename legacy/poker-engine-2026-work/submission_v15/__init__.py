# v15 — avoid importing PlayerAgent here to prevent circular import.
__all__ = ["PlayerAgent"]


def __getattr__(name):
    if name == "PlayerAgent":
        from submission.player import PlayerAgent
        return PlayerAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
