from .guitar_transcriber import MonophonicGuitarTabber

NODE_CLASS_MAPPINGS = {
    "MonophonicGuitarTabber": MonophonicGuitarTabber,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MonophonicGuitarTabber": "🎸 Audio to Guitar Tab (Mono)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
