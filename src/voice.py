"""Voice alerts.

Why this replaces voice_alerts.py:
  * pyttsx3 speaks through the *server's* speakers. On a deployed app the
    visitor hears nothing. The browser's own speech engine is used instead.
  * Streamlit re-runs the script on every click, which made alerts repeat.
    `should_alert` fires only when the image or model actually changes.
  * pyttsx3 is still available as an optional "local mode" (LOCAL_VOICE=1).
"""
import json
import os
import threading

import streamlit.components.v1 as components

# Optional hand-written messages. Anything not listed gets a generic message.
ALERT_OVERRIDES = {
    "PEDESTRIAN_CROSSING": "Pedestrian crossing ahead. Watch out for people.",
    "STOP": "Stop sign ahead. Prepare to stop.",
    "CATTLE": "Cattle crossing ahead. Slow down.",
}


def alert_message(raw_label: str, display_name: str) -> str:
    return ALERT_OVERRIDES.get(raw_label, f"Warning. {display_name} ahead.")


def should_alert(state, key) -> bool:
    """True only the first time we see this (model, image) key in the session."""
    if state.get("last_alert_key") == key:
        return False
    state["last_alert_key"] = key
    return True


def browser_speech(message: str, autoplay: bool) -> None:
    """Render a tiny player in the page. The button is a fallback because some
    browsers block speech that is not started by a click inside the frame."""
    text = json.dumps(message)
    components.html(
        f"""
        <div style="font-family:sans-serif;font-size:14px;">
          <button id="b" style="padding:6px 12px;border:0;border-radius:6px;
                  background:#6D8196;color:#fff;cursor:pointer;">Play voice alert</button>
        </div>
        <script>
          const msg = {text};
          const say = () => {{ speechSynthesis.cancel();
                              speechSynthesis.speak(new SpeechSynthesisUtterance(msg)); }};
          document.getElementById("b").onclick = say;
          if ({str(autoplay).lower()}) {{ say(); }}
        </script>
        """,
        height=44,
    )


class LocalVoice:
    """Optional offline speech through pyttsx3 for running the app on your own PC."""

    enabled = os.getenv("LOCAL_VOICE") == "1"

    @classmethod
    def speak(cls, message: str) -> None:
        if not cls.enabled:
            return

        def _worker():
            try:
                import pyttsx3  # imported lazily so the server needs no audio stack
                engine = pyttsx3.init()
                engine.say(message)
                engine.runAndWait()
            except Exception:
                pass

        threading.Thread(target=_worker, daemon=True).start()