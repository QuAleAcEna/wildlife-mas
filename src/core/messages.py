"""Utility helpers for constructing and serialising SPADE message payloads."""

import json

from spade.message import Message

# Performative constants
INFORM = "inform"  # alerts, telemetry
CFP = "cfp"  # for CNP later
PROPOSE = "propose"
ACCEPT = "accept-proposal"
REJECT = "reject-proposal"

ALERT_ANOMALY = "alert.anomaly"
TELEMETRY = "telemetry"
# NOVO #
CNP_ALERT = "cnp.alert"
# NOVO #
DIRECT_ALERT = "direct.alert"


def make_inform_alert(to_jid: str, payload: dict) -> Message:
    """Create a standardized INFORM+ALERT message addressed to the ranger/drone."""
    msg = Message(to=to_jid)
    msg.set_metadata("performative", INFORM)
    msg.set_metadata("type", ALERT_ANOMALY)
    msg.body = json_dumps(payload)
    return msg


def json_dumps(data):
    """Serialise arbitrary data to JSON using project defaults."""
    return json.dumps(data)


def json_loads(raw):
    """Parse JSON payloads and bubble-up errors to callers when invalid."""
    return json.loads(raw)
