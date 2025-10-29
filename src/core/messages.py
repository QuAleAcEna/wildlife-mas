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


def make_inform_alert(to_jid: str, payload: dict) -> Message:
    msg = Message(to=to_jid)
    msg.set_metadata("performative", INFORM)
    msg.set_metadata("type", ALERT_ANOMALY)
    msg.body = json_dumps(payload)
    return msg


def json_dumps(data):
    return json.dumps(data)


def json_loads(raw):
    return json.loads(raw)
