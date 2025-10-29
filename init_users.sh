#!/usr/bin/env bash

set -euo pipefail

log() {
  echo "[init-users] $*"
}

register_user() {
  local raw_jid="$1"
  local password="$2"

  if [[ -z "$raw_jid" ]]; then
    log "Skipping entry with empty JID."
    return 0
  fi
  if [[ -z "$password" ]]; then
    log "Skipping ${raw_jid}: missing password."
    return 0
  fi

  local user="${raw_jid%@*}"
  local domain="${raw_jid#*@}"

  if [[ "$raw_jid" != *@* ]]; then
    domain="${LOCAL:-localhost}"
  fi

  if prosodyctl list-users "$domain" 2>/dev/null | grep -Fxq "$user"; then
    log "${raw_jid} already exists; skipping."
    return 0
  fi

  log "Registering ${raw_jid}."
  prosodyctl register "$user" "$domain" "$password"
}

# Defaults align with .env but allow overrides via environment.
SENSOR_JID="${SENSOR_JID:-sensor@localhost}"
SENSOR_PASS="${SENSOR_PASS:-sensor123}"
DRONE_JID="${DRONE_JID:-drone@localhost}"
DRONE_PASS="${DRONE_PASS:-drone123}"
RANGER_JID="${RANGER_JID:-ranger@localhost}"
RANGER_PASS="${RANGER_PASS:-ranger123}"

register_user "$SENSOR_JID" "$SENSOR_PASS"
register_user "$DRONE_JID" "$DRONE_PASS"
register_user "$RANGER_JID" "$RANGER_PASS"
