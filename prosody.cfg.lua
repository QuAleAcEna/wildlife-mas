admins = { }
modules_enabled = {
  "roster"; "saslauth"; "tls"; "dialback"; "disco"; "carbons"; "pep"; "private";
  "blocklist"; "vcard4"; "vcard_legacy"; "version"; "uptime"; "time"; "ping"; "bosh";
}
allow_registration = true
c2s_require_encryption = false
s2s_require_encryption = false
allow_unencrypted_plain_auth = true
pidfile = "/var/run/prosody/prosody.pid"
authentication = "internal_hashed"
log = { info = "/dev/stdout" }

VirtualHost "localhost"
