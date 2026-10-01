#!/usr/bin/env bash
# scripts/renew-tailscale-cert.sh
# Issue/renew this machine's Tailscale HTTPS certificate (Let's Encrypt, trusted
# by phones without installing a CA) into caddy/certs/, and restart Caddy when
# it changed. Run on a timer: Tailscale certs last 90 days and `tailscale cert`
# only reissues when renewal is due.
#
# Needs HTTPS enabled for the tailnet (admin console → DNS → HTTPS Certificates)
# and either root or `sudo tailscale set --operator=$USER`.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CERT_DIR="${REPO_DIR}/caddy/certs"
CADDY_CONTAINER="${CADDY_CONTAINER:-assistant-caddy}"

DOMAIN="${1:-$(tailscale status --json | python3 -c 'import json,sys; print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))')}"

mkdir -p "${CERT_DIR}"
before="$(sha256sum "${CERT_DIR}/cert.pem" 2>/dev/null || true)"

tailscale cert --cert-file "${CERT_DIR}/cert.pem" --key-file "${CERT_DIR}/key.pem" "${DOMAIN}"
chmod 600 "${CERT_DIR}/key.pem"

after="$(sha256sum "${CERT_DIR}/cert.pem")"
if [[ "${before}" != "${after}" ]]; then
    echo "[ok] certificate for ${DOMAIN} updated"
    # Caddyfile has `admin off`, so reload by restarting the container.
    if docker inspect "${CADDY_CONTAINER}" >/dev/null 2>&1; then
        docker restart "${CADDY_CONTAINER}" >/dev/null
        echo "[ok] restarted ${CADDY_CONTAINER}"
    fi
else
    echo "[skip] certificate for ${DOMAIN} unchanged"
fi
