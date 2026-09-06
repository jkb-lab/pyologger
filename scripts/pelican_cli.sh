#!/usr/bin/env bash
# Thin wrapper around the `pelican` CLI for the jkb-lab / jkb-lab-public
# namespaces. Mints a short-lived token on demand for private-namespace or
# write operations; public reads need no token at all.
#
# Usage:
#   scripts/pelican_cli.sh list   [<subpath>] [--private]
#   scripts/pelican_cli.sh put    <local-file> <dest-path> [--private]
#   scripts/pelican_cli.sh get    <src-path> <local-dest> [--private]
#
# <dest-path> / <src-path> are relative to the chosen namespace, e.g.
#   scripts/pelican_cli.sh put ./demo.nc demo/2020-04-10_mian-002.nc
#   scripts/pelican_cli.sh get demo/2020-04-10_mian-002.nc ./demo.nc
#   scripts/pelican_cli.sh list demo -l
#
# --private targets jkb-lab (token-gated read+write) instead of the default
# jkb-lab-public (open read, token-gated write). A token is minted fresh for
# any operation that needs one and deleted immediately after.

set -euo pipefail

PELICAN_BIN="${PELICAN_BIN:-pelican}"
NAMESPACE="jkb-lab-public"
TOKEN_LIFETIME=3600  # seconds; short-lived, minted per invocation

usage() {
    grep '^#' "$0" | cut -c3-
    exit 1
}

[ $# -ge 1 ] || usage
CMD="$1"; shift

# Pull --private out of the remaining args, wherever it lands.
ARGS=()
for a in "$@"; do
    if [ "$a" = "--private" ]; then
        NAMESPACE="jkb-lab"
    else
        ARGS+=("$a")
    fi
done
set -- "${ARGS[@]+"${ARGS[@]}"}"

TOKEN_FILE=""
cleanup() { if [ -n "$TOKEN_FILE" ]; then rm -f "$TOKEN_FILE"; fi; }
trap cleanup EXIT

mint_token() {
    TOKEN_FILE="$(mktemp)"
    "$PELICAN_BIN" origin token create \
        --scope "storage.read:/ storage.write:/ storage.modify:/" \
        --issuer "https://osg-htc.org/osdf/jkb-lab" \
        --subject anything \
        --audience "https://wlcg.cern.ch/jwt/v1/any" \
        --lifetime "$TOKEN_LIFETIME" -d > "$TOKEN_FILE" 2>/dev/null
}

case "$CMD" in
    list)
        SUBPATH="${1:-}"; shift || true
        if [ "$NAMESPACE" = "jkb-lab" ]; then
            mint_token
            "$PELICAN_BIN" object ls "osdf:///$NAMESPACE/$SUBPATH" -t "$TOKEN_FILE" "$@"
        else
            "$PELICAN_BIN" object ls "osdf:///$NAMESPACE/$SUBPATH" "$@"
        fi
        ;;

    put)
        [ $# -ge 2 ] || usage
        LOCAL_FILE="$1"; DEST_PATH="$2"; shift 2
        mint_token
        "$PELICAN_BIN" object put "$LOCAL_FILE" \
            "pelican://osg-htc.org/$NAMESPACE/$DEST_PATH" \
            --token "$TOKEN_FILE" "$@"
        echo "Uploaded to osdf:///$NAMESPACE/$DEST_PATH"
        ;;

    get)
        [ $# -ge 2 ] || usage
        SRC_PATH="$1"; LOCAL_DEST="$2"; shift 2
        if [ "$NAMESPACE" = "jkb-lab" ]; then
            mint_token
            "$PELICAN_BIN" object get "osdf:///$NAMESPACE/$SRC_PATH" "$LOCAL_DEST" -t "$TOKEN_FILE" "$@"
        else
            "$PELICAN_BIN" object get "osdf:///$NAMESPACE/$SRC_PATH" "$LOCAL_DEST" "$@"
        fi
        ;;

    *)
        usage
        ;;
esac
