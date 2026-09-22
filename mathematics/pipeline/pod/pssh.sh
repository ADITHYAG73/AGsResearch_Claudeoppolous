#!/bin/bash
# pssh.sh <host> <port> <cmd...>   |   pssh.sh --scp <host> <port> <local files...> <remote path>
K=$HOME/.ssh/id_ed25519_runpod_new
OPTS=(-i "$K" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -o BatchMode=yes)
if [ "$1" = "--scp" ]; then shift; H=$1; P=$2; shift 2; scp -P "$P" "${OPTS[@]}" "$@"; exit $?; fi
H=$1; P=$2; shift 2; ssh -p "$P" "${OPTS[@]}" root@"$H" "$@"
