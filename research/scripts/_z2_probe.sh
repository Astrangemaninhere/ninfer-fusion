#!/bin/bash
# Probe Windows host for a LAN-reachable proxy port (WSL cannot reach 127.0.0.1 of host).
HOST=172.30.128.1
echo "=== probing $HOST ==="
for p in 10808 10809 10810 7890 7891 1080 8080 8888 1087 20171 20172; do
  if timeout 3 bash -c "exec 3<>/dev/tcp/$HOST/$p" 2>/dev/null; then
    echo "OPEN   $p"
  else
    echo "closed $p"
  fi
done
echo "=== also try host.docker.internal style name ==="
getent hosts host.docker.internal 2>/dev/null || echo "no host.docker.internal"
echo "=== windows host ip via resolv ==="
ip route get 1.1.1.1 2>/dev/null | head -2
