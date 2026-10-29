#!/usr/bin/env bash
# 配置 franky/libfranka 所需的实时权限。
# 幂等：可重复执行，不会重复追加。
set -euo pipefail

TARGET_USER="${SUDO_USER:-$(whoami)}"
LIMITS=/etc/security/limits.conf
MARKER="# --- franky realtime limits ---"

echo "==> 目标用户: ${TARGET_USER}"

# 1) realtime 组
if getent group realtime >/dev/null; then
  echo "==> 组 'realtime' 已存在，跳过创建"
else
  addgroup realtime
  echo "==> 已创建组 'realtime'"
fi

# 2) 把用户加进去
if id -nG "${TARGET_USER}" | tr ' ' '\n' | grep -qx realtime; then
  echo "==> 用户 ${TARGET_USER} 已在 realtime 组中，跳过"
else
  usermod -a -G realtime "${TARGET_USER}"
  echo "==> 已将 ${TARGET_USER} 加入 realtime 组"
fi

# 3) limits.conf
if grep -qF "${MARKER}" "${LIMITS}"; then
  echo "==> limits.conf 已配置过，跳过"
else
  cp -a "${LIMITS}" "${LIMITS}.bak.$(date +%Y%m%d%H%M%S)"
  echo "==> 已备份 ${LIMITS}"
  cat >> "${LIMITS}" <<'EOF'

# --- franky realtime limits ---
@realtime soft rtprio 99
@realtime soft priority 99
@realtime soft memlock 102400
@realtime hard rtprio 99
@realtime hard priority 99
@realtime hard memlock 102400
EOF
  echo "==> 已追加实时限额到 ${LIMITS}"
fi

echo
echo "==> 完成。当前 limits.conf 中的 franky 配置："
grep -A7 -F "${MARKER}" "${LIMITS}"
echo
echo "==> 注意：组成员身份需要【重新登录或重启】后才生效。"
echo "    RT 内核装完重启时会一并生效，无需额外重启。"
