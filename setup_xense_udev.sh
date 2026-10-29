#!/bin/bash
# 千觉 Xense 触觉传感器 USB 权限配置（幂等，可重复执行）
#
# 不配这个规则，传感器的 USB 设备节点只有 root 可读写，
# Sensor.scanSerialNumber() 会返回空字典 {}。
#
# 用法：
#   bash setup_xense_udev.sh
#   然后重启电脑（udev 规则对已插入的设备不会追溯生效）
#
# 出处：xensesdk README「步骤 4: ubuntu环境注意事项」

set -e

RULE_FILE=/etc/udev/rules.d/99-xense.rules

# 匹配 vendor id 3938（千觉所有当前和未来设备通用）。
# GROUP 用当前用户的同名组 —— Ubuntu 默认为每个用户建同名组，
# 使用当前用户的主组。
sudo tee "$RULE_FILE" > /dev/null <<EOF
# 99-xense.rules - allow user '${USER}' to access Xense Robotics USB devices
SUBSYSTEM=="usb", ENV{DEVTYPE}=="usb_device", ATTR{idVendor}=="3938", MODE="0660", GROUP="$(id -gn)"
EOF

sudo udevadm control --reload-rules
sudo udevadm trigger

echo "✅ 已写入 $RULE_FILE"
echo
cat "$RULE_FILE"
echo
echo "⚠️  请重启电脑，然后验证："
echo "     lsusb | grep -i 3938"
echo "     conda activate franka_tele"
echo "     python -c \"from xensesdk import Sensor; print(Sensor.scanSerialNumber())\""
