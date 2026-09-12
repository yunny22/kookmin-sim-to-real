#!/usr/bin/env bash
set -u

pass() { printf 'PASS %s\n' "$*"; }
warn() { printf 'WARN %s\n' "$*"; }
fail() { printf 'FAIL %s\n' "$*"; }

TOPIC_LIST=""

check_topic_exists() {
  local topic="$1"
  if printf '%s\n' "$TOPIC_LIST" | grep -qx "$topic"; then
    pass "topic exists: $topic"
    return 0
  fi
  fail "topic missing: $topic"
  return 1
}

check_hz() {
  local topic="$1"
  if ! printf '%s\n' "$TOPIC_LIST" | grep -qx "$topic"; then
    warn "skip hz; topic missing: $topic"
    return 0
  fi
  if timeout 5s ros2 topic hz "$topic" >/tmp/il_topic_hz.txt 2>&1; then
    pass "topic hz responded: $topic"
  else
    warn "topic hz did not finish cleanly within 5s: $topic"
  fi
  sed 's/^/  /' /tmp/il_topic_hz.txt | tail -n 5
}

echo "== ros2 topic list =="
if TOPIC_LIST="$(ros2 topic list)"; then
  printf '%s\n' "$TOPIC_LIST"
  pass "ros2 topic list"
else
  fail "ros2 topic list failed"
fi

echo
echo "== xycar_msgs interface =="
if ros2 interface show xycar_msgs/msg/XycarMotor >/tmp/il_xycar_msgs_interface.txt 2>&1; then
  pass "xycar_msgs/msg/XycarMotor found"
else
  warn "xycar_msgs not found. This is OK when /xycar_motor uses std_msgs/msg/Float32MultiArray. Use motor_msg_type:=xycar only when /xycar_motor is xycar_msgs/msg/XycarMotor."
fi

echo
check_topic_exists "/image_raw"
check_topic_exists "/scan"
check_topic_exists "/imu"
check_topic_exists "/xycar_motor"

echo
check_hz "/image_raw"
check_hz "/scan"
check_hz "/imu"

echo
echo "== /xycar_motor info =="
if printf '%s\n' "$TOPIC_LIST" | grep -qx "/xycar_motor"; then
  if INFO="$(ros2 topic info /xycar_motor -v)"; then
    printf '%s\n' "$INFO"
    pass "topic info /xycar_motor"
    PUBLISHERS="$(printf '%s\n' "$INFO" | awk '/Publisher count:/ {print $3; exit}')"
    if [ -n "$PUBLISHERS" ] && [ "$PUBLISHERS" -gt 1 ]; then
      warn "/xycar_motor has more than one publisher: $PUBLISHERS"
    fi
  else
    fail "topic info /xycar_motor failed"
  fi
else
  warn "skip topic info /xycar_motor; topic is not currently available"
fi

echo
echo "== disk =="
if df -h "$HOME/xycar_ws"; then
  pass "df -h ~/xycar_ws"
else
  warn "df -h ~/xycar_ws failed"
fi
