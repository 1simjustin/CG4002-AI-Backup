"""
Simple MQTT reader for debugging/monitoring sensor data streams.

Connects to the same MQTT broker as live_infer.py and prints incoming
sensor messages in a readable format. No model or inference dependencies.

Usage:
    python mqtt_reader.py
    python mqtt_reader.py --topic "inference/#"
    python mqtt_reader.py --broker 54.79.62.237 --port 8883
"""

import argparse
import json
import os
import ssl
import time

import numpy as np
import paho.mqtt.client as mqtt

# ================= MQTT CONFIG =================
BROKER = "54.79.62.237"
PORT = 8883
CA_CERT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mqtt-certs", "ca.crt")
SENSOR_TOPIC = "sensor/+/aggregated"

# Node order matching the 48-channel layout (8 IMU nodes x 6 features each).
NODE_ORDER = [
    "left_arm_upper_arm", "left_arm_forearm",
    "right_arm_upper_arm", "right_arm_forearm",
    "left_leg_thigh", "left_leg_shin",
    "right_leg_thigh", "right_leg_shin",
]

NODE_GROUP_MAP = {
    "left_arm":  ["left_arm_upper_arm", "left_arm_forearm"],
    "right_arm": ["right_arm_upper_arm", "right_arm_forearm"],
    "left_leg":  ["left_leg_thigh", "left_leg_shin"],
    "right_leg": ["right_leg_thigh", "right_leg_shin"],
}


def parse_frame(nodes_dict: dict) -> np.ndarray:
    """Convert MQTT nodes dict to a flat (48,) feature vector."""
    frame = np.zeros(48, dtype=np.float32)

    for group_name, part_names in NODE_GROUP_MAP.items():
        group_data = nodes_dict.get(group_name, {})
        for part_name in part_names:
            part_data = group_data.get(part_name)
            if part_data is None:
                continue

            idx = NODE_ORDER.index(part_name)
            accel = part_data.get("accel", [0, 0, 0])
            gyro = part_data.get("gyro", [0, 0, 0])
            offset = idx * 6
            frame[offset:offset + 3] = accel
            frame[offset + 3:offset + 6] = gyro

    return frame


def print_sensor_message(player, sequence, nodes_dict):
    """Print a readable summary of a sensor message."""
    frame = parse_frame(nodes_dict)

    print(f"\n[{player.upper()}] seq={sequence}")
    for i, node in enumerate(NODE_ORDER):
        offset = i * 6
        accel = frame[offset:offset + 3]
        gyro = frame[offset + 3:offset + 6]
        print(f"  {node:>25s}  accel=({accel[0]:8.2f},{accel[1]:8.2f},{accel[2]:8.2f})  "
              f"gyro=({gyro[0]:8.2f},{gyro[1]:8.2f},{gyro[2]:8.2f})")


def main():
    p = argparse.ArgumentParser(description="Simple MQTT reader for sensor data")
    p.add_argument("--broker", type=str, default=BROKER,
                   help="MQTT broker address")
    p.add_argument("--port", type=int, default=PORT,
                   help="MQTT broker port")
    p.add_argument("--ca_cert", type=str, default=CA_CERT,
                   help="Path to CA certificate for TLS")
    p.add_argument("--topic", type=str, default=SENSOR_TOPIC,
                   help="MQTT topic to subscribe to")
    p.add_argument("--raw", action="store_true", default=False,
                   help="Print raw JSON payloads instead of parsed output")
    args = p.parse_args()

    # -- MQTT callbacks --
    def on_connect(client, userdata, flags, rc):
        print(f"Connected to MQTT broker (rc={rc})")
        client.subscribe(args.topic, qos=0)
        print(f"Subscribed to {args.topic}")
        print(f"Waiting for messages...\n")

    def on_message(client, userdata, msg):
        if msg.retain:
            return

        try:
            payload = json.loads(msg.payload.decode())
        except json.JSONDecodeError:
            print(f"[{msg.topic}] (non-JSON) {msg.payload.decode()}")
            return

        if args.raw:
            print(f"[{msg.topic}] {json.dumps(payload, indent=2)}")
            return

        # Sensor topic: parse and print structured output
        parts = msg.topic.split("/")
        if len(parts) >= 3 and parts[0] == "sensor" and parts[2] == "aggregated":
            player = parts[1]
            sequence = payload.get("sequence", 0)
            nodes = payload.get("nodes", {})
            print_sensor_message(player, sequence, nodes)
        else:
            # Generic topic: just print the JSON
            print(f"[{msg.topic}] {json.dumps(payload, indent=2)}")

    # -- Connect and run --
    client = mqtt.Client(clean_session=True)
    client.reconnect_delay_set(min_delay=1, max_delay=5)

    if os.path.isfile(args.ca_cert):
        client.tls_set(ca_certs=args.ca_cert, tls_version=ssl.PROTOCOL_TLS_CLIENT)
        client.tls_insecure_set(True)
        print(f"TLS enabled with CA cert: {args.ca_cert}")
    else:
        print(f"WARNING: CA cert not found at {args.ca_cert}, connecting without TLS")

    client.on_connect = on_connect
    client.on_message = on_message

    print(f"Connecting to MQTT broker at {args.broker}:{args.port}...")
    try:
        client.connect(args.broker, args.port)
        client.loop_forever()
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()
