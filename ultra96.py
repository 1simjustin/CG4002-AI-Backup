"""
Ultra96 FPGA Inference Node — MQTT + Siamese LSTM Accelerator
Receives IMU data from shifu/student via MQTT, accumulates sliding windows,
runs FPGA inference, and publishes per-node similarity scores.
"""

import paho.mqtt.client as mqtt
import ssl
import json
import numpy as np
import os
import mmap
import struct
import time
import subprocess
from collections import deque

# ================= CONFIG =================
BROKER = "54.79.62.237"
PORT = 8883
CA_CERT = "/home/xilinx/mqtt-certs/ca.crt"
SENSOR_TOPIC = "sensor/+/aggregated"
INFERENCE_TOPIC = "inference/result"

# ================= FPGA CONFIG =================
BIT_FILE = "/home/xilinx/siamese_lstm.bit"
HLS_BASE = 0xA0000000
SEQ1_PHYS = 0x70000000
SEQ2_PHYS = 0x70100000
RESULT_PHYS = 0x70200000
PAGE_SIZE = 4096
NUM_OUTPUTS = 9

# ================= MODEL CONFIG =================
WINDOW_SIZE = 50       # frames to accumulate before inference (1 second at 50Hz)
INPUT_DIM = 48         # 8 nodes × 6 features
MIN_FRAMES = 20        # minimum frames before first inference

# Node order must match HLS model
NODE_ORDER = [
    "left_arm_upper_arm", "left_arm_forearm",
    "right_arm_upper_arm", "right_arm_forearm",
    "left_leg_thigh", "left_leg_shin",
    "right_leg_thigh", "right_leg_shin",
]

# Maps MQTT node groups → body part names
NODE_GROUP_MAP = {
    "left_arm":  ["left_arm_upper_arm", "left_arm_forearm"],
    "right_arm": ["right_arm_upper_arm", "right_arm_forearm"],
    "left_leg":  ["left_leg_thigh", "left_leg_shin"],
    "right_leg": ["right_leg_thigh", "right_leg_shin"],
}

NODE_NAMES_SHORT = [
    "left_arm_upper_arm", "left_arm_forearm", "right_arm_upper_arm", "right_arm_forearm",
    "left_leg_thigh", "left_leg_shin", "right_leg_thigh", "right_leg_shin",
]


# ================= FPGA DRIVER =================
class FPGAInference:
    def __init__(self, bitfile):
        self.program_fpga(bitfile)
        self.fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
        self.mm = mmap.mmap(self.fd, 0x10000, mmap.MAP_SHARED,
                            mmap.PROT_READ | mmap.PROT_WRITE, offset=HLS_BASE)
        ctrl = self._read(0x00)
        print(f"\033[94m[FPGA]\033[0m Ready, AP_CTRL=0x{ctrl:02X}")

    def program_fpga(self, bitfile):
        subprocess.run(["cp", bitfile, "/lib/firmware/siamese_lstm.bin"], check=True)
        with open("/sys/class/fpga_manager/fpga0/flags", "w") as f:
            f.write("0")
        with open("/sys/class/fpga_manager/fpga0/firmware", "w") as f:
            f.write("siamese_lstm.bin")
        time.sleep(0.5)
        with open("/sys/class/fpga_manager/fpga0/state") as f:
            state = f.read().strip()
        print(f"\033[94m[FPGA]\033[0m Programmed, state: {state}")

    def _read(self, off):
        self.mm.seek(off)
        return struct.unpack("<I", self.mm.read(4))[0]

    def _write(self, off, val):
        self.mm.seek(off)
        self.mm.write(struct.pack("<I", val & 0xFFFFFFFF))

    def _write_buf(self, phys, data):
        raw = data.astype(np.float32).tobytes()
        size = max(PAGE_SIZE, ((len(raw) + PAGE_SIZE - 1) // PAGE_SIZE) * PAGE_SIZE)
        fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
        mm = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=phys)
        mm.seek(0); mm.write(raw)
        mm.close(); os.close(fd)

    def _read_buf(self, phys, count):
        fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
        mm = mmap.mmap(fd, PAGE_SIZE, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=phys)
        mm.seek(0); raw = mm.read(count * 4)
        mm.close(); os.close(fd)
        return np.frombuffer(raw, dtype=np.float32).copy()

    def predict(self, seq1, seq2, node_mask=0xFF):
        """Run inference. seq1/seq2: [seq_len, 48] numpy arrays. Returns 9 scores."""
        self._write_buf(SEQ1_PHYS, seq1.flatten())
        self._write_buf(SEQ2_PHYS, seq2.flatten())

        self._write(0x10, SEQ1_PHYS); self._write(0x14, 0)
        self._write(0x1C, SEQ2_PHYS); self._write(0x20, 0)
        self._write(0x28, RESULT_PHYS); self._write(0x2C, 0)
        self._write(0x34, seq1.shape[0])
        self._write(0x3C, seq2.shape[0])
        self._write(0x44, node_mask)

        self._write(0x00, 0x01)
        while not (self._read(0x00) & 0x06):
            pass

        return self._read_buf(RESULT_PHYS, NUM_OUTPUTS)


# ================= DATA PROCESSING =================
def parse_frame(nodes_dict):
    """
    Convert MQTT nodes dict to a flat [48] feature vector.
    Returns (frame, node_mask) where frame is [48] floats and
    node_mask is a bitmask of which nodes had data.
    """
    frame = np.zeros(INPUT_DIM, dtype=np.float32)
    node_mask = 0

    for group_name, part_names in NODE_GROUP_MAP.items():
        group_data = nodes_dict.get(group_name, {})
        for part_name in part_names:
            part_data = group_data.get(part_name)
            if part_data is None:
                continue

            idx = NODE_ORDER.index(part_name)
            node_mask |= (1 << idx)

            accel = part_data.get("accel", [0, 0, 0])
            gyro = part_data.get("gyro", [0, 0, 0])
            offset = idx * 6
            frame[offset:offset+3] = accel
            frame[offset+3:offset+6] = gyro

    return frame, node_mask


# ================= STATE =================
shifu_buffer = deque(maxlen=WINDOW_SIZE)   # sliding window of shifu frames
student_buffer = deque(maxlen=WINDOW_SIZE) # sliding window of student frames
last_node_mask = 0xFF
fpga = None


# ================= MQTT CALLBACKS =================
def on_connect(client, userdata, flags, rc):
    print(f"\033[94m[Ultra96]\033[0m Connected to MQTT broker (rc={rc})")
    client.subscribe(SENSOR_TOPIC, qos=0)


def on_message(client, userdata, msg):
    global last_node_mask

    if msg.retain:
        return

    player = msg.topic.split("/")[1]
    aggregated = json.loads(msg.payload.decode())
    sequence = aggregated.get("sequence", 0)
    nodes = aggregated.get("nodes", {})

    frame, node_mask = parse_frame(nodes)

    # ─── Accumulate frames ───
    if player == "shifu":
        shifu_buffer.append(frame)
        return

    if player != "student":
        return

    student_buffer.append(frame)
    last_node_mask = node_mask

    # ─── Check if enough frames for inference ───
    if len(shifu_buffer) < MIN_FRAMES or len(student_buffer) < MIN_FRAMES:
        print(f"\033[93m[BUFFER]\033[0m Accumulating... shifu={len(shifu_buffer)}, student={len(student_buffer)}")
        return

    # ─── Run FPGA inference ───
    seq1 = np.array(list(shifu_buffer), dtype=np.float32)   # [window, 48]
    seq2 = np.array(list(student_buffer), dtype=np.float32)  # [window, 48]

    t0 = time.time()
    scores = fpga.predict(seq1, seq2, node_mask=last_node_mask)
    latency = int((time.time() - t0) * 1000)

    # ─── Build response ───
    part_scores = {}
    for i, name in enumerate(NODE_NAMES_SHORT):
        if scores[i] >= 0:
            part_scores[name] = round(float(scores[i]), 3)

    overall = round(float(scores[8]), 3)

    print(
        f"\033[92m[INFERENCE]\033[0m seq={sequence}, "
        f"overall={overall}, latency={latency}ms, "
        f"window={len(seq2)}"
    )
    for name, sc in part_scores.items():
        print(f"  {name:>12s}: {sc:.3f}")

    response = {
        "seq": sequence,
        "scores": {
            "overall": overall,
            "part": part_scores,
        }
    }
    client.publish(INFERENCE_TOPIC, json.dumps(response), qos=0)

    # ================= PER-NODE INFERENCE PUBLISHING =================
    for node_id, parts in NODE_GROUP_MAP.items():
        payload = {
            "seq": sequence
        }

        has_data = False

        for part in parts:
            if part in part_scores:
                payload[part] = part_scores[part]
                has_data = True

        if not has_data:
            continue  # skip if no data for this node

        topic = f"inference/{player}/{node_id}"
        client.publish(topic, json.dumps(payload), qos=0)


# ================= MAIN =================
if __name__ == "__main__":
    print("\033[94m[Ultra96]\033[0m Starting FPGA Inference Node...")
    fpga = FPGAInference(BIT_FILE)

    client = mqtt.Client(clean_session=True)
    client.reconnect_delay_set(min_delay=1, max_delay=5)
    client.tls_set(ca_certs=CA_CERT, tls_version=ssl.PROTOCOL_TLS_CLIENT)
    client.tls_insecure_set(True)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(BROKER, PORT)
    client.loop_forever()
