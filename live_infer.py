"""
Live inference for the open-set AQA Siamese Network via MQTT.

Connects to an MQTT broker and receives real-time IMU data from shifu
(expert) and student streams. Each MQTT message contains one timestep
of 8-node IMU data (4 limbs x 2 segments x 6 axes = 48 channels).

Data flow:
    1. MQTT messages arrive on sensor/shifu/aggregated and
       sensor/student/aggregated with JSON payloads containing
       accel + gyro readings per node.
    2. parse_frame() flattens the JSON into a (48,) feature vector
       matching the training CSV column order.
    3. Frames are pushed into SlidingWindowBuffers (one per stream).
    4. Every step_size student frames (after MIN_FRAMES accumulated),
       run_inference() is called:
       - Extracts (128, 48) windows from both buffers
       - Applies the same normalisation as training
       - Splits into 4 limb tensors per stream
       - Routes arms to arm_extractor, legs to leg_extractor
       - Computes per-limb scores via exp(-dist^2 / tau_group^2)
       - Aggregates via compute_overall_score() (geometric mean +
         worst-limb penalty)
    5. Results are published to inference/result (overall + per-node
       scores) and inference/student/<limb> (per-limb breakdown).

Optionally saves all received frames to CSV files (--save_streams)
for offline replay and debugging.

Usage:
    python live_infer.py
    python live_infer.py --step_size 20
    python live_infer.py --save_streams
"""

import argparse
import csv
import json
import os
import ssl
import sys
import threading
import time
from collections import deque
from datetime import datetime

import numpy as np
import paho.mqtt.client as mqtt
import torch

from aqa_model import AQAModel, LIMB_NAMES
from infer import (
    SlidingWindowBuffer,
    load_model,
    print_scores,
    run_inference,
)

# ================= MQTT CONFIG =================
BROKER = "54.79.62.237"
PORT = 8883
CA_CERT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mqtt-certs", "ca.crt")
# CA_CERT = "/home/xilinx/mqtt-certs/ca.crt"
SENSOR_TOPIC = "sensor/+/aggregated"
INFERENCE_TOPIC = "inference/result"

# ================= MODEL CONFIG =================
MIN_FRAMES = 20  # minimum frames in each buffer before first inference

# Node order matching the 48-channel layout (8 IMU nodes x 6 features each).
# This order must match the CSV column layout used during training:
#   4 limbs x 2 segments x (ax, ay, az, gx, gy, gz) = 48 channels.
NODE_ORDER = [
    "left_arm_upper_arm", "left_arm_forearm",
    "right_arm_upper_arm", "right_arm_forearm",
    "left_leg_thigh", "left_leg_shin",
    "right_leg_thigh", "right_leg_shin",
]

# Maps MQTT node groups to body part names
NODE_GROUP_MAP = {
    "left_arm":  ["left_arm_upper_arm", "left_arm_forearm"],
    "right_arm": ["right_arm_upper_arm", "right_arm_forearm"],
    "left_leg":  ["left_leg_thigh", "left_leg_shin"],
    "right_leg": ["right_leg_thigh", "right_leg_shin"],
}

# Column headers for CSV recording
_CSV_COLUMNS = [
    "timestamp_ms",
    "left_arm_upper_arm_ax", "left_arm_upper_arm_ay", "left_arm_upper_arm_az",
    "left_arm_upper_arm_gx", "left_arm_upper_arm_gy", "left_arm_upper_arm_gz",
    "left_arm_forearm_ax", "left_arm_forearm_ay", "left_arm_forearm_az",
    "left_arm_forearm_gx", "left_arm_forearm_gy", "left_arm_forearm_gz",
    "right_arm_upper_arm_ax", "right_arm_upper_arm_ay", "right_arm_upper_arm_az",
    "right_arm_upper_arm_gx", "right_arm_upper_arm_gy", "right_arm_upper_arm_gz",
    "right_arm_forearm_ax", "right_arm_forearm_ay", "right_arm_forearm_az",
    "right_arm_forearm_gx", "right_arm_forearm_gy", "right_arm_forearm_gz",
    "left_leg_thigh_ax", "left_leg_thigh_ay", "left_leg_thigh_az",
    "left_leg_thigh_gx", "left_leg_thigh_gy", "left_leg_thigh_gz",
    "left_leg_shin_ax", "left_leg_shin_ay", "left_leg_shin_az",
    "left_leg_shin_gx", "left_leg_shin_gy", "left_leg_shin_gz",
    "right_leg_thigh_ax", "right_leg_thigh_ay", "right_leg_thigh_az",
    "right_leg_thigh_gx", "right_leg_thigh_gy", "right_leg_thigh_gz",
    "right_leg_shin_ax", "right_leg_shin_ay", "right_leg_shin_az",
    "right_leg_shin_gx", "right_leg_shin_gy", "right_leg_shin_gz",
]


# ================= DATA PROCESSING =================

def parse_frame(nodes_dict: dict) -> np.ndarray:
    """
    Convert MQTT nodes dict to a flat (48,) feature vector.

    Expected format:
        {"left_arm": {"left_arm_upper_arm": {"accel": [x,y,z], "gyro": [x,y,z]},
                      "left_arm_forearm":   {"accel": [x,y,z], "gyro": [x,y,z]}},
         "right_arm": {...}, "left_leg": {...}, "right_leg": {...}}

    Returns:
        np.ndarray of shape (48,) — 8 nodes x 6 features (ax,ay,az,gx,gy,gz).
    """
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


# ================= STREAM RECORDER =================

class StreamRecorder:
    """Records incoming IMU frames to CSV files for later replay/analysis."""

    def __init__(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self._shifu_path = os.path.join(save_dir, f"shifu_{timestamp}.csv")
        self._student_path = os.path.join(save_dir, f"student_{timestamp}.csv")

        self._shifu_file = open(self._shifu_path, "w", newline="")
        self._student_file = open(self._student_path, "w", newline="")

        self._shifu_writer = csv.writer(self._shifu_file)
        self._student_writer = csv.writer(self._student_file)

        self._shifu_writer.writerow(_CSV_COLUMNS)
        self._student_writer.writerow(_CSV_COLUMNS)

        self._start_ms = int(time.time() * 1000)

        print(f"Recording streams to:")
        print(f"  Shifu:   {self._shifu_path}")
        print(f"  Student: {self._student_path}")

    def record_shifu(self, frame: np.ndarray) -> None:
        ts = int(time.time() * 1000) - self._start_ms
        self._shifu_writer.writerow([ts] + frame.tolist())

    def record_student(self, frame: np.ndarray) -> None:
        ts = int(time.time() * 1000) - self._start_ms
        self._student_writer.writerow([ts] + frame.tolist())

    def close(self) -> None:
        self._shifu_file.close()
        self._student_file.close()
        print(f"\nRecordings saved:")
        print(f"  Shifu:   {self._shifu_path}")
        print(f"  Student: {self._student_path}")


# ================= MAIN =================

def main():
    p = argparse.ArgumentParser(description="Live AQA inference via MQTT")
    p.add_argument("--checkpoint", type=str,
                   default=os.path.join(os.path.dirname(__file__), "checkpoints", "best_original_model.pt"),
                   help="Path to saved model checkpoint")
    p.add_argument("--step_size", type=int, default=10,
                   help="Student frames between inference steps")
    p.add_argument("--device", type=str, default="auto",
                   help="'cpu', 'cuda', or 'auto'")
    p.add_argument("--save_streams", action="store_true", default=False,
                   help="Save received IMU frames to CSV files")
    p.add_argument("--save_dir", type=str,
                   default=os.path.join(os.path.dirname(__file__), "live_recordings"),
                   help="Directory to save stream CSVs (used with --save_streams)")
    p.add_argument("--broker", type=str, default=BROKER,
                   help="MQTT broker address")
    p.add_argument("--port", type=int, default=PORT,
                   help="MQTT broker port")
    p.add_argument("--ca_cert", type=str, default=CA_CERT,
                   help="Path to CA certificate for TLS")
    p.add_argument("--model_type", type=str, default="auto",
                   choices=["auto", "slim", "original"],
                   help="Architecture variant to load. 'auto' detects from "
                        "checkpoint keys (default). 'slim' = dilated-conv "
                        "temporal block; 'original' = Bi-LSTM.")
    args = p.parse_args()

    if not os.path.isfile(args.checkpoint):
        sys.exit(f"Error: checkpoint not found: {args.checkpoint}")

    # -- Device --
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # -- Load model --
    model, checkpoint, global_mean, global_std = load_model(
        args.checkpoint, device, args.model_type)
    val_metric = checkpoint.get("val_loss", checkpoint.get("val_mae", None))
    metric_name = "val_loss" if "val_loss" in checkpoint else "val_mae"
    print(f"Model loaded (trained epoch {checkpoint['epoch']}, "
          f"{metric_name} {val_metric:.4f})")
    if global_mean is not None:
        print("Normalisation stats loaded from checkpoint")
    else:
        print("WARNING: No normalisation stats in checkpoint (old model?)")

    # -- Optional stream recording --
    recorder = StreamRecorder(args.save_dir) if args.save_streams else None

    # -- Buffers and state --
    shifu_buf = SlidingWindowBuffer()
    student_buf = SlidingWindowBuffer()
    frames_since_last = [0]  # mutable for closure access
    last_sequence = [0]
    last_shifu_ts = [None]
    last_student_ts = [None]

    # -- Inference worker (single-slot replace-on-arrival mailbox) --
    # Inference runs on its own thread so the MQTT network loop never blocks.
    # Only the most recent pending job is kept; older ones are silently
    # dropped so lag can't accumulate under load.
    job_cond = threading.Condition()
    latest_job = [None]
    stop_flag = [False]
    dropped_count = [0]

    class _FrozenBuffer:
        __slots__ = ("_window",)
        def __init__(self, window):
            self._window = window
        def get_window(self):
            return self._window

    # -- MQTT callbacks --
    def on_connect(client, userdata, flags, rc):
        print(f"Connected to MQTT broker (rc={rc})")
        client.subscribe(SENSOR_TOPIC, qos=0)
        print(f"Subscribed to {SENSOR_TOPIC}")
        print(f"\nLive mode started. Waiting for sensor data...\n")

    def on_message(client, userdata, msg):
        if msg.retain:
            return

        player = msg.topic.split("/")[1]
        payload = json.loads(msg.payload.decode())
        sequence = payload.get("sequence", 0)
        start_ts = payload.get("start_ts")
        nodes = payload.get("nodes", {})

        frame = parse_frame(nodes)

        # -- Accumulate shifu frames --
        if player == "shifu":
            shifu_buf.push(frame)
            last_shifu_ts[0] = start_ts
            if recorder is not None:
                recorder.record_shifu(frame)
            return

        if player != "student":
            return

        # -- Accumulate student frames --
        student_buf.push(frame)
        last_student_ts[0] = start_ts
        if recorder is not None:
            recorder.record_student(frame)
        last_sequence[0] = sequence
        frames_since_last[0] += 1

        # -- Check if enough frames for inference --
        if shifu_buf.count < MIN_FRAMES or student_buf.count < MIN_FRAMES:
            print(f"Accumulating... shifu={shifu_buf.count}, "
                  f"student={student_buf.count}")
            return

        if frames_since_last[0] < args.step_size:
            return

        frames_since_last[0] = 0

        # -- Snapshot state and hand off to worker thread --
        ts_candidates = [t for t in (last_shifu_ts[0], last_student_ts[0]) if t is not None]
        earliest_start_ts = min(ts_candidates) if ts_candidates else None

        new_job = {
            "shifu_window":   shifu_buf.get_window(),
            "student_window": student_buf.get_window(),
            "sequence":       sequence,
            "start_ts":       earliest_start_ts,
            "filled":         min(shifu_buf.count, student_buf.count),
            "student_count":  student_buf.count,
        }
        with job_cond:
            if latest_job[0] is not None:
                dropped_count[0] += 1
            latest_job[0] = new_job
            job_cond.notify()

    # -- Inference worker thread body --
    limb_to_nodes = {
        "left_arm":  ["left_arm_upper_arm", "left_arm_forearm"],
        "right_arm": ["right_arm_upper_arm", "right_arm_forearm"],
        "left_leg":  ["left_leg_thigh", "left_leg_shin"],
        "right_leg": ["right_leg_thigh", "right_leg_shin"],
    }

    def _run_and_publish(job, client):
        t0 = time.time()
        scores = run_inference(
            model,
            _FrozenBuffer(job["shifu_window"]),
            _FrozenBuffer(job["student_window"]),
            device, global_mean, global_std,
        )
        latency = int((time.time() - t0) * 1000)

        print_scores(job["student_count"], job["filled"], scores)
        missing_limbs = [l for l in LIMB_NAMES if scores[l] < 0]
        if missing_limbs:
            print(f"  Missing limbs: {', '.join(missing_limbs)}")
        dropped_note = f", dropped={dropped_count[0]}" if dropped_count[0] else ""
        print(f"  seq={job['sequence']}, start_ts={job['start_ts']}, "
              f"latency={latency}ms{dropped_note}")

        # -- Publish overall result --
        # Map 4 limb scores to 8-node naming convention (each limb's
        # 2 nodes share the same score since the model operates per-limb)
        part_scores = {}
        for limb in LIMB_NAMES:
            limb_score = round(float(scores[limb]), 3)
            for node_name in limb_to_nodes[limb]:
                part_scores[node_name] = limb_score

        response = {
            "seq": job["sequence"],
            "start_ts": job["start_ts"],
            "scores": {
                "overall": round(float(scores["overall"]), 3),
                "part": part_scores,
            },
        }
        client.publish(INFERENCE_TOPIC, json.dumps(response), qos=0)

        # -- Publish per-limb results --
        for limb in LIMB_NAMES:
            payload = {"seq": job["sequence"], "start_ts": job["start_ts"]}
            for node_name in limb_to_nodes[limb]:
                payload[node_name] = round(float(scores[limb]), 3)
            topic = f"inference/student/{limb}"
            client.publish(topic, json.dumps(payload), qos=0)

    def worker(client):
        while not stop_flag[0]:
            with job_cond:
                while latest_job[0] is None and not stop_flag[0]:
                    job_cond.wait()
                if stop_flag[0]:
                    return
                job = latest_job[0]
                latest_job[0] = None
            try:
                _run_and_publish(job, client)
            except Exception as e:
                print(f"Inference worker error: {e}")

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

    worker_thread = threading.Thread(target=worker, args=(client,),
                                     name="inference-worker", daemon=True)
    worker_thread.start()

    print(f"Connecting to MQTT broker at {args.broker}:{args.port}...")
    try:
        client.connect(args.broker, args.port)
        client.loop_forever()
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        stop_flag[0] = True
        with job_cond:
            job_cond.notify_all()
        worker_thread.join(timeout=2.0)
        client.disconnect()
        if recorder is not None:
            recorder.close()


if __name__ == "__main__":
    main()
