from ultralytics import YOLO
import supervision as sv
import pickle
import os
from utils import get_bbox_width, get_center_of_bbox, get_foot_position
import cv2
import gzip
import numpy as np
from config import *
import pandas as pd
import hashlib


STUB_DIR = "stubs"
os.makedirs(STUB_DIR, exist_ok=True)


def compute_file_hash(filepath, block_size=65536):
    """Compute MD5 hash of a file for cache validation."""
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            md5.update(block)
    return md5.hexdigest()


def get_stub_path(video_path, config):
    """Generate reproducible stub filename based on video + config."""
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    video_hash = compute_file_hash(video_path)[:8] if video_path else "nohash"
    config_hash = hashlib.md5(str(config).encode()).hexdigest()[:8]
    stub_name = f"{base_name}_{video_hash}_{config_hash}.pkl.gz"
    return os.path.join(STUB_DIR, stub_name)


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def add_position_to_tracks(self, tracks):
        for obj_type, object_tracks in tracks.items():
            for frame_num, track_frame in enumerate(object_tracks):
                for track_id, track_info in track_frame.items():
                    bbox = track_info.get('bbox')
                    if not bbox or len(bbox) != 4:
                        continue
                    if obj_type == 'ball':
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[obj_type][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self, ball_positions):
        bboxes = []
        for frame in ball_positions:
            bbox = frame.get(1, {}).get('bbox')
            if bbox is None or bbox == [0, 0, 0, 0]:
                bboxes.append([None, None, None, None])
            else:
                bboxes.append(bbox)

        df = pd.DataFrame(bboxes)
        df = df.interpolate().bfill().ffill()
        return [{1: {'bbox': row.tolist()}} for _, row in df.iterrows()]

    def detect_frames(self, frames):
        detections = []
        for i in range(0, len(frames), BATCH_SIZE):
            batch = self.model.predict(frames[i:i + BATCH_SIZE], conf=CONFIDENCE_THRESHOLD)
            detections.extend(batch)
        return detections

    def get_object_tracks(self, frames, video_path, use_stub=True):
        # Metadata for reproducibility
        config = {
            "model": MODEL_PATH,
            "confidence": CONFIDENCE_THRESHOLD,
            "batch_size": BATCH_SIZE,
        }

        stub_path = get_stub_path(video_path, config)

        # === 1. Load Stub if Exists === #
        if use_stub and os.path.exists(stub_path):
            with gzip.open(stub_path, 'rb') as f:
                saved = pickle.load(f)
            print(f"[INFO] Loaded cached tracks from {stub_path} ✅")
            return saved["tracks"]

        # === 2. Run Detection + Tracking === #
        detections = self.detect_frames(frames)
        tracks = {"players": [], "referees": [], "ball": []}

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Fix goalkeeper → player
            for idx, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[idx] = cls_names_inv["player"]

            tracked = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for det in tracked:
                bbox, cls_id, track_id = det[0].tolist(), det[3], det[4]
                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                elif cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            for det in detection_supervision:
                bbox, cls_id = det[0].tolist(), det[3]
                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        # === 3. Save Stub === #
        with gzip.open(stub_path, 'wb') as f:
            pickle.dump({"config": config, "tracks": tracks}, f)
        print(f"[INFO] Saved tracks to stub: {stub_path}")

        return tracks

    # ---------------- DRAWING ---------------- #

    def draw_player_marker(self, frame, bbox, color, track_id=None, has_ball=False):
        x_center, y_bottom = get_center_of_bbox(bbox)
        y_bottom = int(bbox[3])

        overlay = frame.copy()
        cv2.circle(overlay, (x_center, y_bottom), 20, color, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.circle(frame, (x_center, y_bottom), 20, (0, 0, 0), 2)

        if has_ball:
            cv2.circle(frame, (x_center, y_bottom), 28, (0, 0, 255), 3)

        if track_id is not None:
            cv2.putText(frame, str(track_id), (x_center - 10, y_bottom - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv2.putText(frame, str(track_id), (x_center - 10, y_bottom - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        return frame

    def draw_referee_marker(self, frame, bbox):
        x_center, y_bottom = get_center_of_bbox(bbox)
        y_bottom = int(bbox[3])
        cv2.rectangle(frame, (x_center - 5, y_bottom - 5),
                      (x_center + 5, y_bottom + 5), (0, 255, 255), -1)
        return frame

    def draw_ball_marker(self, frame, bbox):
        x, y = get_center_of_bbox(bbox)
        cv2.circle(frame, (x, y), 8, (255, 255, 255), -1)
        cv2.circle(frame, (x, y), 10, (0, 0, 0), 2)
        cv2.circle(frame, (x, y), 15, (0, 255, 255), 2)
        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control, team_colors=None):
        overlay = frame.copy()
        h, w = frame.shape[:2]

        bar_x1, bar_y1 = 100, h - 60
        bar_x2, bar_y2 = w - 100, h - 20

        team_control_frame = team_ball_control[:frame_num + 1]
        team_counts = {}
        for team_id in set(team_control_frame):
            team_counts[team_id] = team_control_frame.count(team_id)
        total = max(1, sum(team_counts.values()))

        x_start = bar_x1
        for team_id, count in team_counts.items():
            ratio = count / total
            color = team_colors.get(team_id, (0, 0, 255)) if team_colors else (0, 0, 255)
            x_end = x_start + int((bar_x2 - bar_x1) * ratio)

            cv2.rectangle(overlay, (x_start, bar_y1), (x_end, bar_y2), color, -1)
            cv2.rectangle(overlay, (x_start, bar_y1), (x_end, bar_y2), (0, 0, 0), 3)

            percentage = int(ratio * 100)
            text = f"{percentage}% ({count})"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]

            text_x = x_start + (x_end - x_start - text_size[0]) // 2
            text_y = bar_y1 - 10

            cv2.putText(frame, text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
            cv2.putText(frame, text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            x_start = x_end

        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output_video_frames = []

        if tracks['players']:
            first_frame_players = tracks['players'][0]
            team_colors = {p['team_id']: p['team_color'] for p in first_frame_players.values() if 'team_id' in p}
        else:
            team_colors = {}

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            try:
                players_dict = tracks['players'][frame_num]
                referees_dict = tracks['referees'][frame_num]
                ball_dict = tracks['ball'][frame_num]
            except (IndexError, KeyError):
                output_video_frames.append(frame)
                continue

            for track_id, player in players_dict.items():
                frame = self.draw_player_marker(
                    frame, player['bbox'], player['team_color'],
                    track_id, has_ball=player.get('has_ball', False)
                )

            for _, referee in referees_dict.items():
                frame = self.draw_referee_marker(frame, referee['bbox'])

            for _, ball in ball_dict.items():
                frame = self.draw_ball_marker(frame, ball['bbox'])

            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control, team_colors)
            output_video_frames.append(frame)

        return output_video_frames
