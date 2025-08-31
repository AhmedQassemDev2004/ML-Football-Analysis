import cv2
from utils import measure_distance, get_foot_position

class SpeedDistanceEstimator():
    def __init__(self):
        self.frame_window = 5
        self.frame_rate = 24

    def add_speed_and_distance_to_tracks(self, tracks):
        total_distance = {}

        for object, object_tracks in tracks.items():
            if object != 'players':
                continue

            number_of_frames = len(object_tracks)
            for frame_num in range(0, number_of_frames, self.frame_window):
                last_frame = min(frame_num + self.frame_window, number_of_frames - 1)

                for track_id, _ in object_tracks[frame_num].items():
                    if track_id not in object_tracks[last_frame]:
                        continue

                    start_position = object_tracks[frame_num][track_id]['position_transformed']
                    end_position = object_tracks[last_frame][track_id]['position_transformed']

                    if start_position is None or end_position is None:
                        continue

                    distance_covered = measure_distance(start_position, end_position)
                    time_elapsed = (last_frame - frame_num) / self.frame_rate
                    if time_elapsed == 0: time_elapsed=0.0001
                    speed_meteres_per_second = distance_covered / time_elapsed
                    speed_km_per_hour = speed_meteres_per_second * 3.6

                    if object not in total_distance:
                        total_distance[object] = {}

                    if track_id not in total_distance[object]:
                        total_distance[object][track_id] = 0

                    total_distance[object][track_id] += distance_covered

                    for frame_num_batch in range(frame_num, last_frame):
                        if track_id not in tracks[object][frame_num_batch]:
                            continue
                        tracks[object][frame_num_batch][track_id]['speed'] = speed_km_per_hour
                        tracks[object][frame_num_batch][track_id]['distance'] = total_distance[object][track_id]

    def draw_speed_and_distance(self, frames, tracks):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            frame = frame.copy()
            for object, object_tracks in tracks.items():
                if object == "ball" or object == "referees":
                    continue

                for _, track_info in object_tracks[frame_num].items():
                    if "speed" not in track_info or "distance" not in track_info:
                        continue

                    speed = track_info['speed']
                    distance = track_info['distance']
                    bbox = track_info['bbox']

                    # Position box above player head
                    x, y = get_foot_position(bbox)
                    y = int(bbox[1]) - 30  # above top of bbox
                    x = int((bbox[0] + bbox[2]) / 2)

                    # Text lines
                    text1 = f"{speed:.1f} km/h"
                    text2 = f"{distance:.1f} m"

                    # Background box (semi-transparent)
                    (w1, h1), _ = cv2.getTextSize(text1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    (w2, h2), _ = cv2.getTextSize(text2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    box_w = max(w1, w2) + 12
                    box_h = h1 + h2 + 14

                    overlay = frame.copy()
                    cv2.rectangle(overlay, (x - box_w // 2, y - box_h),
                                  (x + box_w // 2, y), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

                    # Draw text (white with black shadow for readability)
                    cv2.putText(frame, text1, (x - w1 // 2, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(frame, text2, (x - w2 // 2, y + h2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            output_frames.append(frame)
        return output_frames
