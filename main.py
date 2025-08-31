import cv2
from speed_distance_estimator import SpeedDistanceEstimator
from utils import read_video, save_video
from trackers import Tracker
from config import *
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer


def run_tracking(video_frames):
    """Run object detection + tracking for players, referees, and ball."""
    print("[INFO] Starting object tracking...")
    tracker = Tracker(MODEL_PATH)
    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=True,
        stub_path=STUB_PATH
    )
    print(f"[INFO] Tracking completed. Frames tracked: Players={len(tracks['players'])}, "
          f"Ball={len(tracks['ball'])}, Referees={len(tracks['referees'])}")

    # Interpolate ball & add positions
    print("[INFO] Interpolating ball positions and adding positions to tracks...")
    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])
    tracker.add_position_to_tracks(tracks)
    print("[INFO] Ball positions and track positions updated.")
    return tracker, tracks


def run_camera_correction(video_frames, tracks):
    """Estimate and adjust positions for camera movement."""
    print("[INFO] Estimating camera movement...")
    camera_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_estimator.get_camera_movement(
        video_frames,
        read_from_stub=True,
        stub_path=CAMERA_MOVEMENT_STUB
    )
    print("[INFO] Adjusting positions for camera movement...")
    camera_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)
    print("[INFO] Camera correction applied.")
    return camera_estimator, camera_movement_per_frame


def run_team_assignment(video_frames, tracks):
    """Assign players to teams using jersey color clustering."""
    print("[INFO] Assigning players to teams...")
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    for frame_num, player_tracks in enumerate(tracks['players']):
        for player_id, player_track in player_tracks.items():
            bbox = player_track['bbox']
            team_id = team_assigner.get_player_team(video_frames[frame_num], bbox, player_id)
            player_track['team_id'] = team_id
            player_track['team_color'] = team_assigner.team_colors[team_id]

    print("[INFO] Team assignment completed.")
    return team_assigner


def run_ball_assignment(tracks):
    """Assign ball possession to players and track team control."""
    print("[INFO] Assigning ball possession...")
    assigner = PlayerBallAssigner()
    team_ball_control = []

    for frame_num, players in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = assigner.assign_ball_to_player(players, ball_bbox)

        if assigned_player != -1:
            players[assigned_player]['has_ball'] = True
            team_ball_control.append(players[assigned_player]['team_id'])
        elif team_ball_control:  # carry last possession
            team_ball_control.append(team_ball_control[-1])

    print("[INFO] Ball possession assignment completed.")
    return team_ball_control


def main():
    # === Step 1: Load video === #
    print(f"[INFO] Loading video from {INPUT_VIDEO_PATH} ...")
    video_frames = read_video(INPUT_VIDEO_PATH)
    print(f"[INFO] Video loaded. Total frames: {len(video_frames)}")

    # === Step 2: Tracking === #
    tracker, tracks = run_tracking(video_frames)

    # === Step 3: Camera Movement Correction === #
    camera_estimator, camera_movement_per_frame = run_camera_correction(video_frames, tracks)

    print(f"[INFO] Players frames: {len(tracks['players'])}")
    print(f"[INFO] Ball frames: {len(tracks['ball'])}")
    print(f"[INFO] Referees frames: {len(tracks['referees'])}")

    # === Step 4: Transform view to top-down (optional) === #
    print("[INFO] Applying view transformation (top-down)...")
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)
    print("[INFO] View transformation applied.")

    # === Step 5: Speed & Distance === #
    print("[INFO] Calculating player speed and distance...")
    speed_distance_estimator = SpeedDistanceEstimator()
    speed_distance_estimator.add_speed_and_distance_to_tracks(tracks)
    print("[INFO] Speed and distance calculation completed.")

    # === Step 6: Assign Teams === #
    run_team_assignment(video_frames, tracks)

    # === Step 7: Assign Ball Possession === #
    team_ball_control = run_ball_assignment(tracks)

    # === Step 8: Draw Outputs === #
    print("[INFO] Drawing annotations and final video frames...")
    output_video_frames = camera_estimator.draw_camera_movement(video_frames, camera_movement_per_frame)
    output_video_frames = tracker.draw_annotations(output_video_frames, tracks, team_ball_control)
    output_video_frames = speed_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    print(f"[INFO] Saving output video to {OUTPUT_VIDEO_PATH} ...")
    save_video(output_video_frames, OUTPUT_VIDEO_PATH)
    print("[INFO] Processing completed successfully.")


if __name__ == '__main__':
    main()
