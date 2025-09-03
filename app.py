import os
import gradio as gr
from utils import read_video, save_video
from trackers import Tracker
from config import *
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_distance_estimator import SpeedDistanceEstimator


def run_tracking(video_frames):
    """Run object detection + tracking for players, referees, and ball."""
    tracker = Tracker(MODEL_PATH)
    tracks = tracker.get_object_tracks(
        video_frames,
        video_path=INPUT_VIDEO_PATH,
    )
    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])
    tracker.add_position_to_tracks(tracks)
    return tracker, tracks


def run_camera_correction(video_frames, tracks):
    """Estimate and adjust positions for camera movement."""
    camera_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_estimator.get_camera_movement(
        video_frames,
        read_from_stub=False,
        stub_path=CAMERA_MOVEMENT_STUB
    )
    camera_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)
    return camera_estimator, camera_movement_per_frame


def run_team_assignment(video_frames, tracks):
    """Assign players to teams using jersey color clustering."""
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    for frame_num, player_tracks in enumerate(tracks['players']):
        for player_id, player_track in player_tracks.items():
            bbox = player_track['bbox']
            team_id = team_assigner.get_player_team(video_frames[frame_num], bbox, player_id)
            player_track['team_id'] = team_id
            player_track['team_color'] = team_assigner.team_colors[team_id]

    return team_assigner


def run_ball_assignment(tracks):
    """Assign ball possession to players and track team control."""
    assigner = PlayerBallAssigner()
    team_ball_control = []

    for frame_num, players in enumerate(tracks['players']):
        ball_dict = tracks['ball'][frame_num]

        if 1 not in ball_dict:  # ball not detected this frame
            if team_ball_control:  # carry last possession
                team_ball_control.append(team_ball_control[-1])
            continue

        ball_bbox = ball_dict[1]['bbox']
        assigned_player = assigner.assign_ball_to_player(players, ball_bbox)

        if assigned_player != -1:
            players[assigned_player]['has_ball'] = True
            team_ball_control.append(players[assigned_player]['team_id'])
        elif team_ball_control:  # carry last possession
            team_ball_control.append(team_ball_control[-1])

    return team_ball_control


def process_video(input_video):
    """Main pipeline to process uploaded video."""
    input_path = input_video  # already a filepath
    output_path = "./output_videos/output_temp.avi"

    os.makedirs("./output_videos", exist_ok=True)

    # === Step 1: Load video === #
    video_frames = read_video(input_path)

    # === Step 2: Tracking === #
    tracker, tracks = run_tracking(video_frames)

    # === Step 3: Camera Movement Correction === #
    camera_estimator, camera_movement_per_frame = run_camera_correction(video_frames, tracks)

    # === Step 4: View Transformation === #
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # === Step 5: Speed & Distance === #
    speed_distance_estimator = SpeedDistanceEstimator()
    speed_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # === Step 6: Team Assignment === #
    run_team_assignment(video_frames, tracks)

    # === Step 7: Ball Assignment === #
    team_ball_control = run_ball_assignment(tracks)

    # === Step 8: Draw Outputs === #
    output_video_frames = camera_estimator.draw_camera_movement(video_frames, camera_movement_per_frame)
    output_video_frames = tracker.draw_annotations(output_video_frames, tracks, team_ball_control)
    output_video_frames = speed_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    save_video(output_video_frames, output_path)

    return output_path


# ===========================
# 🚀 Gradio UI
# ===========================
with gr.Blocks() as demo:
    gr.Markdown("## ⚽ Football Analysis App")
    gr.Markdown("Upload a match video, and the system will analyze players, referees, the ball, and teams.")

    with gr.Row():
        input_video = gr.File(label="Upload Video", file_types=[".mp4", ".avi"], type="filepath")
        output_video = gr.Video(label="Processed Video")

    run_btn = gr.Button("Run Analysis")

    run_btn.click(
        fn=process_video,
        inputs=input_video,
        outputs=output_video
    )


if __name__ == "__main__":
    demo.launch()
