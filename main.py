import gradio as gr
import os
from utils import read_video, save_video
from trackers import Tracker
from config import *
from speed_distance_estimator import SpeedDistanceEstimator
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer


def process_video(video_file, show_boxes, show_ids, show_ball_control, show_speed, detail_level, conf_threshold):
    """Main video processing pipeline, wrapped for Gradio."""
    if video_file is None:
        return None, "❌ Please upload a video first.", None, None

    video_frames = read_video(video_file)
    tracker = Tracker(MODEL_PATH)

    # Step 1: Tracking
    tracks = tracker.get_object_tracks(video_frames, video_file)
    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])
    tracker.add_position_to_tracks(tracks)

    # Step 2: Camera correction
    camera_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_estimator.get_camera_movement(video_frames)
    camera_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # Step 3: Team assignment
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
    for frame_num, players in enumerate(tracks['players']):
        for pid, player in players.items():
            team_id = team_assigner.get_player_team(video_frames[frame_num], player['bbox'], pid)
            player['team_id'] = team_id
            player['team_color'] = team_assigner.team_colors[team_id]

    # Step 4: Ball assignment
    assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, players in enumerate(tracks['players']):
        ball_dict = tracks['ball'][frame_num]
        if 1 not in ball_dict:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else -1)
            continue
        ball_bbox = ball_dict[1]['bbox']
        assigned_player = assigner.assign_ball_to_player(players, ball_bbox)
        if assigned_player != -1:
            players[assigned_player]['has_ball'] = True
            team_ball_control.append(players[assigned_player]['team_id'])
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else -1)

    # Step 5: Speed & Distance
    speed_distance_estimator = SpeedDistanceEstimator()
    speed_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Step 6: Draw video
    output_frames = camera_estimator.draw_camera_movement(video_frames, camera_movement_per_frame)
    output_frames = tracker.draw_annotations(output_frames, tracks, team_ball_control)
    output_frames = speed_distance_estimator.draw_speed_and_distance(output_frames, tracks)

    output_path = "processed_output.mp4"
    save_video(output_frames, output_path)

    # === Build Summary === #
    team1_possession = int(100 * team_ball_control.count(0) / len(team_ball_control)) if team_ball_control else 0
    team2_possession = 100 - team1_possession
    possession_summary = f"Team 1: {team1_possession}% | Team 2: {team2_possession}%"

    # Per-player stats
    player_stats = []
    for frame_players in tracks['players']:
        for pid, pdata in frame_players.items():
            if "distance" in pdata and "speed" in pdata:
                player_stats.append({
                    "Player ID": pid,
                    "Team": pdata.get("team_id", "-"),
                    "Distance (m)": round(pdata.get("distance", 0), 2),
                    "Avg Speed (km/h)": round(pdata.get("speed", 0), 2),
                })

    return output_path, "✅ Processing complete!", possession_summary, player_stats


# Gradio UI
with gr.Blocks(title="⚽ Football Analysis Dashboard") as demo:
    gr.Markdown(
        "## ⚽ Football Video Analytics\nUpload a football video and get detections, tracking, possession stats, and player performance.")

    with gr.Row():
        with gr.Column(scale=1):
            input_video = gr.File(label="📂 Upload Video", file_types=[".mp4", ".avi"], type="filepath")
            show_boxes = gr.Checkbox(label="Show Tracking Boxes", value=True)
            show_ids = gr.Checkbox(label="Show Player IDs", value=True)
            show_ball_control = gr.Checkbox(label="Show Ball Control Bar", value=True)
            show_speed = gr.Checkbox(label="Show Speed & Distance", value=True)
            detail_level = gr.Dropdown(["Fast", "Balanced", "Full"], label="Detail Level", value="Balanced")
            conf_threshold = gr.Slider(0.1, 1.0, 0.5, step=0.05, label="Confidence Threshold")
            run_btn = gr.Button("🚀 Run Analysis", variant="primary")

        with gr.Column(scale=2):
            output_video = gr.Video(label="🎥 Processed Video")
            status = gr.Textbox(label="ℹ️ Status", interactive=False)
            possession_summary = gr.Textbox(label="📊 Ball Possession", interactive=False)
            player_stats = gr.Dataframe(headers=["Player ID", "Team", "Distance (m)", "Avg Speed (km/h)"],
                                        label="🏃 Player Stats")

    run_btn.click(
        fn=process_video,
        inputs=[input_video, show_boxes, show_ids, show_ball_control, show_speed, detail_level, conf_threshold],
        outputs=[output_video, status, possession_summary, player_stats]
    )

if __name__ == "__main__":
    demo.launch()
