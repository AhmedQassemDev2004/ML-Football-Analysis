import cv2
import player_ball_assigner
from team_assigner import TeamAssigner
from utils import read_video, save_video
from trackers import Tracker
from config import *
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer


def main():
    video_frames = read_video(INPUT_VIDEO_PATH)
    tracker = Tracker(MODEL_PATH)
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path=STUB_PATH)

    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])
    
    # Add position data to tracks before camera movement estimation
    tracker.add_position_to_tracks(tracks)
    
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames, 
                                                                            read_from_stub=True, 
                                                                            stub_path='stubs/camera_movement_stub.pkl')
    
    # Adjust object positions for camera movement
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    print(f"Number of player track frames: {len(tracks['players'])}")
    print(f"Number of ball track frames: {len(tracks['ball'])}")
    print(f"Number of referee track frames: {len(tracks['referees'])}")


    # View Transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)


    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    for frame_num, player_tracks in enumerate(tracks['players']):
        for player_id, player_track in player_tracks.items():
            bbox = player_track['bbox']
            team_id = team_assigner.get_player_team(video_frames[frame_num], bbox, player_id) 
            tracks['players'][frame_num][player_id]['team_id'] = team_id
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team_id]
    
    assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team_id'])
        else:
            team_ball_control.append(team_ball_control[-1])

    # Draw camera movement visualization and annotations
    output_video_frames = camera_movement_estimator.draw_camera_movement(video_frames, camera_movement_per_frame)
    output_video_frames = tracker.draw_annotations(output_video_frames, tracks, team_ball_control)
    save_video(output_video_frames, OUTPUT_VIDEO_PATH)

if __name__ == '__main__':
    main()