import cv2
from team_assigner import TeamAssigner
from utils import read_video, save_video
from trackers import Tracker
from config import *


def main():
    video_frames = read_video(INPUT_VIDEO_PATH)
    tracker = Tracker(MODEL_PATH)
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path=STUB_PATH)

    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])

    print(f"Number of player track frames: {len(tracks['players'])}")
    print(f"Number of ball track frames: {len(tracks['ball'])}")
    print(f"Number of referee track frames: {len(tracks['referees'])}")
    
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    for frame_num, player_tracks in enumerate(tracks['players']):
        for player_id, player_track in player_tracks.items():
            bbox = player_track['bbox']
            team_id = team_assigner.get_player_team(video_frames[frame_num], bbox, player_id) 
            tracks['players'][frame_num][player_id]['team_id'] = team_id
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team_id]
            
    
    output_video_frames = tracker.draw_annotations(video_frames, tracks)
    save_video(output_video_frames, OUTPUT_VIDEO_PATH)

if __name__ == '__main__':
    main()