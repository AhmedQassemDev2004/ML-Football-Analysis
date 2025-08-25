import torch
from utils import read_video, save_video
from ultralytics import YOLO
from trackers import Tracker


def main():
    video_frams = read_video("./input_videos/08fd33_4.mp4")
    tracker = Tracker("./models/best.pt")
    tracks = tracker.get_object_tracks(video_frams, read_from_stub=True, stub_path="stubs/track_stubs.pkl")
    print(tracks)
    save_video(video_frams, "./output_videos/output_video.mp4")

if __name__ == '__main__':
    main()