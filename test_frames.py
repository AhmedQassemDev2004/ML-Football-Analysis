from utils import read_video
import cv2

def test_frame_reading():
    print("Testing frame reading...")
    video_frames = read_video("./input_videos/08fd33_4.mp4")
    print(f"Number of frames read: {len(video_frames)}")
    
    if len(video_frames) == 0:
        print("ERROR: No frames were read!")
        return False
    
    print(f"First frame shape: {video_frames[0].shape}")
    print(f"Frame type: {type(video_frames[0])}")
    
    # Test if we can access multiple frames
    for i in range(min(5, len(video_frames))):
        print(f"Frame {i} shape: {video_frames[i].shape}")
    
    return True

if __name__ == "__main__":
    test_frame_reading()
