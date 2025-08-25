import cv2 

def read_video(path):
    cap = cv2.VideoCapture(path)
    frames = []

    while True:
        ret, frame = cap.read() # ret is a flag if there's a frame or not
        if not ret: 
            break

        frames.append(frame)

    return frames

def save_video(output_frames, output_path, fps=30):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 24, (output_frames[0].shape[1], output_frames[0].shape[0]))

    for frame in output_frames:
        out.write(frame)
    
    out.release()
