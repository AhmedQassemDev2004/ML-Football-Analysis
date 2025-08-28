from ultralytics import YOLO
import supervision as sv
import pickle
import os
from utils import get_bbox_width, get_center_of_bbox 
import cv2
import numpy as np
from config import *
import pandas as pd

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        
        df_ball_positions = pd.DataFrame(ball_positions)
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()  # for special case when the first detection is missing
        
        # Fill any remaining NaN values with empty lists to avoid conversion errors
        df_ball_positions = df_ball_positions.fillna(0)

        ball_positions = [{1: {'bbox':x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions
        


    def detect_frames(self, frames):
        detections = []
        for i in range(0, len(frames), BATCH_SIZE):
            detection_batch = self.model.predict(frames[i:i+BATCH_SIZE], conf=CONFIDENCE_THRESHOLD)
            detections.extend(detection_batch)

        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks={
            "players":[],
            "referees":[],
            "ball":[]
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            # Covert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert GoalKeeper to player object
            for object_ind , class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}
                
                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}
            
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)

        return tracks
       
    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            try:
                players_dict = tracks['players'][frame_num]
                referees_dict = tracks['referees'][frame_num]
                ball_dict = tracks['ball'][frame_num]
            except (IndexError, KeyError) as e:
                print(f"Warning: Missing data for frame {frame_num}: {e}")
                continue

            # draw players
            for track_id, player in players_dict.items():
                frame = self.draw_ellipse(frame, player['bbox'], player['team_color'], track_id)
                if player.get('has_ball', False):
                    frame = self.draw_traingle(frame, player['bbox'], (0, 0, 255))

            # draw referees
            for _, referee in referees_dict.items():
                frame = self.draw_ellipse(frame, referee['bbox'], REFEREE_COLOR)
            
            # draw ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball['bbox'], BALL_COLOR)
            
            output_video_frames.append(frame)

        return output_video_frames

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        bbox_width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2), # to make the ellipse at bottom center not the box center
            axes=(int(bbox_width), int(bbox_width*0.35)),
            color=color,
            angle=0,
            startAngle=-45,
            endAngle=225, # startAngle and endAngle determines which parts of the ellipse will be drawn
            thickness=ELLIPSE_THICKNESS,
            lineType=cv2.LINE_4,
        )

        
        # put the track id under each player
        if track_id is not None:
            x1_rect = x_center - RECTANGLE_WIDTH / 2
            x2_rect = x_center + RECTANGLE_WIDTH / 2
            y1_rect = (y2-RECTANGLE_HEIGHT//2)+15
            y2_rect = (y2+RECTANGLE_HEIGHT//2)+15
            
            # Draw a white rectangle
            cv2.rectangle(frame, (int(x1_rect), int(y1_rect)), (int(x2_rect), int(y2_rect)), BACKGROUND_COLOR, -1)

            # Add black text on top of the white rectangle
            cv2.putText(frame, str(track_id), (int(x1_rect+12), int(y1_rect + 15)), 
                        cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)

        return frame

    def draw_traingle(self,frame,bbox,color):
    
        y= int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

        return frame