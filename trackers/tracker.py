from ultralytics import YOLO
import supervision as sv
import pickle
import os 

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detection_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections.extend(detection_batch)

        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        detections = self.detect_frames(frames)

        tracks = {
            "players":[],
            "referees":[],
            "ball":[],
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names # class names {index: name} 
            cls_names_inv = {v:k for k, v in cls_names.items()} # {name: index}

            detection_supervision = sv.Detections.from_ultralytics(detection)

            # convert goalkeeper to player class to avoid missclassification
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == 'goalkeeper':
                    detection_supervision.class_id[object_ind] = cls_names_inv['player']

            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks['players'].append({})
            tracks['referees'].append({})
            tracks['ball'].append({})
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]
                track_id = frame_detection[4]

                if class_id == cls_names_inv['player']:
                    tracks['players'][frame_num][track_id] = bbox
                elif class_id == cls_names_inv['referee']:
                    tracks['referees'][frame_num][track_id] = bbox

            # we don't want to track the ball inside the last loop we'll 
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]

                if class_id == cls_names_inv['ball']:
                    tracks['ball'][frame_num][1] = bbox
            
            if stub_path is not None:
                with open(stub_path, 'wb') as f:
                    pickle.dump(tracks, f)

            return tracks