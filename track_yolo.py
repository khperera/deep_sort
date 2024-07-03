import os
import numpy as np
import cv2
from deep_sort import  nn_matching#preprocessing,
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

# Initialize Deep SORT
model_filename = 'model_data/mars-small128.pb'  # Ensure this file is in the correct path
encoder = gdet.create_box_encoder(model_filename, batch_size=1)

max_cosine_distance = 0.7
nn_budget = None
nms_max_overlap = 1.0

metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

# Example YOLO output format: [frame_number, x1, y1, x2, y2, class_id, confidence]
yolo_detections = [
    [0, 100, 150, 200, 250, 0, 0.9],
    [0, 300, 400, 350, 450, 0, 0.85],
    # Add more detections for each frame...
]

image_dir = 'images/'
image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')])

for frame_number, image_path in enumerate(image_files):
    frame = cv2.imread(image_path)
    
    detections = [d for d in yolo_detections if d[0] == frame_number]
    bboxes = np.array([d[1:5] for d in detections])
    scores = np.array([d[6] for d in detections])
    
    # Encode the detections using the feature encoder
    features = encoder(frame, bboxes)
    detections = [Detection(bbox, score, feature) for bbox, score, feature in zip(bboxes, scores, features)]
    
    # Run non-maxima suppression
    boxes = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    #indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
    #detections = [detections[i] for i in indices]
    
    # Update the tracker
    tracker.predict()
    tracker.update(detections)
    
    # Visualize results
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        bbox = track.to_tlbr()
        obj_id = track.track_id
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {obj_id}', (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    
    cv2.imshow('Tracked Particles', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()