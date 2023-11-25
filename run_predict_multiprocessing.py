# CUDA_VISIBLE_DEVICES=0 python run_predict_multiprocessing.py --start 0 --stop 25 --n_jobs 5

import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import json
import glob
import argparse

import torch.multiprocessing as mp

try:
     mp.set_start_method('spawn')
except RuntimeError:
    pass
    
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


classes2color = {'car': (255, 0, 0), 'bus': (0, 255, 0), 'truck': (0, 0, 255)}
label2name = {2: 'car', 5: 'bus', 7: 'truck'}
name2label = {v:k for k, v in label2name.items()}


def split_on_chunks(data, n_chunks):
    chunk_size = int(len(data) / n_chunks)
    chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
    return chunks

def do_multiprocess(foo, args, n_jobs):
    with mp.Pool(n_jobs) as pool:
        out = pool.map(foo, args)
    return out

def get_capture_info(cap):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return height, width, fps, frame_count

def prepare_zones(json_data, image):
    zones = []
    for item in json_data['areas']:
        points = []
        for x_n, y_n in item:
            x_n = min(x_n, 1.0)
            y_n = min(y_n, 1.0)
            x = int(x_n * image.shape[1])
            y = int(y_n * image.shape[0])
            points.append([x, y])
        zones.append(points)
    return zones

def visialize_points(json_data, img):
    image = img.copy()
    for item in json_data['areas']:
        for x_n, y_n in item:
            x_n = min(x_n, 1.0)
            y_n = min(y_n, 1.0)
            x = int(x_n * image.shape[1])
            y = int(y_n * image.shape[0])
            image = cv2.circle(image, (x, y), radius=10, color=(0, 0, 255), thickness=-1)
            
    return image
    
def shrink_poly(poly, x_shrink, y_shrink):
   
    coords = poly
    xs = [i[0] for i in coords]
    ys = [i[1] for i in coords]

    # simplistic way of calculating a center of the graph, you can choose your own system
    x_center = 0.5 * min(xs) + 0.5 * max(xs)
    y_center = 0.5 * min(ys) + 0.5 * max(ys)

    # shrink figure
    new_xs = [(i - x_center) * (1 - x_shrink) + x_center for i in xs]
    new_ys = [(i - y_center) * (1 - y_shrink) + y_center for i in ys]

    # create list of new coordinates

    new_coords = []
    for x, y in zip(new_xs, new_ys):
        new_coords.append([int(x), int(y)])
        
    return np.array((new_coords), np.int32)

def find_corners(poly):
    x2, y2 = np.max(poly, axis=0)
    x1, y1 = np.min(poly, axis=0)
    w = x2 - x1
    h = y2 - y1
    xc, yc = w // 2 + x1, h // 2 + y1
    return [x1, y1, x2, y2, w, h, xc, yc]

def move_poly(poly, gap_x, gap_y):
    new_poly = []
    for x, y in poly:
        new_x = x - gap_x
        new_y = y - gap_y
        new_poly.append([new_x, new_y])
        
    return np.array((new_poly), np.int32)


def move_and_shrink_poly(poly, bbox):
    poly_info = find_corners(poly)
    
    x1b, y1b, x2b, y2b = bbox
    wb = x2b - x1b 
    hb = (y2b - y1b) //2
    y1b = y1b + hb
    xcb, ycb = wb // 2 + x1b, hb // 2 + y1b
    
    coef_x, coef_y = 1 - (wb / poly_info[4]), 1 - (hb / poly_info[5])
  
    new_poly = shrink_poly(poly, coef_x, coef_y)
    
    new_poly_info = find_corners(new_poly)
    
    gap_x = new_poly_info[6] - xcb
    gap_y = new_poly_info[7] - ycb
       
    new_poly_moved = move_poly(new_poly, gap_x, gap_y)  
    
    poly_right_bottom_x,  poly_right_bottom_y = new_poly_moved[2][0], new_poly_moved[2][1]
    poly_left_bottom_x,  poly_left_bottom_y = new_poly_moved[3][0], new_poly_moved[3][1]
    
    poly_center_y = poly_right_bottom_y + (y2b - poly_right_bottom_y) // 2
    poly_center_x = x1b + wb // 2
    
    return new_poly_moved, (poly_center_x, poly_center_y)

def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def run_process(params):
    video_paths, model_path, json_folder, chunk_index = params
    print(f'[ CHUNK INDEX: {chunk_index} | VIDEO COUNT: {len(video_paths)} ]')
    
    model  = YOLO(model_path)
    
    values = {
        'file_name': [], 
        'quantity_car': [], 
        'average_speed_car': [], 
        'quantity_van': [], 
        'average_speed_van': [], 
        'quantity_bus': [], 
        'average_speed_bus': [],
    }
    
    for video_path in video_paths:
        zoned_frames = []
        tracks_info = {}
        objects_in_area = {}
        
        base_name = os.path.basename(video_path).replace('.mp4', '')
        json_path = os.path.join(json_folder, f'{base_name}.json')
        
        data = load_json(json_path)  # '../markup/jsons/KRA-3-12-2023-08-21-evening.json'
        cap = cv2.VideoCapture(video_path)  # '../videos/KRA-3-12-2023-08-21-evening_Trim_1.mp4'
        height, width, fps, frame_count = get_capture_info(cap)
        
        for frame_num in tqdm(range(frame_count)):
            tracks_info[frame_num] = {}
            
            success, img = cap.read()
            if img is None:
                break
            results = model.track(img, persist=True, classes=[2, 5, 7], verbose=False, conf=0.5, tracker='bytetrack.yaml')
            
            detections = []
            polygons = []
            zones = prepare_zones(data, img)
            for zone in zones:
                polygons.append(Polygon(zone))
            
            for track in results[0].boxes:
                if not track.is_track:
                    continue
                track_id = track.id[0].cpu().item()
                x1, y1, x2, y2 = [int(x) for x in track.xyxy[0].cpu().numpy()]
        
                speed_vector = None
                point_x = x1 + (x2 - x1) // 2
                point_y = y1 + (y2 - y1) // 2
        
                _, (point_x, point_y) = move_and_shrink_poly(zones[0], [x1, y1, x2, y2])
                
                point = Point(point_x, point_y)
        
                in_area = any([point.within(x) for x in polygons])
                tracks_info[frame_num][track_id] = {
                    'class': track.cls[0].cpu().item(),
                    'box': [x1, y1, x2, y2],
                    'speed_vector': speed_vector,
                    'in_area': in_area,
                    'track_point': (point_x, point_y),
                }
        
                if in_area:
                    if track_id not in objects_in_area:
                        objects_in_area[track_id] = {
                            'class': track.cls[0].cpu().item(),
                            'frames_in_area': 1,
                            'crossed': False
                        }
                    else:
                        objects_in_area[track_id]['frames_in_area'] += 1
                        
                if ((frame_num - 1) in tracks_info) and (track_id in objects_in_area) and (not objects_in_area[track_id]['crossed']):
                    if (not in_area) and (track_id in tracks_info[frame_num - 1]) and (tracks_info[frame_num - 1][track_id]['in_area']):
                        objects_in_area[track_id]['crossed'] = True
                        
        cap.release()
        
        object_speed = {'car': [], 'bus': [], 'truck': []}
        object_count = {'car': 0, 'bus': 0, 'truck': 0}
        
        for track_id, object_info in objects_in_area.items():
            if not object_info['crossed']:
                continue
            cls = object_info['class']
            class_name = label2name[cls]

            if object_info['frames_in_area'] < 15:
                continue
            
            object_count[class_name] += 1
            speed = (20  / (object_info['frames_in_area'] / fps)) * 3.6
            object_speed[class_name].append(speed)
        
        mean_speeds = {
            'car': np.mean(object_speed['car']) if len(object_speed['car']) else 0, 
            'bus': np.mean(object_speed['bus']) if len(object_speed['bus']) else 0, 
            'truck': np.mean(object_speed['truck']) if len(object_speed['truck']) else 0,
        }
    
        values['file_name'].append(base_name)
        
        values['quantity_car'].append(object_count['car'])
        values['average_speed_car'].append(round(mean_speeds['car'], 2))
        
        values['quantity_van'].append(object_count['truck'])
        values['average_speed_van'].append(round(mean_speeds['truck'], 2))
        
        values['quantity_bus'].append(object_count['bus'])
        values['average_speed_bus'].append(round(mean_speeds['bus'], 2))
        
    df = pd.DataFrame(values)
    # df.to_csv(f'chunk_{chunk_index}.csv', index=False) 
    return df


if __name__ == '__main__':
    # CUDA_VISIBLE_DEVICES=0 python run_predict_multiprocessing.py --start=0 --end=25 --n_jobs=5 
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--n_jobs", type=int, required=True)
    parser.add_argument("--model_path", type=str, required=True)  # './yolov8l.engine'
    parser.add_argument("--video_folder", type=str, required=True)  # '/home/raid_storage/datasets/hacks/test/videos'
    parser.add_argument("--json_folder", type=str, required=True)  # '/home/raid_storage/datasets/hacks/test/annotations'

    args = parser.parse_args()

    print(f'START VIDEO NUM: {args.start} | END VIDEO NUM: {args.end}')
    video_paths = glob.glob(os.path.join(args.video_folder, '*.mp4'))
    video_paths = list(sorted(video_paths))[args.start:args.end]
    
    video_paths_chunks = split_on_chunks(video_paths, args.n_jobs)
    print(f'[ TOTAL VIDEOS COUNT: {len(video_paths)} | CHUNKS: {len(video_paths_chunks)} ]')
    process_params = [(chunk, args.model_path, args.json_folder, index) for index, chunk in enumerate(video_paths_chunks)]
    
    df_results = do_multiprocess(run_process, process_params, args.n_jobs)
    df_result = pd.concat(df_results, ignore_index=True)
    df_result.to_csv(f'result-{args.start}-{args.end}.csv', index=False)
    