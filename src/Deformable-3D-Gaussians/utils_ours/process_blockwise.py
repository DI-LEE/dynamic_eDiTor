from typing import List, Dict, Tuple, Any
import re
from tqdm import tqdm

def build_camtime_grid(frames: List[Any]) -> Tuple[List[int], int, List[List[Any]]]:
    cam_to_list: Dict[int, List[Any]] = {}
    for c in tqdm(frames, desc="Sorting frames by cam_id"):
        cam_idx = int(re.findall(r'\d+', c.cam_id)[0])
        cam_to_list.setdefault(cam_idx, []).append(c)
    for cam_idx in cam_to_list:
        cam_to_list[cam_idx].sort(key=lambda x: x.time)
    cam_ids_sorted = sorted(cam_to_list.keys())
    if not cam_ids_sorted:
        return [], 0, []
    T = min(len(cam_to_list[k]) for k in cam_ids_sorted)
    grid = [[cam_to_list[cam_id][t] for t in range(T)] for cam_id in cam_ids_sorted]
    return cam_ids_sorted, T, grid

def build_mono_grid(viewpoint_stack: List[Any]) -> List[List[Any]]:
    if not viewpoint_stack:
        return []
    
    def get_frame_number(frame):
        nums = re.findall(r'\d+', frame.image_name)
        return int(nums[-1]) if nums else 0
    
    sorted_frames = sorted(viewpoint_stack, key=get_frame_number)
    
    return sorted_frames

def generate_mono_windows(grid: List[Any]):
    if not grid:
        return
    time_length = len(grid)
    
    i = 0
    while i < time_length:
        if i + 3 < time_length:
            yield [(0, i, grid[i]), (0, i+1, grid[i+1]), (0, i+2, grid[i+2]), (0, i+3, grid[i+3])]
            i += 2
        else:
            remaining = [(0, j, grid[j]) for j in range(i, time_length)]
            if remaining:
                yield remaining
            break

def generate_square_windows(grid: List[List[Any]]):
    cam_length = len(grid) # cam length
    if cam_length == 0:
        return
    time_length = len(grid[0]) # time length

    for i in range(0, cam_length - 1):
        if i == 0:
            for j in range(0, time_length - 1, 1):
                f00 = grid[i][j]       # (i, j)
                f10 = grid[i + 1][j]   # (i+1, j)
                f01 = grid[i][j + 1]   # (i, j+1)
                f11 = grid[i + 1][j + 1]  # (i+1, j+1)
                yield [(i, j, f00), (i + 1, j, f10), (i, j + 1, f01), (i + 1, j + 1, f11)]
        else:
            for j in range(0, time_length-1, 2):
                f00 = grid[i][j]
                f10 = grid[i+1][j]
                f01 = grid[i][j+1]
                f11 = grid[i+1][j+1]
                yield [(i, j, f00), (i+1, j, f10), (i, j+1, f01), (i+1, j+1, f11)]

def generate_square_windows_reversed(grid: List[List[Any]]): # grid: [C][T]
    cam_length = len(grid) # cam length
    if cam_length == 0:
        return
    time_length = len(grid[0]) # time length
    
    for i in range(0, time_length - 1):
        if i == 0:
            for j in range(0, cam_length - 1):
                f00 = grid[j][i]
                f10 = grid[j+1][i]
                f01 = grid[j][i+1]
                f11 = grid[j+1][i+1]
                yield [(j, i, f00), (j+1, i, f10), (j, i+1, f01), (j+1, i+1, f11)]
        else:
            for j in range(0, cam_length - 1, 2):
                f00 = grid[j][i]
                f10 = grid[j+1][i]
                f01 = grid[j][i+1]
                f11 = grid[j+1][i+1]
                yield [(j, i, f00), (j+1, i, f10), (j, i+1, f01), (j+1, i+1, f11)]

def generate_grid_windows_for_optflow(grid: List[List[Any]]):
    cam_length = len(grid)
    if cam_length == 0:
        return
    time_length = len(grid[0])

    for i in range(0, cam_length - 1):
        if i % 2 == 0 or i == cam_length - 2:
            for j in range(0, time_length - 1):
                f00 = grid[i][j]
                f10 = grid[i+1][j]
                f01 = grid[i][j+1]
                f11 = grid[i+1][j+1]
                yield [(i, j, f00), (i+1, j, f10), (i, j+1, f01), (i+1, j+1, f11)]

        else:
            f00 = grid[i][0]
            f10 = grid[i+1][0]
            f01 = grid[i][1]
            f11 = grid[i+1][1]
            yield [(i, 0, f00), (i+1, 0, f10), (i, 1, f01), (i+1, 1, f11)]

def generate_vanilla_windows(grid: List[List[Any]]):
    cam_length = len(grid)
    if cam_length == 0:
        return
    time_length = len(grid[0]) # time length
    for i in range(0, cam_length-1, 2):
        for j in range(0, time_length-1, 2):
            yield [(i, j, grid[i][j]), (i+1, j, grid[i+1][j]), (i, j+1, grid[i][j+1]), (i+1, j+1, grid[i+1][j+1])]
