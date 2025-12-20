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

def generate_subgrid(grid: List[List[Any]]):
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


