import pydicom as pd
import cupy as cp


def process_dcm_file(dcm_path , white_threshold=0.7):
    
    ds = pd.dcmread(dcm_path)

    saved_tiles = []
    
    n_frames = int(ds.get("NumberOfFrames", 1))

    for i in range(n_frames):
        
        frame_np = ds.pixel_array[i]

        frame_norm = (frame_np - frame_np.min()) / (frame_np.max() - frame_np.min())
        frame_gpu = cp.array(frame_norm, dtype=cp.float32)

        white_ratio = cp.mean(frame_gpu > 0.9).get()

        if white_ratio < white_threshold:
            saved_tiles.append((i, frame_np.copy()))


    return saved_tiles       
        
            