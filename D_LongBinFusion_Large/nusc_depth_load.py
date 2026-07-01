import numpy as np
import os 

filename = '/dataset/nuscenes/samples/DEPTH_MAP/CAM_FRONT/samples/CAM_FRONT/n015-2018-07-24-10-42-41+0800__CAM_FRONT__1532400195612460.jpg.npz' 

if os.path.exists(filename):
    try:
        depth = np.load(filename, allow_pickle=True)["depth"]
    except Exception as e:
        print(f"[WARN] Invalid depth file, will regenerate: {filename} ({e})")
        try:
            os.remove(filename)
            depth = np.zeros((384, 360))
        except OSError:
            pass

    print(depth)

depth = np.load(filename, allow_pickle=True)["depth"]