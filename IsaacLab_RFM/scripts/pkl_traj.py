import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np

pkl_path = "/home/phi5090ii/NYX/umi-on-tron-lab/IsaacLab_RFM/data/tossing.pkl" 

with open(pkl_path, "rb") as f:
    episodes = pickle.load(f)

print(f"共 {len(episodes)} 条轨迹")
ep_idx = 10  
pos = np.array(episodes[ep_idx]["ee_pos"])  # (T,3)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
print(pos[0,0], pos[0,1], pos[0,2])
ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], label=f"episode {ep_idx}")
ax.scatter(pos[0, 0], pos[0, 1], pos[0, 2], c="g", s=30, label="start")
ax.scatter(pos[-1, 0], pos[-1, 1], pos[-1, 2], c="r", s=30, label="end")

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.legend()
ax.set_title("EE Trajectory from tossing.pkl")
plt.show()
