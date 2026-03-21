import numpy as np
import matplotlib.pyplot as plt

TOTAL_HANDS = 1000
hands_played = np.arange(0, TOTAL_HANDS)
hands_left = TOTAL_HANDS - hands_played

# 1. Mathematical Win Guarantee (Bleed-out Lock)
# Average fold cost is 1.5 (SB=1, BB=2)
max_bleed = 1.5 * hands_left

# 2. Dynamic Bleed-Out Threshold
# Avoid division by zero for the last hand
safe_hands_left = np.maximum(hands_left, 1)
ratio = 0.33 + (6.0 / np.sqrt(safe_hands_left))
dynamic_threshold = ratio * safe_hands_left

plt.figure(figsize=(12, 8))

# Plot the thresholds
plt.plot(hands_played, max_bleed, label="Hard Lock Threshold (Mathematical Guarantee)", color="red", linestyle="--", linewidth=2)
plt.plot(hands_played, dynamic_threshold, label="Dynamic Bleed-Out Threshold (Soft Lock)", color="orange", linewidth=2)

# Shade the regions
plt.fill_between(hands_played, max_bleed, 2000, color='red', alpha=0.1, label="Hard Lock Region (Fold 100%)")
plt.fill_between(hands_played, dynamic_threshold, max_bleed, color='orange', alpha=0.1, label="Dynamic Bleed-Out Region (Play Extremely Tight)")
plt.fill_between(hands_played, -500, dynamic_threshold, color='green', alpha=0.05, label="Normal Play Region")

# Example trajectories
# Trajectory 1: Very good game, hits dynamic then hard lock
np.random.seed(42)
traj1_pnl = np.cumsum(np.random.normal(1.0, 10.0, TOTAL_HANDS))
# Stop simulating once it hits hard lock, just fold down (-1.5/hand)
for i in range(len(traj1_pnl)):
    if traj1_pnl[i] > max_bleed[i]:
        traj1_pnl[i:] = traj1_pnl[i] - 1.5 * np.arange(0, TOTAL_HANDS - i)
        break
    elif traj1_pnl[i] > dynamic_threshold[i]:
        # Tighter play, lower variance, slight negative drift
        if i + 1 < TOTAL_HANDS:
            traj1_pnl[i+1] = traj1_pnl[i] + np.random.normal(-0.5, 3.0)

plt.plot(hands_played, traj1_pnl, label="Example Bot PnL", color="blue", alpha=0.8, linewidth=1.5)

plt.xlim(0, 1000)
plt.ylim(-200, max(max_bleed[0], 1600))
plt.title("Bot Bleed-Out Logic: PnL Thresholds vs Hands Played", fontsize=14)
plt.xlabel("Hands Played", fontsize=12)
plt.ylabel("Cumulative PnL (Chips)", fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(loc="upper right", fontsize=10)

plt.tight_layout()
plt.savefig("bleedout_visualization.png", dpi=300)
print("Saved bleedout_visualization.png")
