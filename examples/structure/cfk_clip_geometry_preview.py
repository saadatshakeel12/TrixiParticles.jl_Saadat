#!/usr/bin/env python3
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Dimensions in meters (kept consistent with the new Julia example)
charge_length = 0.060
charge_width = 0.022
charge_thickness = 0.0035

tool_length = charge_length + 0.014
tool_width = charge_width + 0.014
feature_w = 0.0035
initial_gap = 0.002
base_thickness = 3 * 0.002
feature_height = 3 * 0.002

# Main bounds
x0 = -0.5 * charge_length
x1 = 0.5 * charge_length
y0 = -0.5 * charge_width
y1 = 0.5 * charge_width

tx0 = -0.5 * tool_length
tx1 = 0.5 * tool_length
ty0 = -0.5 * tool_width
ty1 = 0.5 * tool_width

z0 = 0.0
z1 = z0 + charge_thickness
floor_top = z0 - initial_gap
floor_bottom = floor_top - base_thickness
mold_bottom = z1 + initial_gap
mold_top = mold_bottom + base_thickness

fig, (ax_top, ax_side) = plt.subplots(1, 2, figsize=(12, 5.5))

# Top view (X-Y)
ax_top.add_patch(Rectangle((tx0, ty0), tool_length, tool_width,
                           facecolor="#4daf4a", edgecolor="k", alpha=0.25, label="Lower tool envelope"))
ax_top.add_patch(Rectangle((tx0, ty0), tool_length, feature_w,
                           facecolor="#1b9e77", edgecolor="k", alpha=0.6, label="Female side rails"))
ax_top.add_patch(Rectangle((tx0, ty1 - feature_w), tool_length, feature_w,
                           facecolor="#1b9e77", edgecolor="k", alpha=0.6))
ax_top.add_patch(Rectangle((-0.325 * tool_length, -0.5 * feature_w),
                           0.65 * tool_length, feature_w,
                           facecolor="#66a61e", edgecolor="k", alpha=0.75, label="Female center rib"))

ax_top.add_patch(Rectangle((x0, y0), charge_length, charge_width,
                           facecolor="#d95f02", edgecolor="k", alpha=0.75, label="CFK raw charge"))

ax_top.add_patch(Rectangle((-0.275 * tool_length, -0.5 * feature_w),
                           0.55 * tool_length, feature_w,
                           facecolor="#1f78b4", edgecolor="k", alpha=0.75, label="Male center punch"))
ax_top.add_patch(Rectangle((-0.35 * tool_length, ty0 + 0.22 * tool_width),
                           0.70 * tool_length, feature_w,
                           facecolor="#1f78b4", edgecolor="k", alpha=0.7, label="Male shoulders"))
ax_top.add_patch(Rectangle((-0.35 * tool_length, ty0 + 0.78 * tool_width - feature_w),
                           0.70 * tool_length, feature_w,
                           facecolor="#1f78b4", edgecolor="k", alpha=0.7))

ax_top.set_title("Top View (X-Y)")
ax_top.set_xlabel("X [m]")
ax_top.set_ylabel("Y [m]")
ax_top.set_aspect("equal", adjustable="box")
ax_top.set_xlim(tx0 - 0.004, tx1 + 0.004)
ax_top.set_ylim(ty0 - 0.004, ty1 + 0.004)
ax_top.grid(alpha=0.25)

# Side section (Y=0 cut, X-Z)
ax_side.add_patch(Rectangle((tx0, floor_bottom), tool_length, base_thickness,
                            facecolor="#4daf4a", edgecolor="k", alpha=0.4, label="Lower base"))
ax_side.add_patch(Rectangle((-0.325 * tool_length, floor_top),
                            0.65 * tool_length, feature_height,
                            facecolor="#66a61e", edgecolor="k", alpha=0.8, label="Female rib"))

ax_side.add_patch(Rectangle((x0, z0), charge_length, charge_thickness,
                            facecolor="#d95f02", edgecolor="k", alpha=0.85, label="CFK raw charge"))

ax_side.add_patch(Rectangle((tx0, mold_bottom), tool_length, base_thickness,
                            facecolor="#377eb8", edgecolor="k", alpha=0.4, label="Upper holder"))
ax_side.add_patch(Rectangle((-0.275 * tool_length, mold_bottom - feature_height),
                            0.55 * tool_length, feature_height,
                            facecolor="#1f78b4", edgecolor="k", alpha=0.8, label="Male punch"))

ax_side.set_title("Section View (X-Z at Y=0)")
ax_side.set_xlabel("X [m]")
ax_side.set_ylabel("Z [m]")
ax_side.set_xlim(tx0 - 0.004, tx1 + 0.004)
ax_side.set_ylim(floor_bottom - 0.004, mold_top + 0.004)
ax_side.grid(alpha=0.25)

handles, labels = ax_top.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True)
fig.suptitle("CFK Clip Tooling Concept and Raw Charge Geometry", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("cfk_clip_geometry_preview.png", dpi=240)
print("Saved cfk_clip_geometry_preview.png")
