import torch
from manotorch.manolayer import ManoLayer, MANOOutput

# Select number of principal components for pose space
ncomps = 15

# initialize layers
mano_layer = ManoLayer(use_pca=True, flat_hand_mean=False, ncomps=ncomps)

batch_size = 2
# Generate random shape parameters
random_shape = torch.rand(batch_size, 10)
# Generate random pose parameters, including 3 values for global axis-angle rotation
random_pose = torch.rand(batch_size, 3 + ncomps)

# The mano_layer's output contains:
"""
MANOOutput = namedtuple(
    "MANOOutput",
    [
        "verts",
        "joints",
        "center_idx",
        "center_joint",
        "full_poses",
        "betas",
        "transforms_abs",
    ],
)
"""
# forward mano layer
mano_output: MANOOutput = mano_layer(random_pose, random_shape)

# retrieve 778 vertices, 21 joints and 16 SE3 transforms of each articulation
verts = mano_output.verts  # (B, 778, 3), root(center_joint) relative
joints = mano_output.joints  # (B, 21, 3), root relative
transforms_abs = mano_output.transforms_abs  # (B, 16, 4, 4), root relative

# Print results
print("Random Shape Parameters:\n", random_shape)
print("\nRandom Pose Parameters:\n", random_pose)
print("\nVertices (verts):\n", verts)
print("\nJoints (joints):\n", joints)
print("\nAbsolute Transforms (transforms_abs):\n", transforms_abs)