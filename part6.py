"""
Extra credit 6: Metaball morph -- an implicit-surface animation in which the
topology of the rendered surface changes over time.

Two metaballs orbit toward each other. The surface is the zero level-set of

    F(x, t) = sum_i r_i^2 / ||x - c_i(t)||^2 - 1

which is re-evaluated on a voxel grid and re-meshed with marching cubes at every
frame. When the centers are far apart the level-set has two connected components;
below a critical separation they merge into one. That merge is a change of
topology, which a fixed-connectivity mesh or a parametric function cannot express
-- it falls out of the implicit formulation for free.

For two equal balls of radius parameter r placed at (+/-s, 0, 0), the field at the
origin is 2r^2/s^2 - 1, so the surface first touches itself when s = r*sqrt(2).

Usage:
    python -m part6
"""
import imageio
import mcubes
import numpy as np
import pytorch3d
import pytorch3d.renderer
import pytorch3d.structures
import torch

from starter.utils import get_device, get_mesh_renderer

GRID_MIN = -2.4
GRID_MAX = 2.4
BALL_RADIUS = 0.78
# Separation at which the two components merge (see module docstring).
CRITICAL_SEP = BALL_RADIUS * np.sqrt(2.0)


def metaball_field(grid, centers, radius=BALL_RADIUS, eps=1e-8):
    """
    Evaluates the metaball implicit function on a voxel grid.

    Args:
        grid (torch.Tensor): Sample positions (G, G, G, 3).
        centers (torch.Tensor): Metaball centers (K, 3).
        radius (float): Radius parameter shared by all balls.
        eps (float): Guard against division by zero at a center.

    Returns:
        torch.Tensor: Field values (G, G, G); the surface is the 0 level-set.
    """
    field = torch.full(grid.shape[:3], -1.0, device=grid.device)
    for center in centers:
        sq_dist = ((grid - center) ** 2).sum(dim=-1)
        field = field + radius**2 / (sq_dist + eps)
    return field


def ball_centers(t):
    """
    Returns the metaball centers at animation time t in [0, 1).

    The two balls swing together and apart on a cosine, so the loop is seamless,
    and counter-rotate in the xy-plane so the merge is visible from any azimuth.
    """
    # Separation sweeps 2.06 -> 0.42 -> 2.06, crossing CRITICAL_SEP twice.
    sep = 1.24 + 0.82 * np.cos(2 * np.pi * t)
    angle = 2 * np.pi * t
    offset = torch.tensor(
        [sep * np.cos(angle), sep * np.sin(angle), 0.0], dtype=torch.float32
    )
    return torch.stack([offset, -offset]), sep


def field_to_mesh(field, device):
    """
    Extracts the 0 level-set of a voxel field as a PyTorch3D mesh, rescaling
    marching-cubes index coordinates back into world coordinates.
    """
    voxel_size = field.shape[0]
    vertices, faces = mcubes.marching_cubes(field.cpu().numpy(), isovalue=0)
    vertices = torch.tensor(vertices.copy(), dtype=torch.float32)
    faces = torch.tensor(faces.astype(np.int64))
    vertices = (vertices / (voxel_size - 1)) * (GRID_MAX - GRID_MIN) + GRID_MIN

    # Color by distance from the origin so the neck that forms during the merge
    # reads differently from the two lobes.
    radial = vertices.norm(dim=1, keepdim=True)
    radial = (radial - radial.min()) / (radial.max() - radial.min() + 1e-8)
    colors = torch.cat(
        [0.35 + 0.6 * radial, 0.45 * torch.ones_like(radial), 1.0 - 0.5 * radial],
        dim=1,
    )

    return pytorch3d.structures.Meshes(
        [vertices], [faces],
        textures=pytorch3d.renderer.TexturesVertex(colors.unsqueeze(0)),
    ).to(device)


def render_metaball_morph(
    num_frames=48, voxel_size=80, image_size=256, device=None,
):
    """
    Renders the morph over one full loop, orbiting the camera at the same time.
    """
    if device is None:
        device = get_device()

    axis = torch.linspace(GRID_MIN, GRID_MAX, voxel_size, device=device)
    X, Y, Z = torch.meshgrid(axis, axis, axis, indexing="ij")
    grid = torch.stack([X, Y, Z], dim=-1)

    renderer = get_mesh_renderer(image_size=image_size, device=device)
    lights = pytorch3d.renderer.PointLights(location=[[0.0, 2.0, -4.0]], device=device)

    frames = []
    for i in range(num_frames):
        t = i / num_frames
        centers, sep = ball_centers(t)
        field = metaball_field(grid, centers.to(device))
        mesh = field_to_mesh(field, device)

        azim = 360.0 * t
        R, T = pytorch3d.renderer.look_at_view_transform(
            dist=5.2, elev=12, azim=azim, device=device
        )
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)

        image = renderer(mesh, cameras=cameras, lights=lights)[0, ..., :3]
        image = image.detach().cpu().numpy().clip(0, 1)
        frames.append((image * 255).astype(np.uint8))

        state = "merged" if sep < CRITICAL_SEP else "separate"
        print(f"  frame {i + 1:2d}/{num_frames}  sep={sep:.3f}  {state}")

    return frames


if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}")
    print(f"Critical separation r*sqrt(2) = {CRITICAL_SEP:.4f}")

    frames = render_metaball_morph(device=device)
    imageio.mimsave("results/metaball_morph.gif", frames, duration=1000 // 20, loop=0)
    print("wrote results/metaball_morph.gif")
