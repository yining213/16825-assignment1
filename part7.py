"""
Extra credit 7: Uniformly sample points on a triangle mesh via stratified sampling.

Usage:
    python -m part7
"""
import imageio
import numpy as np
import pytorch3d
import pytorch3d.renderer
import pytorch3d.structures
import torch

from starter.utils import get_device, get_mesh_renderer, get_points_renderer, load_cow_mesh


def sample_points_on_mesh(vertices, faces, num_samples, generator=None):

    device = vertices.device
    triangles = vertices[faces]  # (N_f, 3, 3)

    # Step 1: Sampling proportional to area makes the result uniform over the surface
    edge1 = triangles[:, 1] - triangles[:, 0]
    edge2 = triangles[:, 2] - triangles[:, 0]
    areas = 0.5 * torch.linalg.cross(edge1, edge2, dim=1).norm(dim=1)  # (N_f,)

    face_idx = torch.multinomial(
        areas, num_samples, replacement=True, generator=generator
    ) # randomly choose num_samples faces, with probability proportional to area
    sampled = triangles[face_idx]  # (num_samples, 3, 3)

    # Step 2: uniform barycentric coordinates (w, u, v) for each sampled face
    uv = torch.rand(num_samples, 2, device=device, generator=generator) # (num_samples, 2), uniformly sample u, v in [0, 1]
    over = uv.sum(dim=1) > 1.0
    uv[over] = 1.0 - uv[over]
    u = uv[:, 0:1]
    v = uv[:, 1:2]
    w = 1.0 - u - v

    # Step 3: evaluate the barycentric combination on the chosen face.
    points = w * sampled[:, 0] + u * sampled[:, 1] + v * sampled[:, 2]
    return points


def render_sampled_vs_mesh(
    num_samples,
    cow_path="data/cow.obj",
    image_size=256,
    color=[0.7, 0.7, 1],
    point_radius=0.01,
    device=None,
    generator=None,
):
    
    if device is None:
        device = get_device()

    vertices, faces = load_cow_mesh(cow_path)
    vertices = vertices.to(device)
    faces = faces.to(device)

    points = sample_points_on_mesh(vertices, faces, num_samples, generator=generator)
    point_colors = (points - points.min()) / (points.max() - points.min())
    point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[point_colors]
    ).to(device)

    textures = torch.ones_like(vertices).unsqueeze(0) * torch.tensor(color, device=device)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices.unsqueeze(0),
        faces=faces.unsqueeze(0),
        textures=pytorch3d.renderer.TexturesVertex(textures),
    ).to(device)

    points_renderer = get_points_renderer(
        image_size=image_size, radius=point_radius, background_color=(1, 1, 1),
        device=device,
    )
    mesh_renderer = get_mesh_renderer(image_size=image_size, device=device)
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

    frames = []
    for azim in range(-180, 180, 10):
        R, T = pytorch3d.renderer.look_at_view_transform(
            dist=3, elev=0, azim=azim, at=((0, 0, 0),), device=device
        )
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, fov=60, device=device)

        point_image = points_renderer(point_cloud, cameras=cameras)[0, ..., :3]
        mesh_image = mesh_renderer(mesh, cameras=cameras, lights=lights)[0, ..., :3]

        side_by_side = torch.cat([point_image, mesh_image], dim=1).cpu().numpy()
        frames.append((side_by_side.clip(0, 1) * 255).astype(np.uint8))

    return frames


if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}")

    generator = torch.Generator(device=device).manual_seed(0)
    duration = 1000 // 15

    for num_samples in [10, 100, 1000, 10000]:
        frames = render_sampled_vs_mesh(
            num_samples, device=device, generator=generator
        )
        output_path = f"results/sample_{num_samples}.gif"
        imageio.mimsave(output_path, frames, duration=duration, loop=0)
        print(f"wrote {output_path} ({num_samples} points)")
