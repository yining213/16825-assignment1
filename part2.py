import argparse

import matplotlib.pyplot as plt
import pytorch3d
import torch
import pytorch3d.structures
import pytorch3d.renderer
import numpy as np
import imageio

from starter.utils import get_device, get_mesh_renderer, load_cow_mesh

def render_360degree_tetrahedron(device=None): 
    if device is None:
        device = get_device()
        
    vertices = torch.tensor([
        [0.0,  1.0,  0.0],   # vertex 0
        [-1.0, -1.0, 1.0],   # vertex 1
        [1.0,  -1.0, 1.0],   # vertex 2
        [0.0,  -1.0, -1.0],  # vertex 3
    ], dtype=torch.float32)
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)

    faces = torch.tensor([
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3]
    ], dtype=torch.long)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    
    color = [0.7, 0.7, 1]
    textures = torch.ones_like(vertices) * torch.tensor(color)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)
    
    images = []
    renderer = get_mesh_renderer(image_size=256, device=device)
    
    for azim in range(0, 360, 10):
        img = render(renderer=renderer, mesh=mesh, azim=azim, device=device)
        img_uint8 = (img.clip(0, 1) * 255).astype(np.uint8)
        images.append(img_uint8)
    
    return images

def render_360degree_cube(device=None):
    if device is None:
        device = get_device()

    vertices = torch.tensor([
        [-1.0, -1.0, -1.0],  # 0
        [-1.0, -1.0,  1.0],  # 1
        [-1.0,  1.0, -1.0],  # 2
        [-1.0,  1.0,  1.0],  # 3
        [ 1.0, -1.0, -1.0],  # 4
        [ 1.0, -1.0,  1.0],  # 5
        [ 1.0,  1.0, -1.0],  # 6
        [ 1.0,  1.0,  1.0],  # 7
    ], dtype=torch.float32).unsqueeze(0)

    faces = torch.tensor([
        [1, 5, 7], [1, 7, 3],  # z = +1
        [0, 2, 6], [0, 6, 4],  # z = -1
        [0, 1, 3], [0, 3, 2],  # x = -1
        [4, 6, 7], [4, 7, 5],  # x = +1
        [2, 3, 7], [2, 7, 6],  # y = +1
        [0, 4, 5], [0, 5, 1],  # y = -1
    ], dtype=torch.int64).unsqueeze(0)

    color = torch.tensor([0.2, 0.7, 1.0])
    textures = torch.ones_like(vertices) * color

    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    ).to(device)

    renderer = get_mesh_renderer(image_size=256, device=device)
    images = []

    for azim in range(0, 360, 10):
        R, T = pytorch3d.renderer.look_at_view_transform(
            dist=4,
            elev=20,
            azim=azim,
            device=device,
        )

        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R,
            T=T,
            fov=60,
            device=device,
        )

        lights = pytorch3d.renderer.PointLights(
            location=[[0.0, 0.0, -3.0]],
            device=device,
        )

        render = renderer(mesh, cameras=cameras, lights=lights)
        image = render[0, ..., :3].cpu().numpy()
        image = (image.clip(0, 1) * 255).astype(np.uint8)
        images.append(image)

    return images

def render(
    renderer: pytorch3d.renderer.MeshRenderer,
    mesh: pytorch3d.structures.Meshes, 
    elev = 0, azim = 0, dist = 3, 
    device=None,
):
    if device is None:
        device = get_device()
    
    # Prepare the camera:
    R, T = pytorch3d.renderer.look_at_view_transform(dist=dist, elev=elev, azim=azim, 
                                                     at = ((0, 0, 0),), device=device)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R, T=T, fov=60, device=device
    )

    # Place a point light in front of the cow.
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, 3]], device=device)

    rend = renderer(mesh, cameras=cameras, lights=lights)
    rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
    # The .cpu moves the tensor to GPU (if needed).
    return rend

if __name__ == "__main__":
    device = torch.device("cpu")
    images_tetrahedron = render_360degree_tetrahedron(device=device)
    images_cube = render_360degree_cube(device=device)
    duration = 1000 // 15  # Convert FPS (frames per second) to duration (ms per frame)
    imageio.mimsave('results/tetrahedron_360.gif', images_tetrahedron, duration=duration, loop=0)  # Save the images as a GIF
    imageio.mimsave('results/cube_360.gif', images_cube, duration=duration, loop=0)  # Save the images as a GIF