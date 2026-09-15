from email.mime import image
import imageio
import matplotlib.pyplot as plt
import pytorch3d
import pytorch3d.renderer
import pytorch3d.structures
import torch
import numpy as np   

from starter.utils import get_device, get_mesh_renderer, load_cow_mesh


def render_cow_360degrees_retexture(cow_path="data/cow.obj", image_size=256, color=[0.7, 0.7, 1], device=None):
    
    if device is None:
        device = get_device()
    print(device)
    
    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    
    # Get the vertices, faces, and textures.
    vertices, faces = load_cow_mesh(cow_path)
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    
    color1 = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    color2 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)

    z = vertices[0, :, 2]  # z-coordinate of every vertex
    z_min = z.min()
    z_max = z.max()

    alpha = (z - z_min) / (z_max - z_min)
    alpha = alpha.unsqueeze(1)  # Shape: (N_v, 1)

    vertex_colors = (
        alpha * color2
        + (1.0 - alpha) * color1
    )  # Shape: (N_v, 3)

    textures = vertex_colors.unsqueeze(0)  # Shape: (1, N_v, 3)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)
    
    images = []
    for azim in range(-180, 180, 10):
        img = render(renderer=renderer, mesh=mesh, azim=azim, device=device)
        image_uint8 = (img.clip(0, 1) * 255).astype(np.uint8)
        images.append(image_uint8)
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
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

    rend = renderer(mesh, cameras=cameras, lights=lights)
    rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
    # The .cpu moves the tensor to GPU (if needed).
    return rend

if __name__ == '__main__':
    device = torch.device('cpu')
    my_images = render_cow_360degrees_retexture(device=device)  # List of images [(H, W, 3)]
    duration = 1000 // 15  # Convert FPS (frames per second) to duration (ms per frame)
    imageio.mimsave('results/cow_retexture_360.gif', my_images, duration=duration, loop=0)  # Save the images as a GIF