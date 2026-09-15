"""
Usage:
    python -m starter.camera_transforms --image_size 512
"""
import argparse

import matplotlib.pyplot as plt
import pytorch3d
import pytorch3d.renderer
import pytorch3d.io
import torch

from starter.utils import get_device, get_mesh_renderer


def render_textured_cow(
    cow_path="data/cow.obj",
    image_size=256,
    R_relative=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    T_relative=[0, 0, 0],
    device=None,
):
    if device is None:
        device = get_device()
    meshes = pytorch3d.io.load_objs_as_meshes([cow_path]).to(device)
    R_relative = torch.tensor(R_relative).float()
    T_relative = torch.tensor(T_relative).float()
    R = R_relative @ torch.tensor([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]])
    T = R_relative @ torch.tensor([0.0, 0, 3]) + T_relative
    renderer = get_mesh_renderer(image_size=256, device=device)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R.unsqueeze(0), T=T.unsqueeze(0), device=device,
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -3.0]], device=device,)
    rend = renderer(meshes, cameras=cameras, lights=lights)
    return rend[0, ..., :3].cpu().numpy()


if __name__ == "__main__":

    device = torch.device('cpu')
    
    # rotate camera 90 degrees around z axis 
    R_relative = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
    t1 = render_textured_cow(R_relative=R_relative, device=device)
    plt.imsave("results/textured_cow_t1.jpg", t1)

    # translate camera 2 units towards -Z axis
    T_relative = [0, 0, 2] 
    t2 = render_textured_cow(T_relative=T_relative, device=device)
    plt.imsave("results/textured_cow_t2.jpg", t2)
    
    # translate camera (-0.5, 0.5, 0)
    T_relative = [0.5, -0.5, 0]
    t3 = render_textured_cow(T_relative=T_relative, device=device)
    plt.imsave("results/textured_cow_t3.jpg", t3)
    
    # translate camera to (-3, 0, 0) and look at +X axis
    R_relative = [
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ]
    T_relative = [-3.0, 0.0, 3.0]
    t4 = render_textured_cow(R_relative=R_relative, T_relative=T_relative, device=device)
    plt.imsave("results/textured_cow_t4.jpg", t4)