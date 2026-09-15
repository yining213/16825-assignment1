"""
Sample code to render various representations.

Usage:
    python -m starter.render_generic --render point_cloud  # 5.1
    python -m starter.render_generic --render parametric  --num_samples 100  # 5.2
    python -m starter.render_generic --render implicit  # 5.3
"""
import argparse
import pickle

import matplotlib.pyplot as plt
import mcubes
import numpy as np
import pytorch3d
import pytorch3d.structures
import pytorch3d.renderer
import torch
import imageio

from starter.utils import get_device, get_mesh_renderer, get_points_renderer, unproject_depth_image


def load_rgbd_data(path="data/rgbd_data.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def render_bridge(
    point_cloud_path="data/bridge_pointcloud.npz",
    image_size=256,
    background_color=(1, 1, 1),
    device=None,
):
    """
    Renders a point cloud.
    """
    if device is None:
        device = get_device()
    renderer = get_points_renderer(
        image_size=image_size, background_color=background_color
    )
    point_cloud = np.load(point_cloud_path)
    verts = torch.Tensor(point_cloud["verts"][::50]).to(device).unsqueeze(0)
    rgb = torch.Tensor(point_cloud["rgb"][::50]).to(device).unsqueeze(0)
    point_cloud = pytorch3d.structures.Pointclouds(points=verts, features=rgb)
    R, T = pytorch3d.renderer.look_at_view_transform(4, 10, 0)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rend = renderer(point_cloud, cameras=cameras)
    rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
    return rend


def render_sphere(image_size=256, num_samples=200, device=None):
    """
    Renders a sphere using parametric sampling. Samples num_samples ** 2 points.
    """

    if device is None:
        device = get_device()

    phi = torch.linspace(0, 2 * np.pi, num_samples)
    theta = torch.linspace(0, np.pi, num_samples)
    # Densely sample phi and theta on a grid
    Phi, Theta = torch.meshgrid(phi, theta)

    x = torch.sin(Theta) * torch.cos(Phi)
    y = torch.cos(Theta)
    z = torch.sin(Theta) * torch.sin(Phi)

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())

    sphere_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],
    ).to(device)

    cameras = pytorch3d.renderer.FoVPerspectiveCameras(T=[[0, 0, 3]], device=device)
    renderer = get_points_renderer(image_size=image_size, device=device)
    rend = renderer(sphere_point_cloud, cameras=cameras)
    return rend[0, ..., :3].cpu().numpy()

def render_torus_parametric(image_size=256, num_samples=400, device=None):
    """
    Renders a torus using parametric sampling. Samples num_samples ** 2 points.
    """

    if device is None:
        device = get_device()

    phi = torch.linspace(0, 2 * np.pi, num_samples)
    theta = torch.linspace(0, 2 * np.pi, num_samples)
    # Densely sample phi and theta on a grid
    Phi, Theta = torch.meshgrid(phi, theta)

    R = 1.0  # Major radius
    r = 0.3  # Minor radius

    x = (R + r * torch.cos(Theta)) * torch.cos(Phi)
    y = (R + r * torch.cos(Theta)) * torch.sin(Phi)
    z = r * torch.sin(Theta)

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())

    torus_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],
    ).to(device)

    renderer = get_points_renderer(image_size=image_size, background_color=(0, 0, 0), device=device)
    images = []
    for azim in range(0, 360, 10):
        R, T = pytorch3d.renderer.look_at_view_transform(dist=3, elev=0, azim=azim, device=device)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        rend = renderer(torus_point_cloud, cameras=cameras)
        image = rend[0, ..., :3].cpu().numpy()
        image_uint8 = (image.clip(0, 1) * 255).astype(np.uint8)
        images.append(image_uint8)
    return images

def render_mobius_parametric(image_size=256, num_samples=200, device=None):
    if device is None:
        device = get_device()

    u = torch.linspace(0, 2 * np.pi, num_samples)
    v = torch.linspace(-0.35, 0.35, num_samples)

    U, V = torch.meshgrid(u, v, indexing="ij")

    # Möbius strip parameterization
    x = (1 + V * torch.cos(U / 2)) * torch.cos(U)
    y = (1 + V * torch.cos(U / 2)) * torch.sin(U)
    z = V * torch.sin(U / 2)

    points = torch.stack(
        [x.flatten(), y.flatten(), z.flatten()],
        dim=1,
    )

    color = (points - points.min()) / (points.max() - points.min())

    mobius = pytorch3d.structures.Pointclouds(
        points=[points],
        features=[color],
    ).to(device)

    renderer = get_points_renderer(
        image_size=image_size,
        background_color=(0, 0, 0),
        device=device,
    )

    images = []

    for azim in range(0, 360, 10):
        R, T = pytorch3d.renderer.look_at_view_transform(
            dist=3.5,
            elev=20,
            azim=azim,
            device=device,
        )

        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R,
            T=T,
            device=device,
        )

        rend = renderer(mobius, cameras=cameras)
        image = rend[0, ..., :3].cpu().numpy()
        image = (image.clip(0, 1) * 255).astype(np.uint8)
        images.append(image)

    return images

def render_sphere_mesh(image_size=256, voxel_size=64, device=None):
    if device is None:
        device = get_device()
    min_value = -1.1
    max_value = 1.1
    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)
    voxels = X ** 2 + Y ** 2 + Z ** 2 - 1
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
        device
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device,)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    R, T = pytorch3d.renderer.look_at_view_transform(dist=3, elev=0, azim=180)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rend = renderer(mesh, cameras=cameras, lights=lights)
    return rend[0, ..., :3].detach().cpu().numpy().clip(0, 1)

def render_torus_implicit_mesh(image_size=256, device=None):
    if device is None:
        device = get_device()
    R, r = 1.0, 0.3  # Major and minor radii
    min_value = -(R + r) - 0.1
    max_value = (R + r) + 0.1

    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, 64)] * 3)
    voxels = (torch.sqrt(X**2 + Y**2) - R) ** 2 + Z**2 - r**2
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    vertices = (vertices / 64) * (max_value - min_value) + min_value
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0)) 
    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(device)
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device,)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    images = []
    for azim in range(0, 360, 10):
        R, T = pytorch3d.renderer.look_at_view_transform(dist=3, elev=0, azim=azim)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        rend = renderer(mesh, cameras=cameras, lights=lights)
        image = rend[0, ..., :3].detach().cpu().numpy().clip(0, 1)
        image_uint8 = (image.clip(0, 1) * 255).astype(np.uint8)
        images.append(image_uint8)
    return images

def render_gyroid_implicit_mesh(image_size=256, voxel_size=96, device=None):
    if device is None:
        device = get_device()

    min_value = -2 * np.pi
    max_value = 2 * np.pi
    axis = torch.linspace(min_value, max_value, voxel_size)
    X, Y, Z = torch.meshgrid(axis, axis, axis, indexing="ij")
    voxels = (
        torch.sin(X) * torch.cos(Y)
        + torch.sin(Y) * torch.cos(Z)
        + torch.sin(Z) * torch.cos(X)
    )

    vertices, faces = mcubes.marching_cubes(
        voxels.numpy(),
        isovalue=0,
    )
    vertices = torch.tensor(vertices, dtype=torch.float32)
    faces = torch.tensor(faces.astype(np.int64))
    vertices = (
        vertices / (voxel_size - 1) * (max_value - min_value)
        + min_value
    )

    colors = (vertices - vertices.min()) / (
        vertices.max() - vertices.min()
    )
    textures = pytorch3d.renderer.TexturesVertex(
        colors.unsqueeze(0)
    )
    mesh = pytorch3d.structures.Meshes(
        [vertices],
        [faces],
        textures=textures,
    ).to(device)

    lights = pytorch3d.renderer.PointLights(
        location=[[0.0, 0.0, -4.0]],
        device=device,
    )
    renderer = get_mesh_renderer(
        image_size=image_size,
        device=device,
    )

    images = []
    for azim in range(0, 360, 10):
        R, T = pytorch3d.renderer.look_at_view_transform(
            dist=25,
            elev=20,
            azim=azim,
            device=device,
        )
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R,
            T=T,
            device=device,
        )
        rend = renderer(mesh, cameras=cameras, lights=lights)
        image = rend[0, ..., :3].detach().cpu().numpy().clip(0, 1)
        images.append((image * 255).astype(np.uint8))

    return images

def drop_invalid(points, colors):
    """
    Removes non-finite points (and their colors) from an unprojected point cloud.
    """
    keep = torch.isfinite(points).all(dim=1)
    return points[keep], colors[keep]


def render_plant_rgbd(image_size=256, rgbd_data_path="data/rgbd_data.pkl", device=None):
    if device is None:
        device = get_device()
    data = load_rgbd_data(rgbd_data_path)
    
    print(type(data))
    print(data.keys())
    
    # viewpoint 1
    rgb1 = torch.tensor(data["rgb1"]).float().to(device)
    mask1 = torch.tensor(data["mask1"]).float().to(device)
    depth1 = torch.tensor(data["depth1"]).float().to(device)
    camera1 = data["cameras1"].to(device)

    # viewpoint 2
    rgb2 = torch.tensor(data["rgb2"]).float().to(device)
    mask2 = torch.tensor(data["mask2"]).float().to(device)
    depth2 = torch.tensor(data["depth2"]).float().to(device)
    camera2 = data["cameras2"].to(device)
    
    
    points1, colors1 = unproject_depth_image(rgb1, mask1, depth1, camera1)
    points2, colors2 = unproject_depth_image(rgb2, mask2, depth2, camera2)

    # Pixels with depth 0 (no return) unproject to NaN, which poisons the
    # renderer's bounding box. Drop them before building the point clouds.
    points1, colors1 = drop_invalid(points1, colors1)
    points2, colors2 = drop_invalid(points2, colors2)
    
    pc1 = pytorch3d.structures.Pointclouds(points=[points1], features=[colors1])
    pc2 = pytorch3d.structures.Pointclouds(points=[points2], features=[colors2])
    # The point cloud formed by the union of the first 2 point clouds.
    pc3 = pytorch3d.structures.Pointclouds(points=[torch.cat([points1, points2], dim=0)],
        features=[torch.cat([colors1, colors2], dim=0)],
    )
    renderer = get_points_renderer(image_size=image_size, background_color=(1, 1, 1), device=device)
    images1, images2, images3 = [], [], []
    for azim in range(0, 360, 10):
        # The CO3D cameras use a y-down world convention (note R[1,1] ~= -1 in the
        # stored extrinsics), so the turntable camera needs an inverted up vector
        # for the plant to render upright.
        R, T = pytorch3d.renderer.look_at_view_transform(
            dist=6, elev=0, azim=azim, up=((0, -1, 0),), device=device
        )
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        rend1 = renderer(pc1, cameras=cameras)
        rend2 = renderer(pc2, cameras=cameras)
        rend3 = renderer(pc3, cameras=cameras)
        image1 = rend1[0, ..., :3].cpu().numpy()
        image2 = rend2[0, ..., :3].cpu().numpy()
        image3 = rend3[0, ..., :3].cpu().numpy()
        image1 = (image1.clip(0, 1) * 255).astype(np.uint8)
        image2 = (image2.clip(0, 1) * 255).astype(np.uint8)
        image3 = (image3.clip(0, 1) * 255).astype(np.uint8)
        images1.append(image1)
        images2.append(image2)
        images3.append(image3)
    return images1, images2, images3

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render",
        type=str,
        default="point_cloud",
        choices=["point_cloud", "parametric", "implicit", "plant_rgbd", "torus_parametric", "mobius_parametric", "torus_implicit", "gyroid_implicit"],
    )
    parser.add_argument("--output_path", type=str, default="images/bridge.jpg")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    if args.render == "point_cloud":
        image = render_bridge(image_size=args.image_size, device=device)
        plt.imsave(args.output_path, image)
    elif args.render == "parametric":
        image = render_sphere(image_size=args.image_size, num_samples=args.num_samples, device=device)
        plt.imsave(args.output_path, image)
    elif args.render == "implicit":
        image = render_sphere_mesh(image_size=args.image_size, device=device)
        plt.imsave(args.output_path, image)
    elif args.render == "plant_rgbd": 
        images1, images2, images3 = render_plant_rgbd(image_size=args.image_size, device=device)
        imageio.mimsave("results/rgbd1.gif", images1, duration=0.1, loop=0)  # Save the images as a GIF
        imageio.mimsave("results/rgbd2.gif", images2, duration=0.1, loop=0)  # Save the images as a GIF
        imageio.mimsave("results/rgbd3.gif", images3, duration=0.1, loop=0)  # Save the images as a GIF
    elif args.render == "torus_parametric":
        image = render_torus_parametric(image_size=args.image_size, device=device)
        imageio.mimsave("results/torus.gif", image, duration=0.1, loop=0)
    elif args.render == "mobius_parametric":
        images = render_mobius_parametric(image_size=args.image_size, device=device)
        imageio.mimsave("results/mobius.gif", images, duration=0.1, loop=0)
    elif args.render == "torus_implicit":
        images = render_torus_implicit_mesh(image_size=args.image_size, device=device)
        imageio.mimsave("results/torus_implicit.gif", images, duration=0.1, loop=0)
    elif args.render == "gyroid_implicit":
        images = render_gyroid_implicit_mesh(image_size=args.image_size, device=device)
        imageio.mimsave("results/gyroid_implicit.gif", images, duration=0.1, loop=0)
    else:
        raise Exception("Did not understand {}".format(args.render))
    

