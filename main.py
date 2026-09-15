"""
16-825 Assignment 1: Rendering Basics with PyTorch3D
Yi-Ning Huang (yiningh3)

Runs every question in the assignment and writes all results to results/.

Usage:
    python main.py                 # run everything
    python main.py -q 1.1 3 5.2    # run only the listed questions
    python main.py --list          # show available questions
    python main.py --device cpu    # force CPU (default: GPU if available)

Outputs (all under results/):
    1.1  cow_360.gif
    1.2  dolly.gif
    2.1  tetrahedron_360.gif
    2.2  cube_360.gif
    3    cow_retexture_360.gif
    4    textured_cow_t1.jpg ... textured_cow_t4.jpg
    5.1  rgbd1.gif, rgbd2.gif, rgbd3.gif
    5.2  torus.gif, mobius.gif
    5.3  torus_implicit.gif, gyroid_implicit.gif
    6    metaball_morph.gif
    7    sample_10.gif, sample_100.gif, sample_1000.gif, sample_10000.gif
"""
import argparse
import os
import time

import imageio
import matplotlib

matplotlib.use("Agg")  # no display on the cluster
import matplotlib.pyplot as plt
import torch

RESULTS_DIR = "results"
FPS = 15
DURATION = 1000 // FPS  # ms per frame


def save_gif(frames, name, duration=DURATION):
    path = os.path.join(RESULTS_DIR, name)
    imageio.mimsave(path, frames, duration=duration, loop=0)
    return path


def save_image(image, name):
    path = os.path.join(RESULTS_DIR, name)
    plt.imsave(path, image)
    return path


# --------------------------------------------------------------------------
# Question 1: Practicing with Cameras
# --------------------------------------------------------------------------
def q1_1(device):
    """1.1 360-degree render of the cow mesh."""
    from part1_1 import render_cow_360degrees

    frames = render_cow_360degrees(device=device)
    return [save_gif(frames, "cow_360.gif")]


def q1_2(device):
    """1.2 Dolly zoom."""
    from starter.dolly_zoom import dolly_zoom

    path = os.path.join(RESULTS_DIR, "dolly.gif")
    # dolly_zoom writes the gif itself; 20 frames matches the submitted result.
    dolly_zoom(image_size=256, num_frames=20, duration=3, device=device,
               output_file=path)
    return [path]


# --------------------------------------------------------------------------
# Question 2: Practicing with Meshes
# --------------------------------------------------------------------------
def q2_1(device):
    """2.1 Tetrahedron (4 vertices, 4 triangle faces)."""
    from part2 import render_360degree_tetrahedron

    frames = render_360degree_tetrahedron(device=device)
    return [save_gif(frames, "tetrahedron_360.gif")]


def q2_2(device):
    """2.2 Cube (8 vertices, 12 triangle faces)."""
    from part2 import render_360degree_cube

    frames = render_360degree_cube(device=device)
    return [save_gif(frames, "cube_360.gif")]


# --------------------------------------------------------------------------
# Question 3: Re-texturing a Mesh
# --------------------------------------------------------------------------
def q3(device):
    """3. Cow re-textured with a blue-to-red gradient along z."""
    from part3 import render_cow_360degrees_retexture

    frames = render_cow_360degrees_retexture(device=device)
    return [save_gif(frames, "cow_retexture_360.gif")]


# --------------------------------------------------------------------------
# Question 4: Camera Transformations
# --------------------------------------------------------------------------
def q4(device):
    """4. Four relative camera transformations."""
    from part4 import render_textured_cow

    transforms = [
        # (name, R_relative, T_relative)
        ("textured_cow_t1.jpg",  # 90 deg roll about the optical axis
         [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], [0, 0, 0]),
        ("textured_cow_t2.jpg",  # dolly back 2 units
         [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [0, 0, 2]),
        ("textured_cow_t3.jpg",  # shift parallel to the image plane
         [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [0.5, -0.5, 0]),
        ("textured_cow_t4.jpg",  # 90 deg orbit to a side profile
         [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], [-3.0, 0.0, 3.0]),
    ]

    written = []
    for name, R_relative, T_relative in transforms:
        image = render_textured_cow(
            R_relative=R_relative, T_relative=T_relative, device=device
        )
        written.append(save_image(image, name))
    return written


# --------------------------------------------------------------------------
# Question 5: Rendering Generic 3D Representations
# --------------------------------------------------------------------------
def q5_1(device):
    """5.1 Point clouds unprojected from two RGB-D plant images."""
    from part5 import render_plant_rgbd

    images1, images2, images3 = render_plant_rgbd(device=device)
    return [
        save_gif(images1, "rgbd1.gif", duration=100),
        save_gif(images2, "rgbd2.gif", duration=100),
        save_gif(images3, "rgbd3.gif", duration=100),
    ]


def q5_2(device):
    """5.2 Torus point cloud plus a Mobius strip."""
    from part5 import render_torus_parametric, render_mobius_parametric

    written = [save_gif(render_torus_parametric(device=device), "torus.gif",
                        duration=100)]
    written.append(save_gif(render_mobius_parametric(device=device), "mobius.gif",
                            duration=100))
    return written


def q5_3(device):
    """5.3 Torus mesh from an implicit function, plus a gyroid."""
    from part5 import render_torus_implicit_mesh, render_gyroid_implicit_mesh

    written = [save_gif(render_torus_implicit_mesh(device=device),
                        "torus_implicit.gif", duration=100)]
    written.append(save_gif(render_gyroid_implicit_mesh(device=device),
                            "gyroid_implicit.gif", duration=100))
    return written


# --------------------------------------------------------------------------
# Question 6 (extra credit): Do Something Fun
# --------------------------------------------------------------------------
def q6(device):
    """6. Metaball morph with a topology change (extra credit)."""
    from part6 import render_metaball_morph, CRITICAL_SEP

    print(f"    critical separation r*sqrt(2) = {CRITICAL_SEP:.4f}")
    frames = render_metaball_morph(device=device)
    return [save_gif(frames, "metaball_morph.gif", duration=1000 // 20)]


# --------------------------------------------------------------------------
# Question 7 (extra credit): Sampling Points on Meshes
# --------------------------------------------------------------------------
def q7(device):
    """7. Stratified surface sampling at 10/100/1000/10000 points (extra credit)."""
    from part7 import render_sampled_vs_mesh

    generator = torch.Generator(device=device).manual_seed(0)
    written = []
    for num_samples in [10, 100, 1000, 10000]:
        frames = render_sampled_vs_mesh(
            num_samples, device=device, generator=generator
        )
        written.append(save_gif(frames, f"sample_{num_samples}.gif"))
    return written


# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------
QUESTIONS = {
    "1.1": q1_1,
    "1.2": q1_2,
    "2.1": q2_1,
    "2.2": q2_2,
    "3": q3,
    "4": q4,
    "5.1": q5_1,
    "5.2": q5_2,
    "5.3": q5_3,
    "6": q6,
    "7": q7,
}


def resolve(names):
    """Expands shorthand like '1' or '5' into their sub-questions."""
    if not names:
        return list(QUESTIONS)

    selected = []
    for name in names:
        if name in QUESTIONS:
            selected.append(name)
            continue
        # Allow "1" to mean 1.1 and 1.2, "5" to mean 5.1, 5.2 and 5.3.
        matches = [q for q in QUESTIONS if q.startswith(name + ".")]
        if not matches:
            raise SystemExit(
                f"unknown question '{name}'. Use --list to see the options."
            )
        selected.extend(matches)

    # De-duplicate while keeping the canonical ordering.
    return [q for q in QUESTIONS if q in set(selected)]


def main():
    parser = argparse.ArgumentParser(
        description="Run the results for 16-825 Assignment 1.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-q", "--questions", nargs="+", metavar="Q", default=None,
        help="questions to run, e.g. '1.1 3 5.2'. A bare '1' runs 1.1 and 1.2. "
             "Default: run everything.",
    )
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], default=None,
        help="device to render on. Default: cuda if available, else cpu.",
    )
    parser.add_argument("--list", action="store_true", help="list the questions and exit")
    args = parser.parse_args()

    if args.list:
        print("Available questions:")
        for name, fn in QUESTIONS.items():
            summary = (fn.__doc__ or "").strip().splitlines()[0]
            print(f"  {name:4s}  {summary}")
        return

    if args.device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device(args.device)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    selected = resolve(args.questions)

    print(f"Device: {device}")
    print(f"Running: {', '.join(selected)}\n")

    all_written = []
    for name in selected:
        fn = QUESTIONS[name]
        summary = (fn.__doc__ or "").strip().splitlines()[0]
        print(f"[{name}] {summary}")
        start = time.time()
        written = fn(device)
        elapsed = time.time() - start
        for path in written:
            print(f"    wrote {path}")
        print(f"    done in {elapsed:.1f}s\n")
        all_written.extend(written)

    print(f"Finished. {len(all_written)} file(s) written to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
