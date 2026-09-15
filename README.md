# 16-825 Assignment 1: Rendering Basics with PyTorch3D

**Name:** Yi-Ning Huang
**Andrew ID:** yiningh3

check `handin.md` for the description of the assignment and the questions.

Report webpage: http://www.andrew.cmu.edu/course/16-825/projects/yiningh3/proj1/

---

## Setup

```bash
conda create -n learning3d python=3.10
conda activate learning3d
pip install torch                       # or the CUDA build for your machine
pip install fvcore iopath
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
pip install -r requirements.txt
```

Everything runs on CPU if no GPU is present. The results submitted here were
produced with Python 3.10, PyTorch 2.14 (CUDA 13.0) and PyTorch3D 0.7.8.

---

## Running the code

`main.py` runs every question and writes all outputs to `results/`.

```bash
python main.py                  # run everything
python main.py --list           # list the available questions
python main.py -q 1.1 3 5.2     # run only the listed questions
python main.py -q 5             # a bare "5" expands to 5.1, 5.2 and 5.3
python main.py --device cpu     # force CPU (default: CUDA when available)
```

The `results/` directory is created automatically. A full run takes roughly a
minute on a GPU.

---

## Outputs

| Question | Output file(s) in `results/` |
|---|---|
| 1.1 360-degree render | `cow_360.gif` |
| 1.2 Dolly zoom | `dolly.gif` |
| 2.1 Tetrahedron | `tetrahedron_360.gif` |
| 2.2 Cube | `cube_360.gif` |
| 3 Re-texturing | `cow_retexture_360.gif` |
| 4 Camera transforms | `textured_cow_t1.jpg` ... `textured_cow_t4.jpg` |
| 5.1 RGB-D point clouds | `rgbd1.gif`, `rgbd2.gif`, `rgbd3.gif` |
| 5.2 Parametric surfaces | `torus.gif`, `mobius.gif` |
| 5.3 Implicit surfaces | `torus_implicit.gif`, `gyroid_implicit.gif` |
| 6 Do something fun (EC) | `metaball_morph.gif` |
| 7 Surface sampling (EC) | `sample_10.gif`, `sample_100.gif`, `sample_1000.gif`, `sample_10000.gif` |

---

## Source files

| File | Contents |
|---|---|
| `main.py` | Entry point. Runs any or all questions. |
| `part1_1.py` | Q1.1 turntable render of the cow. |
| `part2.py` | Q2.1 tetrahedron and Q2.2 cube, built by hand. |
| `part3.py` | Q3 per-vertex color interpolation along z. |
| `part4.py` | Q4 the four relative camera transformations. |
| `part5.py` | Q5.1 RGB-D unprojection, Q5.2 parametric surfaces, Q5.3 implicit surfaces. |
| `part6.py` | Q6 metaball morph (extra credit). |
| `part7.py` | Q7 stratified surface sampling (extra credit). |
| `starter/utils.py` | Course-provided renderer helpers and unprojection. |
| `starter/dolly_zoom.py` | Q1.2 dolly zoom (course starter, completed). |
| `data/` | Meshes, textures and the RGB-D pickle. |

`part5.py` also keeps its own CLI from the starter code, e.g.
`python -m part5 --render plant_rgbd`, but `main.py` is the intended entry point.
