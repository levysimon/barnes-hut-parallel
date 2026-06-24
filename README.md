# Barnes-Hut 2D Galaxy Simulator

A fast N-body gravity simulator written in C++17, using the Barnes-Hut algorithm to simulate the formation and rotation of a 2D galaxy disk. Parallelized with Intel TBB on Apple Silicon.

![Galaxy simulation](galaxy.gif)

---

## The N-body Problem

Simulating a galaxy means tracking the gravitational interaction of tens of thousands of bodies. The naive approach computes the force between every pair of bodies — O(N²) work per timestep. At 50,000 bodies that is 2.5 billion pair evaluations every step, which is hopelessly slow.

## The Barnes-Hut Algorithm

Barnes-Hut reduces this to **O(N log N)** by exploiting a simple physical observation: a distant cluster of bodies can be approximated as a single body at the cluster's center of mass, as long as the cluster is small enough relative to the distance.

The algorithm works by recursively subdividing space into a **quadtree** (in 2D) or octree (in 3D). Each node of the tree stores the total mass and center of mass of all bodies within its cell. When computing the force on a body, the tree is traversed from the root: if a node's cell is far enough away (controlled by the parameter θ), the whole subtree is approximated as a single mass. Otherwise the traversal recurses into the node's children.

The accuracy criterion is:

```
s / d < θ
```

where `s` is the cell width and `d` is the distance to the cell's center of mass. Smaller θ means higher accuracy at the cost of more computation. This implementation uses `θ = 0.5`.

The original paper can be found here : Barnes, J., Hut, P. A hierarchical O(N log N) force-calculation algorithm. Nature 324, 446–449 (1986). https://doi.org/10.1038/324446a0

## Current Performance Bottleneck: Tree Construction

The force computation is parallelized with TBB and scales well across cores. The **bottleneck is now the quadtree build**, which is done serially every timestep.

Every iteration, all body states are snapshotted, the node pool is reset, and the tree is rebuilt from scratch by inserting bodies one at a time into the root. This insertion process is inherently sequential — each insertion modifies shared tree structure, so it cannot be parallelized naively.

At 50,000 bodies the tree build takes a significant fraction of total wall time, and its cost grows as O(N log N) with the body count. Possible directions to address this:

- **Parallel tree build** — sort bodies by their Z-order (Morton) curve index, then construct the tree bottom-up in parallel. This is the approach used in GPU N-body codes.
- **Reduce rebuild frequency** — rebuild the tree every k steps and drift body positions between rebuilds (only valid if bodies don't move far).
- **Task-parallel top-down construction** — split the root into quadrants and build each subtree on a separate thread, with synchronization only at the top few levels.

## Build

**With TBB (recommended, parallelized):**
```bash
clang++ -O3 -std=c++17 \
  -mcpu=apple-m2 \
  -fomit-frame-pointer \
  -fno-math-errno -fno-trapping-math -ffinite-math-only \
  -fno-signed-zeros -ffp-contract=fast -funroll-loops \
  -I$(brew --prefix tbb)/include -L$(brew --prefix tbb)/lib -ltbb \
  barneshut_2d.cpp -o barneshut
```

**Without TBB (serial fallback):**
```bash
g++ -O3 -march=native -std=c++17 -DNO_PARALLEL \
    -fno-math-errno -fno-trapping-math -ffinite-math-only \
    -fno-signed-zeros -ffp-contract=fast -funroll-loops \
    -o barneshut barneshut_2d.cpp
```

## Run

```bash
./barneshut
```

Frames are written as PPM images to `bodies_2d/`. Convert to video with:

```bash
ffmpeg -y -r 30 -pattern_type glob -i 'bodies_2d/*.ppm' \
  -vcodec libx264 -crf 18 -pix_fmt yuv420p bodies2d.mp4
```

## Parameters

All simulation parameters are in the `cfg` namespace at the top of the file:

| Parameter | Default | Description |
|---|---|---|
| `numbodies` | 50,000 | Number of bodies |
| `max_iter` | 10,000 | Total timesteps |
| `dt` | 5e-4 | Timestep size |
| `theta` | 0.5 | Barnes-Hut accuracy (lower = more accurate) |
| `ini_radius` | 0.18 | Initial disk radius |
| `inivel` | 0.9 | Circular velocity scaling factor |
| `G` | 4e-6 | Gravitational constant |
| `img_iter` | 10 | Save a frame every N iterations |

## Credits

Based on the Python implementation by Jonas Latt, University of Geneva.
