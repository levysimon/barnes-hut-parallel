# Barnes-Hut 2D Galaxy Simulator

A fast N-body gravity simulator in C++17 simulating galaxy formation and collisions using the Barnes-Hut algorithm. Parallelized with Intel TBB on Apple Silicon.

![Galaxy collision](galaxy.gif)

------------------------------------------------------------------------

## The N-body Problem

Simulating a galaxy means tracking the gravitational interaction of tens of thousands of bodies. The naive approach computes the force between every pair — O(N²) per timestep. At 50,000 bodies that's 2.5 billion pair evaluations every step, which is hopelessly slow.

## The Barnes-Hut Algorithm

Barnes-Hut reduces this to **O(N log N)** by exploiting a simple physical observation: a distant cluster of bodies can be approximated as a single mass at its center of mass, as long as the cluster is small enough relative to the distance.

Space is recursively subdivided into a **quadtree**. When computing the force on a body, if a node's cell satisfies `s / d < θ` (cell width over distance), the whole subtree is treated as one mass. Otherwise the traversal recurses into its children. Smaller θ means higher accuracy at the cost of more work. We use `θ = 0.5`.

> Barnes, J., Hut, P. *A hierarchical O(N log N) force-calculation algorithm.* Nature 324, 446–449 (1986). <https://doi.org/10.1038/324446a0>

## Simulations

**`barneshut_2d.cpp`** — 50,000 bodies forming a single rotating disk. Good starting point for reading the code.

**`barneshut_collision.cpp`** — three galaxies of 3,000 bodies each placed at the vertices of an equilateral triangle, converging at \~50% of escape velocity. Bodies are color-coded by galaxy (blue, orange, green) so you can track mixing during the merger.

## Current Bottleneck: Tree Construction

Force computation is parallelized with TBB and scales well across cores. The **bottleneck is now the quadtree build**, which runs serially every timestep — each insertion modifies shared tree structure so it cannot be parallelized naively. At 50,000 bodies this is a significant fraction of wall time. Possible directions to address it:

-   **Morton-code sort + bottom-up parallel build** — the approach used in GPU N-body codes
-   **Reduce rebuild frequency** — reuse the tree for k steps, drifting positions between rebuilds
-   **Task-parallel top-down construction** — build each quadrant subtree on a separate thread

## Build

**With TBB (recommended):**

``` bash
clang++ -O3 -std=c++17 \
  -mcpu=apple-m2 \
  -fomit-frame-pointer \
  -fno-math-errno -fno-trapping-math -ffinite-math-only \
  -fno-signed-zeros -ffp-contract=fast -funroll-loops \
  -I$(brew --prefix tbb)/include -L$(brew --prefix tbb)/lib -ltbb \
  barneshut_2d.cpp -o barneshut
```

**Without TBB (serial fallback):**

``` bash
g++ -O3 -march=native -std=c++17 -DNO_PARALLEL \
    -fno-math-errno -fno-trapping-math -ffinite-math-only \
    -fno-signed-zeros -ffp-contract=fast -funroll-loops \
    -o barneshut barneshut_2d.cpp
```

## Run

``` bash
./barneshut
```

Frames are written as PPM images to `bodies_2d/`. Convert to video with:

``` bash
ffmpeg -y -r 30 -pattern_type glob -i 'bodies_2d/*.ppm' \
  -vcodec libx264 -crf 18 -pix_fmt yuv420p bodies2d.mp4
```

## Parameters

All parameters are in the `cfg` namespace at the top of each file.

| Parameter    | Single galaxy | Collision | Description            |
|--------------|---------------|-----------|------------------------|
| `numbodies`  | 50,000        | 9,000     | Total body count       |
| `max_iter`   | 10,000        | 20,000    | Total timesteps        |
| `dt`         | 5e-4          | 5e-4      | Timestep size          |
| `theta`      | 0.5           | 0.5       | Barnes-Hut accuracy    |
| `ini_radius` | 0.18          | 0.11      | Disk radius            |
| `G`          | 4e-6          | 4e-6      | Gravitational constant |

Have fun with theses parameteres ! 

## Credits

Based on the Python implementation by Jonas Latt, University of Geneva.
