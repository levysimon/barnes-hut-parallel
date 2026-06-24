// Barnes-Hut 2D Galaxy Simulator - Optimized C++ Implementation
// Based on the Python original by Jonas Latt, University of Geneva
//
// Parallelized with TBB parallel_for (Apple Silicon / Apple Clang compatible)
//
// Compile with TBB (recommended):
// clang++ -O3 -std=c++17 \
//    -mcpu=apple-m2 \
//    -fomit-frame-pointer \
//    -fno-math-errno -fno-trapping-math -ffinite-math-only \
//    -fno-signed-zeros -ffp-contract=fast -funroll-loops \
//    -I$(brew --prefix tbb)/include -L$(brew --prefix tbb)/lib -ltbb \
//    barneshut_2d.cpp -o barneshut
//
// Without TBB (serial fallback):
//   g++ -O3 -march=native -std=c++17 -DNO_PARALLEL \
//       -fno-math-errno -fno-trapping-math -ffinite-math-only \
//       -fno-signed-zeros -ffp-contract=fast -funroll-loops \
//       -o barneshut barneshut_2d.cpp

// Video with: ffmpeg -y -r 30 -pattern_type glob -i 'bodies_2d/*.ppm' -vcodec libx264 -crf 18 -pix_fmt yuv420p bodies2d.mp4

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#ifndef NO_PARALLEL
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#endif

namespace cfg {
    // --- Barnes-Hut accuracy ---
    inline constexpr double theta         = 0.5;
    inline constexpr double theta2        = theta * theta;
    // --- Body parameters ---
    inline constexpr double mass          = 1.0;
    inline constexpr double ini_radius    = 0.18;
  
    inline constexpr double inivel        = 0.9;
    inline constexpr double G             = 4.e-6;

    inline constexpr double dt            = 5e-4;
    inline constexpr int    numbodies     = 50000;
    inline constexpr int    max_iter      = 10000;
    inline constexpr int    img_iter      = 10;

    inline constexpr double cutoff_dist   = 0.005;
    inline constexpr double eps2          = cutoff_dist * cutoff_dist;

    inline constexpr double smallest_quad = 5.e-5;
    inline constexpr int    img_width     = 800;
    inline constexpr int    img_height    = 800;
}

struct Vec2 {
    double x{0.}, y{0.};

    Vec2() = default;
    constexpr Vec2(double x_, double y_) noexcept : x(x_), y(y_) {}

    Vec2  operator+ (const Vec2& o) const noexcept { return {x+o.x, y+o.y}; }
    Vec2  operator- (const Vec2& o) const noexcept { return {x-o.x, y-o.y}; }
    Vec2  operator* (double s)      const noexcept { return {x*s,   y*s  }; }
    Vec2& operator+=(const Vec2& o)       noexcept { x+=o.x; y+=o.y; return *this; }
    double norm2()                  const noexcept { return x*x + y*y; }
    double norm()                   const noexcept { return std::sqrt(x*x + y*y); }
};
inline Vec2 operator*(double s, const Vec2& v) noexcept { return v * s; }

struct Node {
    double m{};
    Vec2   m_pos{};
    Vec2   momentum{};
    double s{};
    double s2{};
    Vec2   relpos{};

    std::array<Node*, 4> child{nullptr, nullptr, nullptr, nullptr};

    Node(double m_, double x, double y)
        : m(m_), m_pos(m_ * Vec2{x, y}) {}

    Vec2 pos() const noexcept {
        const double inv_m = 1.0 / m;
        return {m_pos.x * inv_m, m_pos.y * inv_m};
    }

    double dist2(const Node& other) const noexcept {
        const Vec2   p  = pos();
        const Vec2   op = other.pos();
        const double dx = op.x - p.x;
        const double dy = op.y - p.y;
        return dx*dx + dy*dy;
    }

    Vec2 force_on(const Node& other) const noexcept {
        const Vec2   p     = pos();
        const Vec2   op    = other.pos();
        const double dx    = p.x - op.x;
        const double dy    = p.y - op.y;
        const double d2    = dx*dx + dy*dy;
        const double d2e   = d2 + cfg::eps2;
        const double inv_sqrt = 1.0 / std::sqrt(d2e);
        const double inv_d3 = inv_sqrt * inv_sqrt * inv_sqrt;
        const double coeff = (m * other.m) * inv_d3;

        return {dx * coeff, dy * coeff};
    }

    void reset_to_0th_quadrant() noexcept {
        s      = 1.0;
        s2     = 1.0;
        relpos = pos();
    }

    int subdivide(int i) noexcept {
        double& r = (i == 0) ? relpos.x : relpos.y;
        r *= 2.0;
        if (r < 1.0) return 0;
        r -= 1.0;
        return 1;
    }

    int into_next_quadrant() noexcept {
        s  *= 0.5;
        s2  = s * s;
        return subdivide(1) + 2 * subdivide(0);
    }

    bool is_leaf() const noexcept {
        return child[0] == nullptr && child[1] == nullptr
            && child[2] == nullptr && child[3] == nullptr;
    }
};


class NodePool {
public:
    explicit NodePool(std::size_t) {}

    void reset() noexcept { pool_.clear(); }

    Node* alloc(double m, double x, double y) {
        pool_.emplace_back(m, x, y);
        return &pool_.back();
    }

    Node* clone(const Node& src) {
        pool_.push_back(src);
        return &pool_.back();
    }

private:
    std::deque<Node> pool_;
};

Node* add(Node* body, Node* node, NodePool& pool) {
    if (__builtin_expect(node == nullptr, 0)) return body;

    if (node->s <= cfg::smallest_quad) [[unlikely]] return node;

    Node* new_node = nullptr;

    if (node->is_leaf()) [[unlikely]] {
        new_node = pool.clone(*node);
        new_node->child.fill(nullptr);
        int q = node->into_next_quadrant();
        new_node->child[q] = node;
    } else {
        new_node = node;
    }

    new_node->m     += body->m;
    new_node->m_pos += body->m_pos;

    int q = body->into_next_quadrant();
    new_node->child[q] = add(body, new_node->child[q], pool);
    return new_node;
}

Vec2 force_on(const Node& body, const Node& node, double theta2) noexcept {
    if (node.is_leaf()) [[unlikely]]
        return node.force_on(body);

    const double d2 = node.dist2(body);
    if (node.s2 < d2 * theta2) [[likely]]
        return node.force_on(body);

    Vec2 f{};
    for (const Node* c : node.child)
        if (c) f += force_on(body, *c, theta2);
    return f;
}

void verlet(std::vector<Node*>& bodies, const Node* root,
            double theta2, double G, double dt)
{
#ifdef NO_PARALLEL
    for (Node* body : bodies) {
        Vec2 force = G * force_on(*body, *root, theta2);
        body->momentum += dt * force;
        body->m_pos    += dt * body->momentum;
    }
#else
    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(0, bodies.size(), 1024),
        [&](const tbb::blocked_range<std::size_t>& r) {
            for (std::size_t i = r.begin(); i != r.end(); ++i) {
                Node* body = bodies[i];
                Vec2 force = G * force_on(*body, *root, theta2);
                body->momentum += dt * force;
                body->m_pos    += dt * body->momentum;
            }
        }
    , tbb::simple_partitioner());
#endif
}

void save_ppm(const std::vector<Node*>& bodies, int idx) {
    const int W = cfg::img_width;
    const int H = cfg::img_height;

    std::vector<uint8_t> pixels(W * H * 3, 0);

    for (const Node* b : bodies) {
        Vec2 p  = b->pos();
        int  px = static_cast<int>(std::clamp(p.x, 0., 1.) * (W - 1));
        int  py = static_cast<int>(std::clamp(p.y, 0., 1.) * (H - 1));
        py = H - 1 - py;
        int off = (py * W + px) * 3;
        pixels[off    ] = static_cast<uint8_t>(std::min(255, pixels[off    ] + 200));
        pixels[off + 1] = static_cast<uint8_t>(std::min(255, pixels[off + 1] + 200));
        pixels[off + 2] = static_cast<uint8_t>(std::min(255, pixels[off + 2] + 255));
    }

    char fname[64];
    std::snprintf(fname, sizeof(fname), "bodies_2d/img%06d.ppm", idx);
    std::ofstream ofs(fname, std::ios::binary);
    ofs << "P6\n" << W << " " << H << "\n255\n";
    ofs.write(reinterpret_cast<const char*>(pixels.data()), pixels.size());
}

// Initialise tangential velocities for a stable rotating disk.
void init_momentum_2D(std::vector<Node*>& bodies) {
    const Vec2   center   = {0.5, 0.5};
    const double M_total  = cfg::numbodies * cfg::mass;
    const double v_circ0  = std::sqrt(cfg::G * M_total / cfg::ini_radius);

    for (Node* b : bodies) {
        Vec2   r   = b->pos() - center;
        double rn  = r.norm();
        if (rn < 1e-9) { b->momentum = {}; continue; }
        // Scale v_circ by sqrt(r/R) for a rising rotation curve inside the disk
        double v   = cfg::inivel * v_circ0 * std::sqrt(rn / cfg::ini_radius);
        // Tangential direction: perpendicular to r, counter-clockwise
        Vec2   hat = {-r.y / rn, r.x / rn};
        b->momentum = hat * (cfg::mass * v);
    }
}

int main() {
    std::mt19937_64 rng(1);
    std::uniform_real_distribution<double> dist01(0.0, 1.0);

    const double lo   = 0.5 - cfg::ini_radius;
    const double side = 2.0 * cfg::ini_radius;
    const double r2   = cfg::ini_radius * cfg::ini_radius;

    NodePool pool(0);

    std::vector<Node*> bodies;
    bodies.reserve(cfg::numbodies);

    while (bodies.size() < static_cast<std::size_t>(cfg::numbodies)) {
        double px = dist01(rng) * side + lo;
        double py = dist01(rng) * side + lo;
        const double dx = px - 0.5, dy = py - 0.5;
        if (dx*dx + dy*dy < r2)
            bodies.push_back(pool.alloc(cfg::mass, px, py));
    }

    init_momentum_2D(bodies);

    {
        Vec2 p_sum{};
        for (const Node* b : bodies) p_sum += b->momentum;
        const Vec2 p_mean = p_sum * (1.0 / static_cast<double>(bodies.size()));
        for (Node* b : bodies) b->momentum += p_mean * -1.0;
    }

    {
        Vec2 com{};
        for (const Node* b : bodies) com += b->pos();
        const Vec2 offset = Vec2{0.5, 0.5} - com * (1.0 / static_cast<double>(bodies.size()));
        for (Node* b : bodies) {
            b->m_pos.x += offset.x * b->m;
            b->m_pos.y += offset.y * b->m;
        }
    }

    std::filesystem::create_directories("bodies_2d");

    std::cout << "Barnes-Hut 2D — " << bodies.size()
              << " bodies, " << cfg::max_iter << " iterations\n";
    std::cout << "v_circ = "
              << std::sqrt(cfg::G * cfg::numbodies * cfg::mass / cfg::ini_radius)
              << "  inivel factor = " << cfg::inivel << "\n";

    struct BodyState {
        double m;
        Vec2   m_pos;
        Vec2   momentum;
    };
    std::vector<BodyState> snapshot;
    snapshot.reserve(cfg::numbodies);

    for (int i = 0; i < cfg::max_iter; ++i) {

        snapshot.clear();
        snapshot.resize(bodies.size());
        for (std::size_t k = 0; k < bodies.size(); ++k)
            snapshot[k] = {bodies[k]->m, bodies[k]->m_pos, bodies[k]->momentum};

        pool.reset();
        bodies.clear();

        constexpr double margin = 1e-6;
        for (const auto& s : snapshot) {
            const double inv_m = 1.0 / s.m;
            const double px    = s.m_pos.x * inv_m;
            const double py    = s.m_pos.y * inv_m;
            if (__builtin_expect(px < -margin || px > 1.0 + margin ||
                                 py < -margin || py > 1.0 + margin, 0))
                continue;
            Node* n     = pool.alloc(s.m, px, py);
            n->momentum = s.momentum;
            bodies.push_back(n);
        }

        Node* root = nullptr;
        for (Node* b : bodies) {
            b->reset_to_0th_quadrant();
            root = add(b, root, pool);
        }

        verlet(bodies, root, cfg::theta2, cfg::G, cfg::dt);

        if (i % cfg::img_iter == 0) {
            std::cout << "iter " << i << "  bodies=" << bodies.size() << "\n";
            save_ppm(bodies, i);
        }
    }

    std::cout << "Done.\n";
    return 0;
}