// Barnes-Hut 2D Galaxy Collision Simulator
// Based on the Python original by Jonas Latt, University of Geneva
// Parallelized with TBB parallel_for (Apple Silicon / Apple Clang compatible)
//
// Compile with TBB:
//clang++ -O3 -std=c++17 \
//    -mcpu=apple-m2 \
//    -fomit-frame-pointer \
//    -fno-math-errno -fno-trapping-math -ffinite-math-only \
//    -fno-signed-zeros -ffp-contract=fast -funroll-loops \
//    -I$(brew --prefix tbb)/include -L$(brew --prefix tbb)/lib -ltbb \
//    barneshut_collision.cpp -o barneshut
//
// Video: ffmpeg -y -r 30 -pattern_type glob -i 'bodies_2d/*.ppm' \
//        -vcodec libx264 -crf 18 -pix_fmt yuv420p bodies2d.mp4

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
#  include <tbb/parallel_for.h>
#  include <tbb/blocked_range.h>
#endif

namespace cfg {
    inline constexpr double theta         = 0.5;
    inline constexpr double theta2        = theta * theta;
    inline constexpr double mass          = 1.0;

    // Three galaxies in an equilateral triangle converging on (0.5, 0.5).
    // v_approach = 0.20 ~ 50% of v_escape -> all three stay bound.
    inline constexpr int    bodies_per_galaxy = 3000;
    inline constexpr int    numbodies         = bodies_per_galaxy * 3;
    inline constexpr double ini_radius        = 0.11;

    inline constexpr double inivel        = 0.88;
    inline constexpr double G             = 4.e-6;
    inline constexpr double dt            = 5e-4;

    inline constexpr int    max_iter      = 20000;
    inline constexpr int    img_iter      = 10;

    inline constexpr double cutoff_dist   = 0.006;
    inline constexpr double eps2          = cutoff_dist * cutoff_dist;

    inline constexpr double smallest_quad = 5.e-5;
    inline constexpr int    img_width     = 1800;
    inline constexpr int    img_height    = 1800;
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
    int    galaxy_id{0};   // 0 = galaxy A (blue), 1 = galaxy B (orange)

    std::array<Node*, 4> child{nullptr, nullptr, nullptr, nullptr};

    Node(double m_, double x, double y, int gid)
        : m(m_), m_pos(m_ * Vec2{x, y}), galaxy_id(gid) {}

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
        const double denom = d2e * std::sqrt(d2e);
        const double coeff = (m * other.m) / denom;
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

    Node* alloc(double m, double x, double y, int gid) {
        pool_.emplace_back(m, x, y, gid);
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
        tbb::blocked_range<std::size_t>(0, bodies.size(), 32),
        [&](const tbb::blocked_range<std::size_t>& r) {
            for (std::size_t i = r.begin(); i != r.end(); ++i) {
                Node* body = bodies[i];
                Vec2 force = G * force_on(*body, *root, theta2);
                body->momentum += dt * force;
                body->m_pos    += dt * body->momentum;
            }
        }
    );
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

        if (b->galaxy_id == 0) {
            // Galaxy A: cold blue-white
            pixels[off    ] = static_cast<uint8_t>(std::min(255, pixels[off    ] + 150));
            pixels[off + 1] = static_cast<uint8_t>(std::min(255, pixels[off + 1] + 200));
            pixels[off + 2] = static_cast<uint8_t>(std::min(255, pixels[off + 2] + 255));
        } else if (b->galaxy_id == 1) {
            // Galaxy B: warm orange
            pixels[off    ] = static_cast<uint8_t>(std::min(255, pixels[off    ] + 255));
            pixels[off + 1] = static_cast<uint8_t>(std::min(255, pixels[off + 1] + 160));
            pixels[off + 2] = static_cast<uint8_t>(std::min(255, pixels[off + 2] +  40));
        } else {
            // Galaxy C: green
            pixels[off    ] = static_cast<uint8_t>(std::min(255, pixels[off    ] +  80));
            pixels[off + 1] = static_cast<uint8_t>(std::min(255, pixels[off + 1] + 255));
            pixels[off + 2] = static_cast<uint8_t>(std::min(255, pixels[off + 2] + 120));
        }
    }

    char fname[64];
    std::snprintf(fname, sizeof(fname), "bodies_2d/img%04d.ppm", idx);
    std::ofstream ofs(fname, std::ios::binary);
    ofs << "P6\n" << W << " " << H << "\n255\n";
    ofs.write(reinterpret_cast<const char*>(pixels.data()), pixels.size());
}

// Populate one galaxy disk centred at (cx, cy) with N bodies,
// internal CCW rotation, and bulk velocity (vx, vy).
void init_galaxy(std::vector<Node*>& bodies, NodePool& pool,
                 std::mt19937_64& rng,
                 double cx, double cy,
                 double vx, double vy,
                 int gid)
{
    std::uniform_real_distribution<double> dist01(0.0, 1.0);
    const double R    = cfg::ini_radius;
    const double r2   = R * R;
    const double lo   = -R;
    const double side = 2.0 * R;
    const double M    = cfg::bodies_per_galaxy * cfg::mass;
    const double v_circ0 = std::sqrt(cfg::G * M / R);

    std::vector<Node*> disk;
    disk.reserve(cfg::bodies_per_galaxy);

    while (static_cast<int>(disk.size()) < cfg::bodies_per_galaxy) {
        double lx = dist01(rng) * side + lo;
        double ly = dist01(rng) * side + lo;
        if (lx*lx + ly*ly < r2) {
            Node* b = pool.alloc(cfg::mass, cx + lx, cy + ly, gid);
            disk.push_back(b);
        }
    }

    // Internal tangential velocity
    for (Node* b : disk) {
        Vec2   p   = b->pos();
        Vec2   r   = {p.x - cx, p.y - cy};
        double rn  = r.norm();
        if (rn < 1e-9) { b->momentum = {cfg::mass * vx, cfg::mass * vy}; continue; }
        double v   = cfg::inivel * v_circ0 * std::sqrt(rn / R);
        Vec2   hat = {-r.y / rn, r.x / rn};
        b->momentum = hat * (cfg::mass * v) + Vec2{cfg::mass * vx, cfg::mass * vy};
    }

    // Zero net momentum of this disk so its CoM moves only with (vx,vy)
    Vec2 p_sum{};
    for (const Node* b : disk) p_sum += b->momentum;
    Vec2 p_bulk = {cfg::mass * vx * cfg::bodies_per_galaxy,
                   cfg::mass * vy * cfg::bodies_per_galaxy};
    Vec2 p_residual = (p_sum - p_bulk) * (1.0 / cfg::bodies_per_galaxy);
    for (Node* b : disk) b->momentum += p_residual * -1.0;

    for (Node* b : disk) bodies.push_back(b);
}

int main() {
    std::mt19937_64 rng(42);

    NodePool pool(0);

    std::vector<Node*> bodies;
    bodies.reserve(cfg::numbodies);

    // Three galaxies placed at the vertices of an equilateral triangle
    // centred on (0.5, 0.5) with circumradius 0.22.
    // Each moves toward the common centre at v=0.20 (~50% of v_escape).
    //
    //        A (top, blue)
    //       / \
    //      B   C
    // (orange) (green)
    //
    // Angles: A=90deg, B=210deg, C=330deg  (measured from +x axis)
    constexpr double tri_r  = 0.273;
    constexpr double v_in   = 0.20;
    constexpr double pi     = 3.14159265358979323846;
    constexpr double a0     = pi / 2.0;
    constexpr double a1     = pi / 2.0 + 2.0 * pi / 3.0;
    constexpr double a2     = pi / 2.0 + 4.0 * pi / 3.0;

    init_galaxy(bodies, pool, rng,
                0.5 + tri_r * std::cos(a0), 0.5 + tri_r * std::sin(a0),
                -v_in * std::cos(a0), -v_in * std::sin(a0), 0);

    init_galaxy(bodies, pool, rng,
                0.5 + tri_r * std::cos(a1), 0.5 + tri_r * std::sin(a1),
                -v_in * std::cos(a1), -v_in * std::sin(a1), 1);

    init_galaxy(bodies, pool, rng,
                0.5 + tri_r * std::cos(a2), 0.5 + tri_r * std::sin(a2),
                -v_in * std::cos(a2), -v_in * std::sin(a2), 2);

    std::filesystem::remove_all("bodies_2d");
    std::filesystem::create_directories("bodies_2d");

    std::cout << "Barnes-Hut 2D triple collision — " << bodies.size()
              << " bodies, " << cfg::max_iter << " iterations\n";

    const double v_esc = std::sqrt(2.0 * cfg::G * cfg::bodies_per_galaxy * 3
                                   * cfg::mass / (tri_r * std::sqrt(3.0)));
    std::cout << "v_in = " << v_in
              << "  v_escape = " << v_esc
              << "  ratio = " << v_in / v_esc << "\n";

    struct BodyState {
        double m;
        Vec2   m_pos;
        Vec2   momentum;
        int    galaxy_id;
    };
    std::vector<BodyState> snapshot;
    snapshot.reserve(cfg::numbodies);

    for (int i = 0; i < cfg::max_iter; ++i) {

        snapshot.clear();
        snapshot.resize(bodies.size());
        for (std::size_t k = 0; k < bodies.size(); ++k)
            snapshot[k] = {bodies[k]->m, bodies[k]->m_pos,
                           bodies[k]->momentum, bodies[k]->galaxy_id};

        pool.reset();
        bodies.clear();

        constexpr double margin = 0.05;
        for (const auto& s : snapshot) {
            const double inv_m = 1.0 / s.m;
            const double px    = s.m_pos.x * inv_m;
            const double py    = s.m_pos.y * inv_m;
            if (__builtin_expect(px < -margin || px > 1.0 + margin ||
                                 py < -margin || py > 1.0 + margin, 0))
                continue;
            Node* n        = pool.alloc(s.m, px, py, s.galaxy_id);
            n->momentum    = s.momentum;
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
