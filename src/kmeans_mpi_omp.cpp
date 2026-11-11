#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

struct Options {
    std::uint64_t n = 200000;
    int d = 32;
    int k = 16;
    int iterations = 15;
    std::uint64_t seed = 5208;
    bool validate = false;
};

static std::uint64_t mix(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static double uniform(std::uint64_t x) {
    return static_cast<double>(mix(x) >> 11) * (1.0 / 9007199254740992.0);
}

static double point(std::uint64_t id, int dimension, int k, std::uint64_t seed) {
    const int cluster = static_cast<int>(id % static_cast<std::uint64_t>(k));
    const double center = dimension == 0 ? 3.0 * cluster : 0.2 * ((cluster + dimension) % 5);
    const std::uint64_t key = seed ^ (id * 0x9e3779b97f4a7c15ULL)
        ^ (static_cast<std::uint64_t>(dimension + 1) * 0xbf58476d1ce4e5b9ULL);
    return center + 2.0 * uniform(key) - 1.0;
}

static Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--validate") { o.validate = true; continue; }
        if (i + 1 >= argc) throw std::runtime_error("missing value for " + arg);
        const std::string value = argv[++i];
        if (arg == "--n") o.n = std::stoull(value);
        else if (arg == "--d") o.d = std::stoi(value);
        else if (arg == "--k") o.k = std::stoi(value);
        else if (arg == "--iterations") o.iterations = std::stoi(value);
        else if (arg == "--seed") o.seed = std::stoull(value);
        else throw std::runtime_error("unknown option " + arg);
    }
    if (o.n == 0 || o.d < 1 || o.k < 1 || o.iterations < 1 || o.n < static_cast<std::uint64_t>(o.k))
        throw std::runtime_error("require n >= k >= 1, d >= 1, iterations >= 1");
    if (static_cast<std::uint64_t>(o.d) * o.k > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
        throw std::runtime_error("k*d exceeds MPI count limit");
    return o;
}

static std::uint64_t block_start(std::uint64_t n, int rank, int ranks) {
    return n / ranks * rank + std::min<std::uint64_t>(rank, n % ranks);
}

static std::uint64_t block_size(std::uint64_t n, int rank, int ranks) {
    return n / ranks + (static_cast<std::uint64_t>(rank) < n % ranks);
}

static int nearest(const double* x, const std::vector<double>& centers, int d, int k) {
    int best = 0;
    double best_distance = std::numeric_limits<double>::infinity();
    for (int c = 0; c < k; ++c) {
        double distance = 0;
        for (int j = 0; j < d; ++j) {
            const double delta = x[j] - centers[static_cast<std::size_t>(c) * d + j];
            distance += delta * delta;
        }
        if (distance < best_distance) { best_distance = distance; best = c; }
    }
    return best;
}

static std::vector<double> serial_reference(const Options& o) {
    std::vector<double> centers(static_cast<std::size_t>(o.k) * o.d);
    for (int c = 0; c < o.k; ++c)
        for (int j = 0; j < o.d; ++j)
            centers[static_cast<std::size_t>(c) * o.d + j] = point(c, j, o.k, o.seed);
    std::vector<double> x(o.d), sums(centers.size());
    std::vector<std::uint64_t> counts(o.k);
    for (int iteration = 0; iteration < o.iterations; ++iteration) {
        std::fill(sums.begin(), sums.end(), 0.0);
        std::fill(counts.begin(), counts.end(), 0);
        for (std::uint64_t id = 0; id < o.n; ++id) {
            for (int j = 0; j < o.d; ++j) x[j] = point(id, j, o.k, o.seed);
            const int c = nearest(x.data(), centers, o.d, o.k);
            ++counts[c];
            for (int j = 0; j < o.d; ++j) sums[static_cast<std::size_t>(c) * o.d + j] += x[j];
        }
        for (int c = 0; c < o.k; ++c)
            if (counts[c])
                for (int j = 0; j < o.d; ++j)
                    centers[static_cast<std::size_t>(c) * o.d + j] = sums[static_cast<std::size_t>(c) * o.d + j] / counts[c];
    }
    return centers;
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 2);

    Options o;
    try { o = parse(argc, argv); }
    catch (const std::exception& e) {
        if (rank == 0) std::cerr << e.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    const int threads = omp_get_max_threads();
    const std::uint64_t start = block_start(o.n, rank, ranks);
    const std::uint64_t local_n = block_size(o.n, rank, ranks);
    std::vector<double> data(static_cast<std::size_t>(local_n) * o.d);
    for (std::uint64_t i = 0; i < local_n; ++i)
        for (int j = 0; j < o.d; ++j)
            data[static_cast<std::size_t>(i) * o.d + j] = point(start + i, j, o.k, o.seed);

    std::vector<double> centers(static_cast<std::size_t>(o.k) * o.d);
    for (int c = 0; c < o.k; ++c)
        for (int j = 0; j < o.d; ++j)
            centers[static_cast<std::size_t>(c) * o.d + j] = point(c, j, o.k, o.seed);

    const std::size_t width = centers.size();
    std::vector<double> thread_sums(static_cast<std::size_t>(threads) * width);
    std::vector<unsigned long long> thread_counts(static_cast<std::size_t>(threads) * o.k);
    std::vector<double> local_sums(width), global_sums(width);
    std::vector<unsigned long long> local_counts(o.k), global_counts(o.k);
    double communication = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    const double begun = MPI_Wtime();
    for (int iteration = 0; iteration < o.iterations; ++iteration) {
        std::fill(thread_sums.begin(), thread_sums.end(), 0.0);
        std::fill(thread_counts.begin(), thread_counts.end(), 0);
#pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            double* sums = thread_sums.data() + static_cast<std::size_t>(tid) * width;
            unsigned long long* counts = thread_counts.data() + static_cast<std::size_t>(tid) * o.k;
#pragma omp for schedule(static)
            for (std::uint64_t i = 0; i < local_n; ++i) {
                const double* x = data.data() + static_cast<std::size_t>(i) * o.d;
                const int c = nearest(x, centers, o.d, o.k);
                ++counts[c];
                for (int j = 0; j < o.d; ++j) sums[static_cast<std::size_t>(c) * o.d + j] += x[j];
            }
        }
        std::fill(local_sums.begin(), local_sums.end(), 0.0);
        std::fill(local_counts.begin(), local_counts.end(), 0);
        for (int t = 0; t < threads; ++t) {
            for (std::size_t j = 0; j < width; ++j) local_sums[j] += thread_sums[static_cast<std::size_t>(t) * width + j];
            for (int c = 0; c < o.k; ++c) local_counts[c] += thread_counts[static_cast<std::size_t>(t) * o.k + c];
        }
        const double before = MPI_Wtime();
        MPI_Allreduce(local_sums.data(), global_sums.data(), static_cast<int>(width), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(local_counts.data(), global_counts.data(), o.k, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        communication += MPI_Wtime() - before;
        for (int c = 0; c < o.k; ++c)
            if (global_counts[c])
                for (int j = 0; j < o.d; ++j)
                    centers[static_cast<std::size_t>(c) * o.d + j] = global_sums[static_cast<std::size_t>(c) * o.d + j] / global_counts[c];
    }
    const double elapsed = MPI_Wtime() - begun;
    double max_elapsed = 0, max_communication = 0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&communication, &max_communication, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    double local_inertia = 0;
    unsigned long long local_purity = 0;
#pragma omp parallel for reduction(+:local_inertia,local_purity) schedule(static)
    for (std::uint64_t i = 0; i < local_n; ++i) {
        const double* x = data.data() + static_cast<std::size_t>(i) * o.d;
        const int c = nearest(x, centers, o.d, o.k);
        local_purity += (c == static_cast<int>((start + i) % o.k));
        for (int j = 0; j < o.d; ++j) {
            const double delta = x[j] - centers[static_cast<std::size_t>(c) * o.d + j];
            local_inertia += delta * delta;
        }
    }
    double inertia = 0;
    unsigned long long purity = 0;
    MPI_Reduce(&local_inertia, &inertia, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_purity, &purity, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        double max_error = -1;
        if (o.validate) {
            const auto reference = serial_reference(o);
            max_error = 0;
            for (std::size_t j = 0; j < width; ++j)
                max_error = std::max(max_error, std::abs(centers[j] - reference[j]));
        }
        std::cout << std::setprecision(12)
            << "{\"n\":" << o.n << ",\"d\":" << o.d << ",\"k\":" << o.k
            << ",\"iterations\":" << o.iterations << ",\"ranks\":" << ranks
            << ",\"threads\":" << threads << ",\"seconds\":" << max_elapsed
            << ",\"communication_seconds\":" << max_communication
            << ",\"inertia\":" << inertia << ",\"label_agreement\":" << static_cast<double>(purity) / o.n
            << ",\"max_reference_error\":" << max_error << "}\n";
        if (o.validate && max_error > 1e-8) std::cerr << "validation failed: " << max_error << '\n';
        if (o.validate && max_error > 1e-8) MPI_Abort(MPI_COMM_WORLD, 3);
    }
    MPI_Finalize();
    return 0;
}
