import argparse
import os
import taichi as ti

parser = argparse.ArgumentParser()
parser.add_argument("--arch", type=str, default='vulkan')
parser.add_argument("--cgraph", action='store_true', default=False)
args = parser.parse_args()

def get_save_dir(name, arch):
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(curr_dir, f"{name}_{arch}")

def compile_mpm3D(arch, save_compute_graph):
    ti.init(arch, vk_api_version="1.0")

    if ti.lang.impl.current_cfg().arch != arch:
        return

    n_particles = 8192 * 5
    n_grid = 32
    dt = 4e-4

    p_rho = 1
    gravity = 9.8
    bound = 3
    E = 400

    @ti.kernel
    def substep_reset_grid(grid_v: ti.types.ndarray(ndim=3),
                           grid_m: ti.types.ndarray(ndim=3)):
        for i, j, k in grid_m:
            grid_v[i, j, k] = [0, 0, 0]
            grid_m[i, j, k] = 0

    @ti.kernel
    def substep_p2g(x: ti.types.ndarray(ndim=1), v: ti.types.ndarray(ndim=1),
                    C: ti.types.ndarray(ndim=1), J: ti.types.ndarray(ndim=1),
                    grid_v: ti.types.ndarray(ndim=3),
                    grid_m: ti.types.ndarray(ndim=3)):
        for p in x:
            dx = 1 / grid_v.shape[0]
            p_vol = (dx * 0.5)**3
            p_mass = p_vol * p_rho
            Xp = x[p] / dx
            base = int(Xp - 0.5)
            fx = Xp - base
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
            stress = -dt * 4 * E * p_vol * (J[p] - 1) / dx**2
            affine = ti.Matrix([[stress, 0, 0], [0, stress, 0], [0, 0, stress]]) + p_mass * C[p]
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                dpos = (offset - fx) * dx
                weight = w[i].x * w[j].y * w[k].z
                grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
                grid_m[base + offset] += weight * p_mass

    @ti.kernel
    def substep_update_grid_v(grid_v: ti.types.ndarray(ndim=3),
                              grid_m: ti.types.ndarray(ndim=3)):
        for i, j, k in grid_m:
            num_grid = grid_v.shape[0]
            if grid_m[i, j, k] > 0:
                grid_v[i, j, k] /= grid_m[i, j, k]
            grid_v[i, j, k].y -= dt * gravity
            if i < bound and grid_v[i, j, k].x < 0:
                grid_v[i, j, k].x = 0
            if i > num_grid - bound and grid_v[i, j, k].x > 0:
                grid_v[i, j, k].x = 0
            if j < bound and grid_v[i, j, k].y < 0:
                grid_v[i, j, k].y = 0
            if j > num_grid - bound and grid_v[i, j, k].y > 0:
                grid_v[i, j, k].y = 0
            if k < bound and grid_v[i, j, k].z < 0:
                grid_v[i, j, k].z = 0
            if k > num_grid - bound and grid_v[i, j, k].z > 0:
                grid_v[i, j, k].z = 0

    @ti.kernel
    def substep_g2p(x: ti.types.ndarray(ndim=1), v: ti.types.ndarray(ndim=1),
                    C: ti.types.ndarray(ndim=1), J: ti.types.ndarray(ndim=1),
                    grid_v: ti.types.ndarray(ndim=3),
                    pos: ti.types.ndarray(ndim=1)):
        for p in x:
            dx = 1 / grid_v.shape[0]
            Xp = x[p] / dx
            base = int(Xp - 0.5)
            fx = Xp - base
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
            new_v = ti.Vector.zero(float, 3)
            new_C = ti.Matrix.zero(float, 3, 3)
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                dpos = (offset - fx) * dx
                weight = w[i].x * w[j].y * w[k].z
                g_v = grid_v[base + offset]
                new_v += weight * g_v
                new_C += 4 * weight * g_v.outer_product(dpos) / dx**2
            v[p] = new_v
            x[p] += dt * v[p]
            pos[p] = [x[p][0], x[p][1], x[p][2]]
            J[p] *= 1 + dt * new_C.trace()
            C[p] = new_C

    @ti.kernel
    def init_particles(x: ti.types.ndarray(ndim=1), v: ti.types.ndarray(ndim=1),
                       J: ti.types.ndarray(ndim=1)):
        for i in range(x.shape[0]):
            x[i] = [ti.random() * 0.4 + 0.2, ti.random() * 0.4 + 0.2, ti.random() * 0.4 + 0.2]
            v[i] = [0, -1, 0]
            J[i] = 1

    N_ITER = 50

    sym_x = ti.graph.Arg(ti.graph.ArgKind.NDARRAY,
                            'x',
                            ndim=1,
                            dtype=ti.types.vector(3, ti.f32))
    sym_v = ti.graph.Arg(ti.graph.ArgKind.NDARRAY,
                            'v',
                            ndim=1,
                            dtype=ti.types.vector(3, ti.f32))
    sym_C = ti.graph.Arg(ti.graph.ArgKind.NDARRAY,
                            'C',
                            ndim=1,
                            dtype=ti.types.matrix(3, 3, ti.f32))
    sym_J = ti.graph.Arg(ti.graph.ArgKind.NDARRAY,
                            'J',
                            ti.f32,
                            ndim=1)
    sym_grid_v = ti.graph.Arg(ti.graph.ArgKind.NDARRAY,
                                'grid_v',
                                ndim=3,
                                dtype=ti.types.vector(3, ti.f32))
    sym_grid_m = ti.graph.Arg(ti.graph.ArgKind.NDARRAY,
                                'grid_m',
                                ti.f32,
                                ndim=3)
    sym_pos = ti.graph.Arg(ti.graph.ArgKind.NDARRAY,
                            'pos',
                            ndim=1,
                            dtype=ti.types.vector(3, ti.f32))

    g_init_builder = ti.graph.GraphBuilder()
    g_init_builder.dispatch(init_particles, sym_x, sym_v, sym_J)

    g_update_builder = ti.graph.GraphBuilder()
    substep = g_update_builder.create_sequential()

    substep.dispatch(substep_reset_grid, sym_grid_v, sym_grid_m)
    substep.dispatch(substep_p2g, sym_x, sym_v, sym_C, sym_J, sym_grid_v,
                        sym_grid_m)
    substep.dispatch(substep_update_grid_v, sym_grid_v, sym_grid_m)
    substep.dispatch(substep_g2p, sym_x, sym_v, sym_C, sym_J, sym_grid_v,
                        sym_pos)

    for i in range(N_ITER):
        g_update_builder.append(substep)

    g_init = g_init_builder.compile()
    g_update = g_update_builder.compile()

    pos = ti.Vector.ndarray(3, ti.f32, n_particles)
    x = ti.Vector.ndarray(3, ti.f32, shape=(n_particles))
    v = ti.Vector.ndarray(3, ti.f32, shape=(n_particles))

    C = ti.Matrix.ndarray(3, 3, ti.f32, shape=(n_particles))
    J = ti.ndarray(ti.f32, shape=(n_particles))
    grid_v = ti.Vector.ndarray(3, ti.f32, shape=(n_grid, n_grid, n_grid))
    grid_m = ti.ndarray(ti.f32, shape=(n_grid, n_grid, n_grid))

    mod = ti.aot.Module(arch)
    mod.add_graph('init', g_init)
    mod.add_graph('update', g_update)
    
    mod.archive("Assets/Resources/TaichiModules/mpm3d.cgraph.tcm")
    print("AOT done")

if __name__ == "__main__":
    compile_for_cgraph = args.cgraph

    if args.arch == "vulkan":
        compile_mpm3D(arch=ti.vulkan, save_compute_graph=compile_for_cgraph)
    elif args.arch == "cuda":
        compile_mpm3D(arch=ti.cuda, save_compute_graph=compile_for_cgraph)
    elif args.arch == "x64":
        compile_mpm3D(arch=ti.x64, save_compute_graph=compile_for_cgraph)
    else:
        assert False
