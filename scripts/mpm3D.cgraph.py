import argparse
import os
import numpy as np
import taichi as ti

parser = argparse.ArgumentParser()
parser.add_argument("--arch", type=str, default='vulkan')
parser.add_argument("--cgraph", action='store_true', default=False)
args = parser.parse_args()

def T(a):
    phi, theta = np.radians(28), np.radians(32)

    a = a - 0.5
    x, y, z = a[:, 0], a[:, 1], a[:, 2]
    cp, sp = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    x, z = x * cp + z * sp, z * cp - x * sp
    u, v = x, y * ct + z * st
    return np.array([u, v]).swapaxes(0, 1) + 0.5

def get_save_dir(name, arch):
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(curr_dir, f"{name}_{arch}")

def compile_mpm3D(arch, save_compute_graph,run=False):
    ti.init(arch, vk_api_version="1.0")

    if ti.lang.impl.current_cfg().arch != arch:
        return

    # dim, n_grid, steps, dt = 2, 128, 20, 2e-4
    # dim, n_grid, steps, dt = 2, 256, 32, 1e-4
    #dim, n_grid, steps, dt = 3, 32, 25, 4e-4
    dim, n_grid, steps, dt = 3, 64, 25, 2e-4
    #dim, n_grid, steps, dt = 3, 128, 25, 8e-5

    n_particles = n_grid**dim // 2 ** (dim - 1)
    dx = 1 / n_grid

    p_rho = 1
    p_vol = (dx * 0.5) ** 2
    p_mass = p_vol * p_rho
    gravity = 9.8
    bound = 3
    E = 400


    neighbour = (3,) * dim

    ti.init(arch=arch)

    @ti.kernel
    def substep_reset_grid(grid_v: ti.types.ndarray(ndim=3),
                           grid_m: ti.types.ndarray(ndim=3)):
        for i, j ,k in grid_m:
            grid_v[i, j,k] = [0, 0, 0]
            grid_m[i, j,k] = 0

    @ti.kernel
    def substep_p2g(x: ti.types.ndarray(ndim=1), v: ti.types.ndarray(ndim=1),
                    C: ti.types.ndarray(ndim=1), J: ti.types.ndarray(ndim=1),
                    grid_v: ti.types.ndarray(ndim=3),
                    grid_m: ti.types.ndarray(ndim=3)):
        pass
        for p in x:
            Xp = x[p] / dx
            base = int(Xp - 0.5)
            fx = Xp - base
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            stress = -dt * 4 * E * p_vol * (J[p] - 1) / dx**2
            affine = ti.Matrix.identity(float, dim) * stress + p_mass * C[p]
            for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
                dpos = (offset - fx) * dx
                weight = 1.0
                for i in ti.static(range(dim)):
                    weight *= w[offset[i]][i]
                grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
                grid_m[base + offset] += weight * p_mass

    @ti.kernel
    def substep_update_grid_v(grid_v: ti.types.ndarray(ndim=3),
                              grid_m: ti.types.ndarray(ndim=3)):
        for I in ti.grouped(grid_m):
            if grid_m[I] > 0:
                grid_v[I] /= grid_m[I]
            grid_v[I][1] -= dt * gravity
            cond = (I < bound) & (grid_v[I] < 0) | (I > n_grid - bound) & (grid_v[I] > 0)
            grid_v[I] = ti.select(cond, 0, grid_v[I])

    @ti.kernel
    def substep_g2p(x: ti.types.ndarray(ndim=1), v: ti.types.ndarray(ndim=1),
                    C: ti.types.ndarray(ndim=1), J: ti.types.ndarray(ndim=1),
                    grid_v: ti.types.ndarray(ndim=3),
                    pos: ti.types.ndarray(ndim=1)):
        for p in x:
            Xp = x[p] / dx
            base = int(Xp - 0.5)
            fx = Xp - base
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            new_v = ti.zero(v[p])
            new_C = ti.zero(C[p])
            for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
                dpos = (offset - fx) * dx
                weight = 1.0
                for i in ti.static(range(dim)):
                    weight *= w[offset[i]][i]
                g_v = grid_v[base + offset]
                new_v += weight * g_v
                new_C += 4 * weight * g_v.outer_product(dpos) / dx**2
            v[p] = new_v
            x[p] += dt * v[p]
            J[p] *= 1 + dt * new_C.trace()
            C[p] = new_C

    @ti.kernel
    def init_particles(x: ti.types.ndarray(ndim=1), v: ti.types.ndarray(ndim=1),
                       J: ti.types.ndarray(ndim=1)):
        for i in range(x.shape[0]):
            x[i] = [ti.random() * 0.4 + 0.2, ti.random() * 0.4 + 0.2, ti.random() * 0.4 + 0.2]
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

    # GGUI only supports vec3 vertex so we need an extra `pos` here
    # This is not necessary if you're not going to render it using GGUI.
    # Let's keep this hack here so that the shaders serialized by this
    # script can be loaded and rendered in the provided script in taichi-aot-demo.
    pos = ti.Vector.ndarray(3, ti.f32, n_particles)
    x = ti.Vector.ndarray(3, ti.f32, shape=(n_particles))
    v = ti.Vector.ndarray(3, ti.f32, shape=(n_particles))

    C = ti.Matrix.ndarray(3, 3, ti.f32, shape=(n_particles))
    J = ti.ndarray(ti.f32, shape=(n_particles))
    grid_v = ti.Vector.ndarray(3, ti.f32, shape=(n_grid, n_grid, n_grid))
    grid_m = ti.ndarray(ti.f32, shape=(n_grid, n_grid, n_grid))
    if run == True:
        gui=ti.GUI('MPM3D',res=(800,800))
        init_particles(x,v,J)
        while gui.running and not gui.get_event(gui.ESCAPE):
            for i in range(50):
                substep_reset_grid(grid_v,grid_m)
                substep_p2g(x,v,C,J,grid_v,grid_m)
                substep_update_grid_v(grid_v,grid_m)
                substep_g2p(x,v,C,J,grid_v,pos)
            gui.circles(T(x.to_numpy()),radius=1.5,color=0x66CCFF)
            gui.show()
    mod = ti.aot.Module(arch)
    mod.add_graph('init', g_init)
    mod.add_graph('update', g_update)
    
    # save_dir = get_save_dir("mpm3D", args.arch)
    # os.makedirs(save_dir, exist_ok=True)
    # mod.save(save_dir, '')
    mod.archive("Assets/Resources/TaichiModules/mpm3D.cgraph.tcm")
    print("AOT done")

if __name__ == "__main__":
    compile_for_cgraph = args.cgraph

    if args.arch == "vulkan":
        compile_mpm3D(arch=ti.vulkan, save_compute_graph=compile_for_cgraph,run=True)
    elif args.arch == "cuda":
        compile_mpm3D(arch=ti.cuda, save_compute_graph=compile_for_cgraph)
    elif args.arch == "x64":
        compile_mpm3D(arch=ti.x64, save_compute_graph=compile_for_cgraph,run=True)
    else:
        assert False
