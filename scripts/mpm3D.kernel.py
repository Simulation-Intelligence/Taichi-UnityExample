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

def compile_mpm3D(arch, save_compute_graph, run=False):
    ti.init(arch, vk_api_version="1.0")

    if ti.lang.impl.current_cfg().arch != arch:
        return

    dim, n_grid, steps, dt = 3, 64, 25, 1e-4
    n_particles = n_grid**dim // 2 ** (dim - 1)
    dx = 0.5 / n_grid

    p_rho = 1
    p_vol = (dx * 0.5) ** 2
    p_mass = p_vol * p_rho
    gx=0
    gy=-9.8
    gz=0
    bound = 3
    E = 400

    neighbour = (3,) * dim

    @ti.kernel 
    def substep_reset_grid(grid_v: ti.types.ndarray(ndim=3), grid_m: ti.types.ndarray(ndim=3)):
        for I in ti.grouped(grid_m):
            grid_v[I] = [0, 0, 0]
            grid_m[I] = 0

    @ti.kernel
    def substep_p2g(x: ti.types.ndarray(ndim=1), v: ti.types.ndarray(ndim=1), C: ti.types.ndarray(ndim=1),
                    J: ti.types.ndarray(ndim=1), grid_v: ti.types.ndarray(ndim=3), grid_m: ti.types.ndarray(ndim=3)):
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
    def substep_calculate_signed_distance_field(obstacle_pos: ti.types.ndarray(ndim=1), obstacle_velocity: ti.types.ndarray(ndim=1),  sdf: ti.types.ndarray(ndim=3), grid_obstacle_vel: ti.types.ndarray(ndim=3),obstacle_radius:float):
        for I in ti.grouped(sdf):
            pos=I*dx+dx*0.5
            sdf[I] = (pos - obstacle_pos[0]).norm() - obstacle_radius
            grid_obstacle_vel[I] = obstacle_velocity[0]
            cond = (I < bound)  | (I > n_grid - bound) 
            

    @ti.kernel
    def substep_update_grid_v(grid_v: ti.types.ndarray(ndim=3), grid_m: ti.types.ndarray(ndim=3),sdf: ti.types.ndarray(ndim=3),grid_obstacle_vel:ti.types.ndarray(ndim=3),gx:float,gy:float,gz:float):
        for I in ti.grouped(grid_m):
            if grid_m[I] > 0:
                grid_v[I] /= grid_m[I]
            gravity = ti.Vector([gx,gy,gz])
            grid_v[I] += dt * gravity
            if sdf[I] <= 0:
                grid_v[I] = grid_obstacle_vel[I]
            cond = (I < bound) & (grid_v[I] < 0) | (I > n_grid - bound) & (grid_v[I] > 0)
            grid_v[I] = ti.select(cond, 0, grid_v[I])




    @ti.kernel
    def substep_g2p(x: ti.types.ndarray(ndim=1), v: ti.types.ndarray(ndim=1), C: ti.types.ndarray(ndim=1),
                    J: ti.types.ndarray(ndim=1), grid_v: ti.types.ndarray(ndim=3)):
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
    def init_particles(x: ti.types.ndarray(ndim=1), v: ti.types.ndarray(ndim=1), J: ti.types.ndarray(ndim=1)):
        for i in range(x.shape[0]):
            x[i] = [ti.random() * 0.2 + 0.1, ti.random() * 0.2 + 0.1, ti.random() * 0.2 + 0.1]
            J[i] = 1

    x = ti.Vector.ndarray(3, ti.f32, shape=(n_particles))
    v = ti.Vector.ndarray(3, ti.f32, shape=(n_particles))
    C = ti.Matrix.ndarray(3, 3, ti.f32, shape=(n_particles))
    J = ti.ndarray(ti.f32, shape=(n_particles))
    grid_v = ti.Vector.ndarray(3, ti.f32, shape=(n_grid, n_grid, n_grid))
    grid_m = ti.ndarray(ti.f32, shape=(n_grid, n_grid, n_grid))
    sdf = ti.ndarray(ti.f32, shape=(n_grid, n_grid, n_grid))
    grid_obstacle_vel = ti.Vector.ndarray(3, ti.f32, shape=(n_grid, n_grid, n_grid))
    obstacle_pos =  ti.Vector.ndarray(3, ti.f32, shape=(1))
    obstacle_velocity = ti.Vector.ndarray(3, ti.f32, shape=(1))
    obstacle_pos[0]   = ti.Vector([0.25,0.25,0.25])
    obstacle_velocity[0] = ti.Vector([0.1,0.1,0.1])
    obstacle_radius = 0.1
    


    def substep():
        substep_reset_grid(grid_v, grid_m)
        substep_p2g(x, v, C, J, grid_v, grid_m)
        substep_calculate_signed_distance_field(obstacle_pos, obstacle_velocity ,sdf,grid_obstacle_vel,obstacle_radius)
        substep_update_grid_v(grid_v, grid_m,sdf,grid_obstacle_vel,gx,gy,gz)
        substep_g2p(x, v, C, J, grid_v)

    def run_aot():
        mod = ti.aot.Module(arch)
        mod.add_kernel(substep_reset_grid, template_args={'grid_v': grid_v, 'grid_m': grid_m})
        mod.add_kernel(substep_p2g, template_args={'x': x, 'v': v, 'C': C, 'J': J, 'grid_v': grid_v, 'grid_m': grid_m})
        mod.add_kernel(substep_calculate_signed_distance_field, template_args={'obstacle_pos': obstacle_pos, 'obstacle_velocity': obstacle_velocity, 'sdf': sdf, 'grid_obstacle_vel': grid_obstacle_vel})
        mod.add_kernel(substep_update_grid_v, template_args={'grid_v': grid_v, 'grid_m': grid_m, 'sdf': sdf, 'grid_obstacle_vel': grid_obstacle_vel})
        mod.add_kernel(substep_g2p, template_args={'x': x, 'v': v, 'C': C, 'J': J, 'grid_v': grid_v})
        mod.add_kernel(init_particles, template_args={'x': x, 'v': v, 'J': J})
        mod.archive("Assets/Resources/TaichiModules/mpm3D.kernel.tcm")
        print("AOT done")

    if run:
        gui = ti.GUI('MPM3D', res=(800, 800))
        init_particles(x, v, J)
        while gui.running and not gui.get_event(gui.ESCAPE):
            for i in range(50):
                substep()
            gui.circles(T(x.to_numpy()), radius=1.5, color=0x66CCFF)
            gui.show()
    run_aot()

if __name__ == "__main__":
    compile_for_cgraph = args.cgraph

    if args.arch == "vulkan":
        compile_mpm3D(arch=ti.vulkan, save_compute_graph=compile_for_cgraph, run=True)
    elif args.arch == "cuda":
        compile_mpm3D(arch=ti.cuda, save_compute_graph=compile_for_cgraph)
    elif args.arch == "x64":
        compile_mpm3D(arch=ti.x64, save_compute_graph=compile_for_cgraph, run=True)
    else:
        assert False
