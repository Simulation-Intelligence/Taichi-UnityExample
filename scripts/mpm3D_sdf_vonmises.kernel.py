import argparse
import os
import numpy as np
import taichi as ti
import json

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
    ti.init(arch, vk_api_version="1.0",debug=False)  

    if ti.lang.impl.current_cfg().arch != arch:
        return

    dim, n_grid, steps, dt,cube_size ,particle_per_grid= 3, 64, 25, 1e-4,0.2,16
    n_particles  = int((((n_grid*cube_size)**dim) *particle_per_grid))
    print(n_particles)
    dx = 1/n_grid

    p_rho = 1000
    p_vol = dx** 3  
    p_mass = p_vol * p_rho/ particle_per_grid
    allowed_cfl = 0.5
    v_allowed = dx * allowed_cfl / dt
    gx=0
    gy=-9.8
    gz=0
    k=0.5
    bound = 3
    E = 10000  # Young's modulus for snow
    SigY=1000
    nu = 0.45  # Poisson's ratio
    mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters
    friction_angle = 30.0
    sin_phi = ti.sin(friction_angle / 180 * 3.141592653)
    alpha = ti.sqrt(2 / 3) * 2 * sin_phi / (3 - sin_phi)

    neighbour = (3,) * dim

    @ti.kernel 
    def substep_reset_grid(grid_v: ti.types.ndarray(ndim=3), grid_m: ti.types.ndarray(ndim=3)):
        for I in ti.grouped(grid_m):
            grid_v[I] = [0, 0, 0]
            grid_m[I] = 0

    @ti.kernel
    def substep_neohookean_p2g(x: ti.types.ndarray(ndim=1), v: ti.types.ndarray(ndim=1), C: ti.types.ndarray(ndim=1), 
                               dg: ti.types.ndarray(ndim=1), grid_v: ti.types.ndarray(ndim=3), grid_m: ti.types.ndarray(ndim=3),
                               mu:ti.f32,la:ti.f32,p_vol:ti.f32,p_mass:ti.f32,dx:ti.f32,dt:ti.f32):
        for p in x:
            Xp = x[p] / dx
            base = int(Xp - 0.5)
            fx = Xp - base
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]

            dg[p] = (ti.Matrix.identity(float, dim) + dt * C[p]) @ dg[p]
            J=dg[p].determinant()
            cauchy=mu*(dg[p]@dg[p].transpose())+ti.Matrix.identity(float, dim)*(la*ti.log(J)-mu)
            stress=-(dt * p_vol * 4 /dx**2) * cauchy
            affine = stress + p_mass * C[p]

            for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
                dpos = (offset - fx) * dx
                weight = 1.0
                for i in ti.static(range(dim)):
                    weight *= w[offset[i]][i]
                grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
                grid_m[base + offset] += weight * p_mass
        
    @ti.kernel
    def substep_kirchhoff_p2g(x: ti.types.ndarray(ndim=1), v: ti.types.ndarray(ndim=1), C: ti.types.ndarray(ndim=1),
                               dg: ti.types.ndarray(ndim=1), grid_v: ti.types.ndarray(ndim=3), grid_m: ti.types.ndarray(ndim=3),
                               mu:ti.f32,la:ti.f32,p_vol:ti.f32,p_mass:ti.f32,dx:ti.f32,dt:ti.f32):
        for p in x:
            Xp = x[p] / dx
            base = int(Xp - 0.5)
            fx = Xp - base
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]

            dg[p] = (ti.Matrix.identity(float, dim) + dt * C[p]) @ dg[p]
            U, sig, V = ti.svd(dg[p])
            J_new = sig[0, 0] * sig[1, 1] * sig[2, 2]
            stress = 2 * mu * (dg[p] - U @ V.transpose()) @ dg[p].transpose() + ti.Matrix.identity(
                float, dim) * la * J_new * (J_new - 1)
            stress = (-dt * p_vol * 4) * stress / dx**2
            affine = stress + p_mass * C[p]

            for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
                dpos = (offset - fx) * dx
                weight = 1.0
                for i in ti.static(range(dim)):
                    weight *= w[offset[i]][i]
                grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
                grid_m[base + offset] += weight * p_mass
    
    @ti.kernel
    def substep_calculate_signed_distance_field(obstacle_pos: ti.types.ndarray(ndim=1), obstacle_velocity: ti.types.ndarray(ndim=1),  
                                                sdf: ti.types.ndarray(ndim=3), grid_obstacle_vel: ti.types.ndarray(ndim=3),
                                                obstacle_radius:ti.types.ndarray(ndim=1),
                                                k:ti.f32,dx:ti.f32,dt:ti.f32):
        for I in ti.grouped(sdf):
            pos = I * dx + dx * 0.5
            min_dist = float('inf')
            min_vel = ti.Vector([0.0, 0.0, 0.0])
            
            for j in obstacle_pos:
                dist = (pos - obstacle_pos[j]).norm() - obstacle_radius[j]
                if dist < min_dist:
                    min_dist = dist
                    norm= (pos-obstacle_pos[j]).normalized()
                    min_vel = norm*(-dist)/dt*k

            sdf[I] = min_dist
            grid_obstacle_vel[I] = min_vel

    @ti.kernel
    def substep_update_grid_v(grid_v: ti.types.ndarray(ndim=3), grid_m: ti.types.ndarray(ndim=3),sdf: ti.types.ndarray(ndim=3),
                              grid_obstacle_vel:ti.types.ndarray(ndim=3),gx:float,gy:float,gz:float,
                              v_allowed:ti.f32,dt:ti.f32):
        for I in ti.grouped(grid_m):
            if grid_m[I] > 0:
                grid_v[I] /= grid_m[I]
            gravity = ti.Vector([gx,gy,gz])
            grid_v[I] += dt * gravity
            if sdf[I] <= 0:
                grid_v[I] = grid_obstacle_vel[I]
            cond = (I < bound) & (grid_v[I] < 0) | (I > n_grid - bound) & (grid_v[I] > 0)
            grid_v[I] = ti.select(cond, 0, grid_v[I])
            grid_v[I] = min(max(grid_v[I], -v_allowed), v_allowed)
            sdf[I] = 1
            grid_obstacle_vel[I] = [0,0,0]

    @ti.kernel
    def substep_g2p(x: ti.types.ndarray(ndim=1), v: ti.types.ndarray(ndim=1), C: ti.types.ndarray(ndim=1),grid_v: ti.types.ndarray(ndim=3),
                    dx:ti.f32,dt:ti.f32):
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
            C[p] = new_C

    @ti.kernel
    def substep_apply_Drucker_Prager_plasticity(dg: ti.types.ndarray(ndim=1),lambda_0:ti.f32,mu_0:ti.f32,alpha:ti.f32):
        for p in dg:
            U, sig, V = ti.svd(dg[p])

            # 将 sig 转换为向量
            sig_vec = ti.Vector([sig[i, i] for i in range(dim)])

            epsilon = ti.log(ti.abs(sig_vec))
            trace_epsilon = epsilon.sum()

            epsilon_hat = epsilon - trace_epsilon / 3 * ti.Vector([1.0, 1.0, 1.0])
            epsilon_hat_squared_norm = epsilon_hat.norm_sqr()
            epsilon_hat_norm = ti.sqrt(epsilon_hat_squared_norm)
            delta_gamma = 0.0

            if trace_epsilon <= 0:
                delta_gamma = epsilon_hat_norm + (3 * lambda_0 + 2 * mu_0) / (2 * mu_0) * trace_epsilon * alpha
            else:
                delta_gamma = epsilon_hat_norm

            Z = ti.Matrix.identity(float, 3)
            if delta_gamma <= 0:
                for i in range(dim):
                    Z[i, i] = sig_vec[i]
            else:
                H = epsilon - (delta_gamma / epsilon_hat_norm) * epsilon_hat
                E = ti.exp(H)
                Z = ti.Matrix([[E[0], 0, 0], [0, E[1], 0], [0, 0, E[2]]])
                
            dg[p] = U @ Z @ V.transpose()
    
    @ti.kernel
    def substep_apply_Von_Mises_plasticity(dg: ti.types.ndarray(ndim=1),mu_0:ti.f32,SigY:ti.f32):
        for p in dg:
            U, sig, V = ti.svd(dg[p])

            # 将 sig 转换为向量
            sig_vec = ti.Vector([sig[i, i] for i in range(dim)])

            epsilon = ti.log(ti.abs(sig_vec))
            trace_epsilon = epsilon.sum()

            epsilon_hat = epsilon - trace_epsilon / 3 * ti.Vector([1.0, 1.0, 1.0])
            epsilon_hat_squared_norm = epsilon_hat.norm_sqr()
            epsilon_hat_norm = ti.sqrt(epsilon_hat_squared_norm)
            delta_gamma = epsilon_hat_norm - SigY / (2 * mu_0)

            Z = ti.Matrix.identity(float, 3)
            if delta_gamma <= 0:
                for i in range(dim):
                    Z[i, i] = sig_vec[i]
            else:
                H = epsilon - (delta_gamma / epsilon_hat_norm) * epsilon_hat
                E = ti.exp(H)
                Z = ti.Matrix([[E[0], 0, 0], [0, E[1], 0], [0, 0, E[2]]])
                
            dg[p] = U @ Z @ V.transpose()

    @ti.kernel
    def substep_apply_clamp_plasticity(dg: ti.types.ndarray(ndim=1),min_clamp:ti.f32,max_clamp:ti.f32):
        for p in dg:
            U, sig, V = ti.svd(dg[p]) 
            for i in ti.static(range(dim)):
                sig[i, i] = min(max(sig[i, i], 1-min_clamp), 1+max_clamp)
            dg[p] = U @ sig @ V.transpose()

    @ti.kernel
    def init_particles(x: ti.types.ndarray(ndim=1), v: ti.types.ndarray(ndim=1), dg: ti.types.ndarray(ndim=1),cube_size:ti.f32):
        for i in range(x.shape[0]):
            x[i] = [ti.random() * cube_size + (0.5-cube_size/2), ti.random() * cube_size, ti.random() * cube_size+0.001 + (0.5-cube_size/2)]
            dg[i] = ti.Matrix.identity(float, dim)
    
    @ti.func
    def calculate_point_segment_tangent(px: ti.f32, py: ti.f32, pz: ti.f32,
                                        sx: ti.f32, sy: ti.f32, sz: ti.f32,
                                        ex: ti.f32, ey: ti.f32, ez: ti.f32) -> ti.Vector:
        point = ti.Vector([px, py, pz])
        start = ti.Vector([sx, sy, sz])
        end = ti.Vector([ex, ey, ez])
        v = end - start
        w = point - start
        c1 = w.dot(v)
        c2 = v.dot(v)
        if c1 <= 0:
            normal_vector = point - start
        elif c1 >= c2:
            normal_vector = point - end
        else:
            b = c1 / c2
            Pb = start + b * v
            normal_vector = point - Pb
        return normal_vector
    
    @ti.func
    def calculate_point_segment_distance(px: ti.f32, py: ti.f32, pz: ti.f32,
                              sx: ti.f32, sy: ti.f32, sz: ti.f32,
                              ex: ti.f32, ey: ti.f32, ez: ti.f32) -> ti.f32:
        point = ti.Vector([px, py, pz])
        start = ti.Vector([sx, sy, sz])
        end = ti.Vector([ex, ey, ez])
        v = end - start
        w = point - start
        c1 = w.dot(v)
        c2 = v.dot(v)
        distance = 0.0
        if c1 <= 0:
            distance = (point - start).norm()
        elif c1 >= c2:
            distance = (point - end).norm()
        else:
            b = c1 / c2
            Pb = start + b * v
            distance = (point - Pb).norm()
        return distance
    
    @ti.kernel
    def substep_calculate_hand_sdf(skeleton_segments: ti.types.ndarray(ndim=2), 
                                   hand_sdf: ti.types.ndarray(ndim=3),
                                   skeleton_capsule_radius: ti.types.ndarray(ndim=1),
                                   dx:ti.f32):
        #if (skeleton_segments.shape[0] != 0):
        for I in ti.grouped(hand_sdf):
            pos = I * dx + dx * 0.5
            min_dist = float('inf')
            
            for i in range(skeleton_segments.shape[0]):
                sx, sy, sz = skeleton_segments[i, 0][0], skeleton_segments[i, 0][1], skeleton_segments[i, 0][2]
                ex, ey, ez = skeleton_segments[i, 1][0], skeleton_segments[i, 1][1], skeleton_segments[i, 1][2]
                dist = calculate_point_segment_distance(pos[0], pos[1], pos[2], sx, sy, sz, ex, ey, ez)-skeleton_capsule_radius[i]
                if dist < min_dist:
                    min_dist = dist
            hand_sdf[I] = min_dist
    
    # store distance for all nodes
    hand_sdf = ti.ndarray(ti.f32, shape=(n_grid, n_grid, n_grid))
    skeleton_segments = ti.Vector.ndarray(3, ti.f32, shape=(24, 2))
    skeleton_capsule_radius = ti.ndarray(ti.f32, shape=(24))
    
    x = ti.Vector.ndarray(3, ti.f32, shape=(n_particles))
    v = ti.Vector.ndarray(3, ti.f32, shape=(n_particles))
    C = ti.Matrix.ndarray(3, 3, ti.f32, shape=(n_particles))
    dg = ti.Matrix.ndarray(3, 3, ti.f32, shape=(n_particles))
    J = ti.ndarray(ti.f32, shape=(n_particles))
    grid_v = ti.Vector.ndarray(3, ti.f32, shape=(n_grid, n_grid, n_grid))
    grid_m = ti.ndarray(ti.f32, shape=(n_grid, n_grid, n_grid))
    sdf = ti.ndarray(ti.f32, shape=(n_grid, n_grid, n_grid))
    grid_obstacle_vel = ti.Vector.ndarray(3, ti.f32, shape=(n_grid, n_grid, n_grid))
    obstacle_pos = ti.Vector.ndarray(3, ti.f32, shape=(1))
    obstacle_velocity = ti.Vector.ndarray(3, ti.f32, shape=(1))
    obstacle_pos[0] = ti.Vector([0.25, 0.25, 0.25])
    obstacle_velocity[0] = ti.Vector([0.1, 0.1, 0.1])
    obstacle_radius = ti.ndarray(ti.f32, shape=(1))
    obstacle_radius[0] = 0
    
    def substep():
        substep_reset_grid(grid_v, grid_m)
        substep_kirchhoff_p2g(x, v, C,  dg, grid_v, grid_m, mu_0, lambda_0, p_vol, p_mass, dx, dt)
        substep_neohookean_p2g(x, v, C,  dg, grid_v, grid_m, mu_0, lambda_0, p_vol, p_mass, dx, dt)
        substep_calculate_signed_distance_field(obstacle_pos, obstacle_velocity ,sdf,grid_obstacle_vel,obstacle_radius,k,dx,dt)
        substep_calculate_hand_sdf(skeleton_segments, hand_sdf, skeleton_capsule_radius, dx)
        substep_update_grid_v(grid_v, grid_m,sdf,grid_obstacle_vel,gx,gy,gz,v_allowed,dt)
        substep_g2p(x, v, C,  grid_v, dx, dt)
        substep_apply_Von_Mises_plasticity(dg, mu_0, SigY)
        substep_apply_clamp_plasticity(dg, 0.1,0.1)
        substep_apply_Drucker_Prager_plasticity(dg, lambda_0, mu_0, alpha)

    def run_aot():
        mod = ti.aot.Module(arch)
        mod.add_kernel(substep_reset_grid, template_args={'grid_v': grid_v, 'grid_m': grid_m})
        mod.add_kernel(substep_kirchhoff_p2g, template_args={'x': x, 'v': v, 'C': C,  'dg': dg, 'grid_v': grid_v, 'grid_m': grid_m})  
        mod.add_kernel(substep_neohookean_p2g, template_args={'x': x, 'v': v, 'C': C,  'dg': dg, 'grid_v': grid_v, 'grid_m': grid_m})
        mod.add_kernel(substep_g2p, template_args={'x': x, 'v': v, 'C': C, 'grid_v': grid_v})
        mod.add_kernel(substep_apply_Drucker_Prager_plasticity, template_args={'dg': dg})
        mod.add_kernel(substep_apply_Von_Mises_plasticity, template_args={'dg': dg})
        mod.add_kernel(substep_apply_clamp_plasticity, template_args={'dg': dg,})
        mod.add_kernel(init_particles, template_args={'x': x, 'v': v, 'dg': dg})
        mod.add_kernel(substep_calculate_signed_distance_field, template_args={'obstacle_pos': obstacle_pos, 'obstacle_velocity': obstacle_velocity, 'sdf': sdf, 'grid_obstacle_vel': grid_obstacle_vel, 'obstacle_radius': obstacle_radius})
        mod.add_kernel(substep_update_grid_v, template_args={'grid_v': grid_v, 'grid_m': grid_m, 'sdf': sdf, 'grid_obstacle_vel': grid_obstacle_vel})
        
        mod.add_kernel(substep_calculate_hand_sdf, template_args={'skeleton_segments': skeleton_segments, 'hand_sdf': hand_sdf, 'skeleton_capsule_radius': skeleton_capsule_radius})
        
        mod.archive("Assets/Resources/TaichiModules/mpm3DVonmisesSDF.kernel.tcm")
        print("AOT done")
    
    if run:
        with open('./scripts/HandSkeletonSegments.json', 'r') as file:
            data = json.load(file)
        segments_list = []
        for segment in data['skeleton_segments']:
            segments_list.append([
                [segment['start']['x'], segment['start']['y'], segment['start']['z']],
                [segment['end']['x'], segment['end']['y'], segment['end']['z']]
            ])
        segments_np = np.array(segments_list, dtype=np.float32)
        skeleton_segments.from_numpy(segments_np)
        substep_calculate_hand_sdf(skeleton_segments, hand_sdf, skeleton_capsule_radius, dx)
        print(hand_sdf.to_numpy())
        
        gui = ti.GUI('MPM3D', res=(800, 800))
        init_particles(x, v, dg,cube_size)
        while gui.running and not gui.get_event(gui.ESCAPE):
            for i in range(50):
                substep()
            gui.circles(T(x.to_numpy()), radius=1.5, color=0x66CCFF)
            gui.show()
    # run_aot()
    
if __name__ == "__main__":
    compile_for_cgraph = args.cgraph
    
    if args.arch == "vulkan":
        compile_mpm3D(arch=ti.vulkan, save_compute_graph=compile_for_cgraph, run=True)
    elif args.arch == "cuda":
        compile_mpm3D(arch=ti.cuda, save_compute_graph=compile_for_cgraph, run=True)
    elif args.arch == "x64":
        compile_mpm3D(arch=ti.x64, save_compute_graph=compile_for_cgraph, run=True)
    else:
        assert False
