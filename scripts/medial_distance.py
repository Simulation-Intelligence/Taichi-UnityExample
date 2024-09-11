import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

@ti.dataclass
class DistanceResult:
    distance: ti.types.vector(3, ti.f32)
    b: ti.f32

@ti.func
def calculate_point_segment_distance(point: ti.types.vector(3, ti.f32), start: ti.types.vector(3, ti.f32), end: ti.types.vector(3, ti.f32)) -> DistanceResult:
    v = end - start
    w = point - start
    c1 = w.dot(v)
    c2 = v.dot(v)
    b = 0.0
    distance = ti.Vector([0.0, 0.0, 0.0])
    if c1 <= 0:
        distance = (point - start)
    elif c1 >= c2:
        distance = (point - end)
        b = 1
    else:
        b = c1 / c2
        Pb = start + b * v
        distance = (point - Pb)
    return DistanceResult(distance=distance, b=b)

@ti.func
def inv_square(x):  # A Taichi function
    return 1.0 / (x * x)

@ti.kernel
def partial_sum(n: int) -> float:  # A kernel
    total = 0.0
    for i in range(1, n + 1):
        total += inv_square(n)
    return total

def initialize_medial_spheres():
    sphere_centers[0] = ti.Vector([0.0, 3.0, 0.0])
    sphere_centers[1] = ti.Vector([1.0, 3.0, 1.0])
    sphere_centers[2] = ti.Vector([2.0, 3.0, 0.0])
    sphere_centers[3] = ti.Vector([2.0, 0.0, 0.0])
    sphere_radii[0] = 0.5
    sphere_radii[1] = 0.5
    sphere_radii[2] = 0.5
    sphere_radii[3] = 0.5
    
@ti.func
def value_of_quadric_surface_2d(x, y, A, B, C, D, E, F):
    return A * x * x + B * x * y + C * y * y + D * x + E * y + F

@ti.kernel
def compute_sphere_cone_distance(sphere_centers: ti.types.ndarray(ndim=1),
                               sphere_radii: ti.types.ndarray(ndim=1),
                               alpha: ti.types.ndarray(ndim=1), 
                               beta: ti.types.ndarray(ndim=1)):
    # sphere_centers and sphere_radii should contain 4 elements
    sC1 = ti.Vector([0.0, 0.0, 0.0])
    sC2 = ti.Vector([0.0, 0.0, 0.0])
    sC3 = ti.Vector([0.0, 0.0, 0.0])
    for i in ti.static(range(3)):
        sC1[i] = sphere_centers[0][i] - sphere_centers[1][i]
        sC2[i] = sphere_centers[3][i] - sphere_centers[2][i]
        sC3[i] = sphere_centers[1][i] - sphere_centers[3][i]
    sR1 = sphere_radii[0] - sphere_radii[1]
    sR2 = sphere_radii[2] - sphere_radii[3]
    sR3 = sphere_radii[1] + sphere_radii[3]
    
    A = sC1.dot(sC1) - sR1 * sR1
    B = 2.0 * (sC1.dot(sC2) - sR1 * sR2)
    C = sC2.dot(sC2) - sR2 * sR2
    D = 2.0 * (sC1.dot(sC3) - sR1 * sR3)
    E = 2.0 * (sC2.dot(sC3) - sR2 * sR3)
    F = sC3.dot(sC3) - sR3 * sR3
    
    delta = 4 * A * C - B * B
    print("delta: ", delta)
    temp_alpha = 0.0
    temp_beta = 0.0
    
    alpha[0] = 0.0
    beta[0] = 0.0
    dist_f = value_of_quadric_surface_2d(alpha[0], beta[0], A, B, C, D, E, F)
    
    # parallel cases
    for temp_alpha, temp_beta in ti.static([(1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]):
        temp_dist = value_of_quadric_surface_2d(temp_alpha, temp_beta, A, B, C, D, E, F)
        if dist_f > temp_dist:
            dist_f = temp_dist
            alpha[0] = temp_alpha
            beta[0] = temp_beta
            print("case0: ", dist_f)
    
    # temp_alpha = 0, temp_beta = -E / (2.0 * C)
    temp_alpha = 0.0
    temp_beta = -E / (2.0 * C)
    if 0.0 < temp_beta < 1.0:
        temp_dist = value_of_quadric_surface_2d(temp_alpha, temp_beta, A, B, C, D, E, F)
        if dist_f > temp_dist:
            dist_f = temp_dist
            alpha[0] = temp_alpha
            beta[0] = temp_beta
            print("case1: ", dist_f)

    # temp_alpha = 1.0, temp_beta = -(B + E) / (2.0 * C)
    temp_alpha = 1.0
    temp_beta = -(B + E) / (2.0 * C)
    if 0.0 < temp_beta < 1.0:
        temp_dist = value_of_quadric_surface_2d(temp_alpha, temp_beta, A, B, C, D, E, F)
        if dist_f > temp_dist:
            dist_f = temp_dist
            alpha[0] = temp_alpha
            beta[0] = temp_beta
            print("case2: ", dist_f)
    
    # temp_alpha = -D / (2.0 * A), temp_beta = 0.0
    temp_alpha = -D / (2.0 * A)
    temp_beta = 0.0
    if 0.0 < temp_alpha < 1.0:
        temp_dist = value_of_quadric_surface_2d(temp_alpha, temp_beta, A, B, C, D, E, F)
        if dist_f > temp_dist:
            dist_f = temp_dist
            alpha[0] = temp_alpha
            beta[0] = temp_beta
            print("case3: ", dist_f)
    
    # temp_alpha = -(B + D) / (2.0 * A), temp_beta = 1.0
    temp_alpha = -(B + D) / (2.0 * A)
    temp_beta = 1.0
    if 0.0 < temp_alpha < 1.0:
        temp_dist = value_of_quadric_surface_2d(temp_alpha, temp_beta, A, B, C, D, E, F)
        if dist_f > temp_dist:
            dist_f = temp_dist
            alpha[0] = temp_alpha
            beta[0] = temp_beta
            print("case4: ", dist_f)
    
    # temp_alpha = (B * E - 2.0 * C * D) / delta, temp_beta = (B * D - 2.0 * A * E) / delta
    if delta != 0.0:
        temp_alpha = (B * E - 2.0 * C * D) / delta
        temp_beta = (B * D - 2.0 * A * E) / delta
        print(temp_alpha, temp_beta)
        if 0.0 < temp_alpha < 1.0 and 0.0 < temp_beta < 1.0:
            temp_dist = value_of_quadric_surface_2d(temp_alpha, temp_beta, A, B, C, D, E, F)
            if dist_f > temp_dist:
                dist_f = temp_dist
                alpha[0] = temp_alpha
                beta[0] = temp_beta
                print("case5: ", dist_f)

    # Compute the distance
    print("alpha: ", alpha[0], "beta: ", beta[0])
    cp = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
    cq = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
    for i in ti.static(range(3)):
        cp[i] = alpha[0] * sphere_centers[0][i] + (1.0 - alpha[0]) * sphere_centers[1][i]
        cq[i] = beta[0] * sphere_centers[2][i] + (1.0 - beta[0]) * sphere_centers[3][i]
    rp = alpha[0] * sphere_radii[0] + (1.0 - alpha[0]) * sphere_radii[1]
    rq = beta[0] * sphere_radii[2] + (1.0 - beta[0]) * sphere_radii[3]
    dir = cq - cp
    distance = dir.norm() - (rp + rq)
    normal = dir.normalized()
    print(distance, normal)

@ti.kernel
def compute_sphere_slab_distance(sphere_centers: ti.types.ndarray(ndim=1), 
                                 sphere_radii: ti.types.ndarray(ndim=1),
                                 alpha: ti.types.ndarray(ndim=1), 
                                 beta: ti.types.ndarray(ndim=1)):
    # sphere_centers and sphere_radii should contain 4 elements
    # The first three sphers compose a slab, and the last one is a sphere
    sC1 = ti.Vector([0.0, 0.0, 0.0])
    sC2 = ti.Vector([0.0, 0.0, 0.0])
    sC3 = ti.Vector([0.0, 0.0, 0.0])
    for i in ti.static(range(3)):
        sC1[i] = sphere_centers[0][i] - sphere_centers[2][i]
        sC2[i] = sphere_centers[1][i] - sphere_centers[2][i]
        sC3[i] = sphere_centers[2][i] - sphere_centers[3][i]
    sR1 = sphere_radii[0] - sphere_radii[2]
    sR2 = sphere_radii[1] - sphere_radii[2]
    sR3 = sphere_radii[2] + sphere_radii[3]
    
    A = sC1.dot(sC1) - sR1 * sR1
    B = 2.0 * (sC1.dot(sC2) - sR1 * sR2)
    C = sC2.dot(sC2) - sR2 * sR2
    D = 2.0 * (sC1.dot(sC3) - sR1 * sR3)
    E = 2.0 * (sC2.dot(sC3) - sR2 * sR3)
    F = sC3.dot(sC3) - sR3 * sR3
    
    delta = 4 * A * C - B * B
    print("delta: ", delta)
    temp_alpha = 0.0
    temp_beta = 0.0
    
    alpha[0] = 0.0
    beta[0] = 0.0
    dist_f = value_of_quadric_surface_2d(alpha[0], beta[0], A, B, C, D, E, F)
    
    # parallel cases
    for temp_alpha, temp_beta in ti.static([(1.0, 0.0), (0.0, 1.0)]):
        temp_dist = value_of_quadric_surface_2d(temp_alpha, temp_beta, A, B, C, D, E, F)
        if dist_f > temp_dist:
            dist_f = temp_dist
            alpha[0] = temp_alpha
            beta[0] = temp_beta
            print("case0: ", dist_f)
    
    # temp_alpha = 0, temp_beta = -E / (2.0 * C)
    temp_alpha = 0.0
    temp_beta = -E / (2.0 * C)
    if 0.0 < temp_beta < 1.0:
        temp_dist = value_of_quadric_surface_2d(temp_alpha, temp_beta, A, B, C, D, E, F)
        if dist_f > temp_dist:
            dist_f = temp_dist
            alpha[0] = temp_alpha
            beta[0] = temp_beta
            print("case1: ", dist_f)

    # temp_alpha = -D / (2.0 * A), temp_beta = 0.0
    temp_alpha = -D / (2.0 * A)
    temp_beta = 0.0
    if 0.0 < temp_alpha < 1.0:
        temp_dist = value_of_quadric_surface_2d(temp_alpha, temp_beta, A, B, C, D, E, F)
        if dist_f > temp_dist:
            dist_f = temp_dist
            alpha[0] = temp_alpha
            beta[0] = temp_beta
            print("case3: ", dist_f)
    
    temp_alpha = 0.5 * (2.0 * C + E - B - D) / (A - B + C)
    temp_beta = 1.0 - temp_alpha
    if 0.0 < temp_alpha < 1.0:
        temp_dist = value_of_quadric_surface_2d(temp_alpha, temp_beta, A, B, C, D, E, F)
        if dist_f > temp_dist:
            dist_f = temp_dist
            alpha[0] = temp_alpha
            beta[0] = temp_beta
            print("case4: ", dist_f)
    
    # can be ignored
    if delta != 0.0:
        temp_alpha = (B * E - 2.0 * C * D) / delta
        temp_beta = (B * D - 2.0 * A * E) / delta
        print(temp_alpha, temp_beta)
        if 0.0 < temp_alpha < 1.0 and 0.0 < temp_beta < 1.0 and temp_alpha + temp_beta < 1.0:
            temp_dist = value_of_quadric_surface_2d(temp_alpha, temp_beta, A, B, C, D, E, F)
            if dist_f > temp_dist:
                dist_f = temp_dist
                alpha[0] = temp_alpha
                beta[0] = temp_beta
                print("case5: ", dist_f)

    # Compute the distance
    print("alpha: ", alpha[0], "beta: ", beta[0])
    cp = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
    cq = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
    for i in ti.static(range(3)):
        cp[i] = alpha[0] * sphere_centers[0][i] + beta[0] * sphere_centers[1][i] + (1.0 - alpha[0] - beta[0]) * sphere_centers[2][i]
        cq[i] = sphere_centers[3][i]

    rp = alpha[0] * sphere_radii[0] + beta[0] * sphere_radii[1] + (1.0 - alpha[0] - beta[0]) * sphere_radii[2]
    rq = sphere_radii[3]
    dir = cq - cp
    distance = dir.norm() - (rp + rq)
    normal = dir.normalized()
    print(distance, normal)


sphere_centers = ti.Vector.ndarray(3, ti.f32, shape=(4,))
sphere_radii = ti.ndarray(ti.f32, shape=(4,))
alpha = ti.ndarray(ti.f32, shape=(1,))
beta = ti.ndarray(ti.f32, shape=(1,))

# test
initialize_medial_spheres()
compute_sphere_cone_distance(sphere_centers, sphere_radii, alpha, beta)
compute_sphere_slab_distance(sphere_centers, sphere_radii, alpha, beta)