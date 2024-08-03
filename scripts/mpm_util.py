import taichi as ti
import numpy as np
@ti.func
def PackSmallest3Rotation(q):
    # find biggest component
    absQ = ti.abs(q)
    index = 0
    maxV = absQ[0]
    if absQ[1] > maxV:
        index = 1
        maxV = absQ[1]
    if absQ[2] > maxV:
        index = 2
        maxV = absQ[2]
    if absQ[3] > maxV:
        index = 3
        maxV = absQ[3]
    if index == 0:
        q = ti.Vector([q[1], q[2], q[3], q[0]])
    elif index == 1:
        q = ti.Vector([q[0], q[2], q[3], q[1]])
    elif index == 2:
        q = ti.Vector([q[0], q[1], q[3], q[2]])
    three = q.xyz * (1 if q[3] >= 0 else -1)
    three = (three * ti.sqrt(2.0)) * 0.5 + 0.5
    return ti.Vector([three[0], three[1], three[2], index / 3.0])
    
@ti.func
def EncodeQuatToNorm10(v)->ti.u32:
    return (ti.cast(v[0] * 1023.5, ti.u32) |
            (ti.cast(v[1] * 1023.5, ti.u32) << 10) |
            (ti.cast(v[2] * 1023.5, ti.u32) << 20) |
            (ti.cast(v[3] * 3.5, ti.u32) << 30))

@ti.func
def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1[3], q1[0], q1[1], q1[2]
    w2, x2, y2, z2 = q2[3], q2[0], q2[1], q2[2]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return ti.Vector([x, y, z, w])

@ti.func
def DecodePacked_10_10_10_2(enc: ti.u32)->ti.Vector:
    return ti.Vector([
        (enc & 1023) / 1023.0,
        ((enc >> 10) & 1023) / 1023.0,
        ((enc >> 20) & 1023) / 1023.0,
        ((enc >> 30) & 3) / 3.0
    ])

@ti.func
def DecodeRotation(pq):
    idx = ti.cast(ti.round(pq[3] * 3.0), ti.u32)
    q = ti.Vector([
        pq[0] * ti.sqrt(2.0) - (1.0 / ti.sqrt(2.0)),
        pq[1] * ti.sqrt(2.0) - (1.0 / ti.sqrt(2.0)),
        pq[2] * ti.sqrt(2.0) - (1.0 / ti.sqrt(2.0)),
        0.0
    ])
    q[3] = ti.sqrt(1.0 - ti.min(1.0, q[0]**2 + q[1]**2 + q[2]**2))
    if idx == 0:
        q = ti.Vector([q[3], q[0], q[1], q[2]])
    elif idx == 1:
        q = ti.Vector([q[0], q[3], q[1], q[2]])
    elif idx == 2:
        q = ti.Vector([q[0], q[1], q[3], q[2]])
    return q


@ti.func
def rotation_matrix_to_quaternion(R):
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
    if trace > 0:
        s = 0.5 / ti.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * ti.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * ti.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * ti.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
    return ti.Vector([qx, qy, qz, qw])

# 常数
kSqrt03_02 = np.sqrt(3.0 / 2.0)
kSqrt01_03 = np.sqrt(1.0 / 3.0)
kSqrt02_03 = np.sqrt(2.0 / 3.0)
kSqrt04_03 = np.sqrt(4.0 / 3.0)
kSqrt01_04 = np.sqrt(1.0 / 4.0)
kSqrt03_04 = np.sqrt(3.0 / 4.0)
kSqrt01_05 = np.sqrt(1.0 / 5.0)
kSqrt03_05 = np.sqrt(3.0 / 5.0)
kSqrt06_05 = np.sqrt(6.0 / 5.0)
kSqrt08_05 = np.sqrt(8.0 / 5.0)
kSqrt09_05 = np.sqrt(9.0 / 5.0)
kSqrt05_06 = np.sqrt(5.0 / 6.0)
kSqrt01_06 = np.sqrt(1.0 / 6.0)
kSqrt03_08 = np.sqrt(3.0 / 8.0)
kSqrt05_08 = np.sqrt(5.0 / 8.0)
kSqrt09_08 = np.sqrt(9.0 / 8.0)
kSqrt05_09 = np.sqrt(5.0 / 9.0)
kSqrt08_09 = np.sqrt(8.0 / 9.0)
kSqrt01_10 = np.sqrt(1.0 / 10.0)
kSqrt03_10 = np.sqrt(3.0 / 10.0)
kSqrt01_12 = np.sqrt(1.0 / 12.0)
kSqrt04_15 = np.sqrt(4.0 / 15.0)
kSqrt01_16 = np.sqrt(1.0 / 16.0)
kSqrt15_16 = np.sqrt(15.0 / 16.0)
kSqrt01_18 = np.sqrt(1.0 / 18.0)

@ti.func
def Dot3(srcIdx, coeffsIn, vec):
    return ti.Vector([
        coeffsIn[srcIdx,0] * vec[0] + coeffsIn[srcIdx + 1,0] * vec[1] + coeffsIn[srcIdx + 2,0] * vec[2],
        coeffsIn[srcIdx,1] * vec[0] + coeffsIn[srcIdx + 1,1] * vec[1] + coeffsIn[srcIdx + 2,1] * vec[2],
        coeffsIn[srcIdx,2] * vec[0] + coeffsIn[srcIdx + 1,2] * vec[1] + coeffsIn[srcIdx + 2,2] * vec[2]
    ])

@ti.func
def Dot5(srcIdx, coeffsIn, vec):
    return ti.Vector([
        coeffsIn[srcIdx,0] * vec[0] + coeffsIn[srcIdx + 1,0] * vec[1] + coeffsIn[srcIdx + 2,0] * vec[2] + coeffsIn[srcIdx + 3,0] * vec[3] + coeffsIn[srcIdx + 4,0] * vec[4],
        coeffsIn[srcIdx,1] * vec[0] + coeffsIn[srcIdx + 1,1] * vec[1] + coeffsIn[srcIdx + 2,1] * vec[2] + coeffsIn[srcIdx + 3,1] * vec[3] + coeffsIn[srcIdx + 4,1] * vec[4],
        coeffsIn[srcIdx,2] * vec[0] + coeffsIn[srcIdx + 1,2] * vec[1] + coeffsIn[srcIdx + 2,2] * vec[2] + coeffsIn[srcIdx + 3,2] * vec[3] + coeffsIn[srcIdx + 4,2] * vec[4]
    ])

@ti.func
def Dot7(srcIdx, coeffsIn, vec):
    return ti.Vector([
        coeffsIn[srcIdx,0] * vec[0] + coeffsIn[srcIdx + 1,0] * vec[1] + coeffsIn[srcIdx + 2,0] * vec[2] + coeffsIn[srcIdx + 3,0] * vec[3] + coeffsIn[srcIdx + 4,0] * vec[4] + coeffsIn[srcIdx + 5,0] * vec[5] + coeffsIn[srcIdx + 6,0] * vec[6],
        coeffsIn[srcIdx,1] * vec[0] + coeffsIn[srcIdx + 1,1] * vec[1] + coeffsIn[srcIdx + 2,1] * vec[2] + coeffsIn[srcIdx + 3,1] * vec[3] + coeffsIn[srcIdx + 4,1] * vec[4] + coeffsIn[srcIdx + 5,1] * vec[5] + coeffsIn[srcIdx + 6,1] * vec[6],
        coeffsIn[srcIdx,2] * vec[0] + coeffsIn[srcIdx + 1,2] * vec[1] + coeffsIn[srcIdx + 2,2] * vec[2] + coeffsIn[srcIdx + 3,2] * vec[3] + coeffsIn[srcIdx + 4,2] * vec[4] + coeffsIn[srcIdx + 5,2] * vec[5] + coeffsIn[srcIdx + 6,2] * vec[6]
    ])

@ti.func
def RotateSH(orient: ti.types.matrix(3, 3, float), coeffsIn: ti.types.matrix(16, 3, float)) -> ti.types.matrix(16, 3, float):
    coeffs = ti.Matrix.zero(float, 16, 3)

    ## no band 0
    #coeffs[0, :] = coeffsIn[0, :]
    #if 4 < 2:
    #    return coeffs

    # band 1
    sh1 = ti.Matrix.zero(float, 3, 3)
    sh1[0, 0] = orient[1, 1]
    sh1[0, 1] = -orient[1, 2]
    sh1[0, 2] = orient[1, 0]
    sh1[1, 0] = -orient[2, 1]
    sh1[1, 1] = orient[2, 2]
    sh1[1, 2] = -orient[2, 0]
    sh1[2, 0] = orient[0, 1]
    sh1[2, 1] = -orient[0, 2]
    sh1[2, 2] = orient[0, 0]

    srcIdx = 0
    dstIdx = 0
    coeffs[dstIdx, :] = Dot3(srcIdx, coeffsIn, sh1[0, :])
    dstIdx += 1
    coeffs[dstIdx, :] = Dot3(srcIdx, coeffsIn, sh1[1, :])
    dstIdx += 1
    coeffs[dstIdx, :] = Dot3(srcIdx, coeffsIn, sh1[2, :])
    dstIdx += 1


    # band 2
    srcIdx += 3
    sh2 = ti.Matrix.zero(float, 5, 5)

    sh2[0, 0] = kSqrt01_04 * ((sh1[2, 2] * sh1[0, 0] + sh1[2, 0] * sh1[0, 2]) + (sh1[0, 2] * sh1[2, 0] + sh1[0, 0] * sh1[2, 2]))
    sh2[0, 1] = (sh1[2, 1] * sh1[0, 0] + sh1[0, 1] * sh1[2, 0])
    sh2[0, 2] = kSqrt03_04 * (sh1[2, 1] * sh1[0, 1] + sh1[0, 1] * sh1[2, 1])
    sh2[0, 3] = (sh1[2, 1] * sh1[0, 2] + sh1[0, 1] * sh1[2, 2])
    sh2[0, 4] = kSqrt01_04 * ((sh1[2, 2] * sh1[0, 2] - sh1[2, 0] * sh1[0, 0]) + (sh1[0, 2] * sh1[2, 2] - sh1[0, 0] * sh1[2, 0]))

    coeffs[dstIdx, :] = Dot5(srcIdx, coeffsIn, sh2[0, :])
    dstIdx += 1

    sh2[1, 0] = kSqrt01_04 * ((sh1[1, 2] * sh1[0, 0] + sh1[1, 0] * sh1[0, 2]) + (sh1[0, 2] * sh1[1, 0] + sh1[0, 0] * sh1[1, 2]))
    sh2[1, 1] = sh1[1, 1] * sh1[0, 0] + sh1[0, 1] * sh1[1, 0]
    sh2[1, 2] = kSqrt03_04 * (sh1[1, 1] * sh1[0, 1] + sh1[0, 1] * sh1[1, 1])
    sh2[1, 3] = sh1[1, 1] * sh1[0, 2] + sh1[0, 1] * sh1[1, 2]
    sh2[1, 4] = kSqrt01_04 * ((sh1[1, 2] * sh1[0, 2] - sh1[1, 0] * sh1[0, 0]) + (sh1[0, 2] * sh1[1, 2] - sh1[0, 0] * sh1[1, 0]))

    coeffs[dstIdx, :] = Dot5(srcIdx, coeffsIn, sh2[1, :])
    dstIdx += 1

    sh2[2, 0] = kSqrt01_03 * (sh1[1, 2] * sh1[1, 0] + sh1[1, 0] * sh1[1, 2]) + -kSqrt01_12 * ((sh1[2, 2] * sh1[2, 0] + sh1[2, 0] * sh1[2, 2]) + (sh1[0, 2] * sh1[0, 0] + sh1[0, 0] * sh1[0, 2]))
    sh2[2, 1] = kSqrt04_03 * sh1[1, 1] * sh1[1, 0] + -kSqrt01_03 * (sh1[2, 1] * sh1[2, 0] + sh1[0, 1] * sh1[0, 0])
    sh2[2, 2] = sh1[1, 1] * sh1[1, 1] + -kSqrt01_04 * (sh1[2, 1] * sh1[2, 1] + sh1[0, 1] * sh1[0, 1])
    sh2[2, 3] = kSqrt04_03 * sh1[1, 1] * sh1[1, 2] + -kSqrt01_03 * (sh1[2, 1] * sh1[2, 2] + sh1[0, 1] * sh1[0, 2])
    sh2[2, 4] = kSqrt01_03 * (sh1[1, 2] * sh1[1, 2] - sh1[1, 0] * sh1[1, 0]) + -kSqrt01_12 * ((sh1[2, 2] * sh1[2, 2] - sh1[2, 0] * sh1[2, 0]) + (sh1[0, 2] * sh1[0, 2] - sh1[0, 0] * sh1[0, 0]))

    coeffs[dstIdx, :] = Dot5(srcIdx, coeffsIn, sh2[2, :])
    dstIdx += 1

    sh2[3, 0] = kSqrt01_04 * ((sh1[1, 2] * sh1[2, 0] + sh1[1, 0] * sh1[2, 2]) + (sh1[2, 2] * sh1[1, 0] + sh1[2, 0] * sh1[1, 2]))
    sh2[3, 1] = sh1[1, 1] * sh1[2, 0] + sh1[2, 1] * sh1[1, 0]
    sh2[3, 2] = kSqrt03_04 * (sh1[1, 1] * sh1[2, 1] + sh1[2, 1] * sh1[1, 1])
    sh2[3, 3] = sh1[1, 1] * sh1[2, 2] + sh1[2, 1] * sh1[1, 2]
    sh2[3, 4] = kSqrt01_04 * ((sh1[1, 2] * sh1[2, 2] - sh1[1, 0] * sh1[2, 0]) + (sh1[2, 2] * sh1[1, 2] - sh1[2, 0] * sh1[1, 0]))

    coeffs[dstIdx, :] = Dot5(srcIdx, coeffsIn, sh2[3, :])
    dstIdx += 1

    sh2[4, 0] = kSqrt01_04 * ((sh1[2, 2] * sh1[2, 0] + sh1[2, 0] * sh1[2, 2]) - (sh1[0, 2] * sh1[0, 0] + sh1[0, 0] * sh1[0, 2]))
    sh2[4, 1] = (sh1[2, 1] * sh1[2, 0] - sh1[0, 1] * sh1[0, 0])
    sh2[4, 2] = kSqrt03_04 * (sh1[2, 1] * sh1[2, 1] - sh1[0, 1] * sh1[0, 1])
    sh2[4, 3] = (sh1[2, 1] * sh1[2, 2] - sh1[0, 1] * sh1[0, 2])
    sh2[4, 4] = kSqrt01_04 * ((sh1[2, 2] * sh1[2, 2] - sh1[2, 0] * sh1[2, 0]) - (sh1[0, 2] * sh1[0, 2] - sh1[0, 0] * sh1[0, 0]))

    coeffs[dstIdx, :] = Dot5(srcIdx, coeffsIn, sh2[4, :])
    dstIdx += 1


    # band 3
    srcIdx += 5
    sh3 = ti.Matrix.zero(float, 7, 7)

    sh3[0, 0] = kSqrt01_04 * ((sh1[2, 2] * sh2[0, 0] + sh1[2, 0] * sh2[0, 4]) + (sh1[0, 2] * sh2[4, 0] + sh1[0, 0] * sh2[4, 4]))
    sh3[0, 1] = kSqrt03_02 * (sh1[2, 1] * sh2[0, 0] + sh1[0, 1] * sh2[4, 0])
    sh3[0, 2] = kSqrt15_16 * (sh1[2, 1] * sh2[0, 1] + sh1[0, 1] * sh2[4, 1])
    sh3[0, 3] = kSqrt05_06 * (sh1[2, 1] * sh2[0, 2] + sh1[0, 1] * sh2[4, 2])
    sh3[0, 4] = kSqrt15_16 * (sh1[2, 1] * sh2[0, 3] + sh1[0, 1] * sh2[4, 3])
    sh3[0, 5] = kSqrt03_02 * (sh1[2, 1] * sh2[0, 4] + sh1[0, 1] * sh2[4, 4])
    sh3[0, 6] = kSqrt01_04 * ((sh1[2, 2] * sh2[0, 4] - sh1[2, 0] * sh2[0, 0]) + (sh1[0, 2] * sh2[4, 4] - sh1[0, 0] * sh2[4, 0]))

    coeffs[dstIdx, :] = Dot7(srcIdx, coeffsIn, sh3[0, :])
    dstIdx += 1

    sh3[1, 0] = kSqrt01_06 * (sh1[1, 2] * sh2[0, 0] + sh1[1, 0] * sh2[0, 4]) + kSqrt01_06 * ((sh1[2, 2] * sh2[1, 0] + sh1[2, 0] * sh2[1, 4]) + (sh1[0, 2] * sh2[3, 0] + sh1[0, 0] * sh2[3, 4]))
    sh3[1, 1] = sh1[1, 1] * sh2[0, 0] + (sh1[2, 1] * sh2[1, 0] + sh1[0, 1] * sh2[3, 0])
    sh3[1, 2] = kSqrt05_08 * sh1[1, 1] * sh2[0, 1] + kSqrt05_08 * (sh1[2, 1] * sh2[1, 1] + sh1[0, 1] * sh2[3, 1])
    sh3[1, 3] = kSqrt05_09 * sh1[1, 1] * sh2[0, 2] + kSqrt05_09 * (sh1[2, 1] * sh2[1, 2] + sh1[0, 1] * sh2[3, 2])
    sh3[1, 4] = kSqrt05_08 * sh1[1, 1] * sh2[0, 3] + kSqrt05_08 * (sh1[2, 1] * sh2[1, 3] + sh1[0, 1] * sh2[3, 3])
    sh3[1, 5] = sh1[1, 1] * sh2[0, 4] + (sh1[2, 1] * sh2[1, 4] + sh1[0, 1] * sh2[3, 4])
    sh3[1, 6] = kSqrt01_06 * (sh1[1, 2] * sh2[0, 4] - sh1[1, 0] * sh2[0, 0]) + kSqrt01_06 * ((sh1[2, 2] * sh2[1, 4] - sh1[2, 0] * sh2[1, 0]) + (sh1[0, 2] * sh2[3, 4] - sh1[0, 0] * sh2[3, 0]))

    coeffs[dstIdx, :] = Dot7(srcIdx, coeffsIn, sh3[1, :])
    dstIdx += 1

    sh3[2, 0] = kSqrt04_15 * (sh1[1, 2] * sh2[1, 0] + sh1[1, 0] * sh2[1, 4]) + kSqrt01_05 * (sh1[0, 2] * sh2[2, 0] + sh1[0, 0] * sh2[2, 4]) + -ti.sqrt(1.0 / 60.0) * ((sh1[2, 2] * sh2[0, 0] + sh1[2, 0] * sh2[0, 4]) - (sh1[0, 2] * sh2[4, 0] + sh1[0, 0] * sh2[4, 4]))
    sh3[2, 1] = kSqrt08_05 * sh1[1, 1] * sh2[1, 0] + kSqrt06_05 * sh1[0, 1] * sh2[2, 0] + -kSqrt01_10 * (sh1[2, 1] * sh2[0, 0] - sh1[0, 1] * sh2[4, 0])
    sh3[2, 2] = sh1[1, 1] * sh2[1, 1] + kSqrt03_04 * sh1[0, 1] * sh2[2, 1] + -kSqrt01_16 * (sh1[2, 1] * sh2[0, 1] - sh1[0, 1] * sh2[4, 1])
    sh3[2, 3] = kSqrt08_09 * sh1[1, 1] * sh2[1, 2] + kSqrt02_03 * sh1[0, 1] * sh2[2, 2] + -kSqrt01_18 * (sh1[2, 1] * sh2[0, 2] - sh1[0, 1] * sh2[4, 2])
    sh3[2, 4] = sh1[1, 1] * sh2[1, 3] + kSqrt03_04 * sh1[0, 1] * sh2[2, 3] + -kSqrt01_16 * (sh1[2, 1] * sh2[0, 3] - sh1[0, 1] * sh2[4, 3])
    sh3[2, 5] = kSqrt08_05 * sh1[1, 1] * sh2[1, 4] + kSqrt06_05 * sh1[0, 1] * sh2[2, 4] + -kSqrt01_10 * (sh1[2, 1] * sh2[0, 4] - sh1[0, 1] * sh2[4, 4])
    sh3[2, 6] = kSqrt04_15 * (sh1[1, 2] * sh2[1, 4] - sh1[1, 0] * sh2[1, 0]) + kSqrt01_05 * (sh1[0, 2] * sh2[2, 4] - sh1[0, 0] * sh2[2, 0]) + -ti.sqrt(1.0 / 60.0) * ((sh1[2, 2] * sh2[0, 4] - sh1[2, 0] * sh2[0, 0]) - (sh1[0, 2] * sh2[4, 4] - sh1[0, 0] * sh2[4, 0]))

    coeffs[dstIdx, :] = Dot7(srcIdx, coeffsIn, sh3[2, :])
    dstIdx += 1

    sh3[3, 0] = kSqrt03_10 * (sh1[1, 2] * sh2[2, 0] + sh1[1, 0] * sh2[2, 4]) + -kSqrt01_10 * ((sh1[2, 2] * sh2[3, 0] + sh1[2, 0] * sh2[3, 4]) + (sh1[0, 2] * sh2[1, 0] + sh1[0, 0] * sh2[1, 4]))
    sh3[3, 1] = kSqrt09_05 * sh1[1, 1] * sh2[2, 0] + -kSqrt03_05 * (sh1[2, 1] * sh2[3, 0] + sh1[0, 1] * sh2[1, 0])
    sh3[3, 2] = kSqrt09_08 * sh1[1, 1] * sh2[2, 1] + -kSqrt03_08 * (sh1[2, 1] * sh2[3, 1] + sh1[0, 1] * sh2[1, 1])
    sh3[3, 3] = sh1[1, 1] * sh2[2, 2] + -kSqrt01_03 * (sh1[2, 1] * sh2[3, 2] + sh1[0, 1] * sh2[1, 2])
    sh3[3, 4] = kSqrt09_08 * sh1[1, 1] * sh2[2, 3] + -kSqrt03_08 * (sh1[2, 1] * sh2[3, 3] + sh1[0, 1] * sh2[1, 3])
    sh3[3, 5] = kSqrt09_05 * sh1[1, 1] * sh2[2, 4] + -kSqrt03_05 * (sh1[2, 1] * sh2[3, 4] + sh1[0, 1] * sh2[1, 4])
    sh3[3, 6] = kSqrt03_10 * (sh1[1, 2] * sh2[2, 4] - sh1[1, 0] * sh2[2, 0]) + -kSqrt01_10 * ((sh1[2, 2] * sh2[3, 4] - sh1[2, 0] * sh2[3, 0]) + (sh1[0, 2] * sh2[1, 4] - sh1[0, 0] * sh2[1, 0]))

    coeffs[dstIdx, :] = Dot7(srcIdx, coeffsIn, sh3[3, :])
    dstIdx += 1

    sh3[4, 0] = kSqrt04_15 * (sh1[1, 2] * sh2[3, 0] + sh1[1, 0] * sh2[3, 4]) + kSqrt01_05 * (sh1[2, 2] * sh2[2, 0] + sh1[2, 0] * sh2[2, 4]) + -ti.sqrt(1.0 / 60.0) * ((sh1[2, 2] * sh2[4, 0] + sh1[2, 0] * sh2[4, 4]) + (sh1[0, 2] * sh2[0, 0] + sh1[0, 0] * sh2[0, 4]))
    sh3[4, 1] = kSqrt08_05 * sh1[1, 1] * sh2[3, 0] + kSqrt06_05 * sh1[2, 1] * sh2[2, 0] + -kSqrt01_10 * (sh1[2, 1] * sh2[4, 0] + sh1[0, 1] * sh2[0, 0])
    sh3[4, 2] = sh1[1, 1] * sh2[3, 1] + kSqrt03_04 * sh1[2, 1] * sh2[2, 1] + -kSqrt01_16 * (sh1[2, 1] * sh2[4, 1] + sh1[0, 1] * sh2[0, 1])
    sh3[4, 3] = kSqrt08_09 * sh1[1, 1] * sh2[3, 2] + kSqrt02_03 * sh1[2, 1] * sh2[2, 2] + -kSqrt01_18 * (sh1[2, 1] * sh2[4, 2] + sh1[0, 1] * sh2[0, 2])
    sh3[4, 4] = sh1[1, 1] * sh2[3, 3] + kSqrt03_04 * sh1[2, 1] * sh2[2, 3] + -kSqrt01_16 * (sh1[2, 1] * sh2[4, 3] + sh1[0, 1] * sh2[0, 3])
    sh3[4, 5] = kSqrt08_05 * sh1[1, 1] * sh2[3, 4] + kSqrt06_05 * sh1[2, 1] * sh2[2, 4] + -kSqrt01_10 * (sh1[2, 1] * sh2[4, 4] + sh1[0, 1] * sh2[0, 4])
    sh3[4, 6] = kSqrt04_15 * (sh1[1, 2] * sh2[3, 4] - sh1[1, 0] * sh2[3, 0]) + kSqrt01_05 * (sh1[2, 2] * sh2[2, 4] - sh1[2, 0] * sh2[2, 0]) + -ti.sqrt(1.0 / 60.0) * ((sh1[2, 2] * sh2[4, 4] - sh1[2, 0] * sh2[4, 0]) + (sh1[0, 2] * sh2[0, 4] - sh1[0, 0] * sh2[0, 0]))

    coeffs[dstIdx, :] = Dot7(srcIdx, coeffsIn, sh3[4, :])
    dstIdx += 1

    sh3[5, 0] = kSqrt01_06 * (sh1[1, 2] * sh2[4, 0] + sh1[1, 0] * sh2[4, 4]) + kSqrt01_06 * ((sh1[2, 2] * sh2[3, 0] + sh1[2, 0] * sh2[3, 4]) - (sh1[0, 2] * sh2[1, 0] + sh1[0, 0] * sh2[1, 4]))
    sh3[5, 1] = sh1[1, 1] * sh2[4, 0] + (sh1[2, 1] * sh2[3, 0] - sh1[0, 1] * sh2[1, 0])
    sh3[5, 2] = kSqrt05_08 * sh1[1, 1] * sh2[4, 1] + kSqrt05_08 * (sh1[2, 1] * sh2[3, 1] - sh1[0, 1] * sh2[1, 1])
    sh3[5, 3] = kSqrt05_09 * sh1[1, 1] * sh2[4, 2] + kSqrt05_09 * (sh1[2, 1] * sh2[3, 2] - sh1[0, 1] * sh2[1, 2])
    sh3[5, 4] = kSqrt05_08 * sh1[1, 1] * sh2[4, 3] + kSqrt05_08 * (sh1[2, 1] * sh2[3, 3] - sh1[0, 1] * sh2[1, 3])
    sh3[5, 5] = sh1[1, 1] * sh2[4, 4] + (sh1[2, 1] * sh2[3, 4] - sh1[0, 1] * sh2[1, 4])
    sh3[5, 6] = kSqrt01_06 * (sh1[1, 2] * sh2[4, 4] - sh1[1, 0] * sh2[4, 0]) + kSqrt01_06 * ((sh1[2, 2] * sh2[3, 4] - sh1[2, 0] * sh2[3, 0]) - (sh1[0, 2] * sh2[1, 4] - sh1[0, 0] * sh2[1, 0]))

    coeffs[dstIdx, :] = Dot7(srcIdx, coeffsIn, sh3[5, :])
    dstIdx += 1

    sh3[6, 0] = kSqrt01_04 * ((sh1[2, 2] * sh2[4, 0] + sh1[2, 0] * sh2[4, 4]) - (sh1[0, 2] * sh2[0, 0] + sh1[0, 0] * sh2[0, 4]))
    sh3[6, 1] = kSqrt03_02 * (sh1[2, 1] * sh2[4, 0] - sh1[0, 1] * sh2[0, 0])
    sh3[6, 2] = kSqrt15_16 * (sh1[2, 1] * sh2[4, 1] - sh1[0, 1] * sh2[0, 1])
    sh3[6, 3] = kSqrt05_06 * (sh1[2, 1] * sh2[4, 2] - sh1[0, 1] * sh2[0, 2])
    sh3[6, 4] = kSqrt15_16 * (sh1[2, 1] * sh2[4, 3] - sh1[0, 1] * sh2[0, 3])
    sh3[6, 5] = kSqrt03_02 * (sh1[2, 1] * sh2[4, 4] - sh1[0, 1] * sh2[0, 4])
    sh3[6, 6] = kSqrt01_04 * ((sh1[2, 2] * sh2[4, 4] - sh1[2, 0] * sh2[4, 0]) - (sh1[0, 2] * sh2[0, 4] - sh1[0, 0] * sh2[0, 0]))

    coeffs[dstIdx, :] = Dot7(srcIdx, coeffsIn, sh3[6, :])
    dstIdx += 1

    return coeffs


# region - Hash Grid Functions
@ti.func
def get_hash(pos, n_grid):
    # use position to determine the hash key
    x_index = int(ti.floor(pos[0] * n_grid))
    y_index = int(ti.floor(pos[1] * n_grid))
    z_index = int(ti.floor(pos[2] * n_grid))
    return ti.Vector([x_index, y_index, z_index])
    
@ti.func
def min_bounding_box(seg_start, seg_end):
    return ti.Vector([min(seg_start[0], seg_end[0]),
                      min(seg_start[1], seg_end[1]),
                      min(seg_start[2], seg_end[2])])
    
@ti.func
def max_bounding_box(seg_start, seg_end):
    return ti.Vector([max(seg_start[0], seg_end[0]),
                      max(seg_start[1], seg_end[1]),
                      max(seg_start[2], seg_end[2])])
    
@ti.func
def insert_segments(skeleton_segments: ti.types.ndarray(ndim=2),
                    n_grid: ti.i32,
                    hash_table: ti.types.ndarray(ndim=4),
                    segments_count_per_cell: ti.types.ndarray(ndim=3)):
    for i in range(skeleton_segments.shape[0]):
        seg_start = skeleton_segments[i, 0]
        seg_end = skeleton_segments[i, 1]
        min_bb = min_bounding_box(seg_start, seg_end)
        max_bb = max_bounding_box(seg_start, seg_end)
        min_cell = get_hash(min_bb, n_grid)
        max_cell = get_hash(max_bb, n_grid)
        
        for x in range(min_cell[0], max_cell[0] + 1):
            if 0 <= x < n_grid:
                for y in range(min_cell[1], max_cell[1] + 1):
                    if 0 <= y < n_grid:
                        for z in range(min_cell[2], max_cell[2] + 1):
                            if 0 <= z < n_grid:
                                idx = ti.atomic_add(segments_count_per_cell[x, y, z], 1)
                                hash_table[x, y, z, idx] = i
    
@ti.func
def clear_hash_table(hash_table: ti.types.ndarray(ndim=4),
                     segments_count_per_cell: ti.types.ndarray(ndim=3)):
    # for i, j, k in ti.ndrange(n_grid, n_grid, n_grid):
    #     segments_count_per_cell[i, j, k] = 0
    #     for l in range(skeleton_segments.shape[0]):
    #         hash_table[i, j, k, l] = -1
    for I in ti.grouped(segments_count_per_cell):
        segments_count_per_cell[I] = 0
        for l in range(hash_table.shape[3]):
            hash_table[I[0], I[1], I[2], l] = -1
# endregion

@ti.dataclass
class DistanceResult:
    distance: ti.types.vector(3, ti.f32)
    b: ti.f32
    
@ti.func
def calculate_point_segment_distance(px: ti.f32, py: ti.f32, pz: ti.f32,
                                     sx: ti.f32, sy: ti.f32, sz: ti.f32,
                                     ex: ti.f32, ey: ti.f32, ez: ti.f32) -> DistanceResult:
    point = ti.Vector([px, py, pz])
    start = ti.Vector([sx, sy, sz])
    end = ti.Vector([ex, ey, ez])
    v = end - start
    w = point - start
    c1 = w.dot(v)
    c2 = v.dot(v)
    b=0.0
    distance = ti.Vector([0.0, 0.0, 0.0])
    if c1 <= 0:
        distance = (point - start)
    elif c1 >= c2:
        distance = (point - end)
        b=1
    else:
        b = c1 / c2
        Pb = start + b * v
        distance = (point - Pb)
    return DistanceResult(distance=distance, b=b)