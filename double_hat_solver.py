#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import json


def get_polar_point(p1, p2, r, theta):
    """
    (p1 - p2) vector is the x-axis, p2 is the origin. Find the point at a
    distance r at an angle CCW theta.
    """
    alpha = np.arctan2(p1[1] - p2[1], p1[0] - p2[0])
    beta = alpha + theta
    return np.array([p2[0] + r * np.cos(beta), p2[1] + r * np.sin(beta)])


def get_extended_point(p1, p2, l):
    """
    Find a point at a distance l from p2, away from p1 in the direction p2 - p1.
    """
    d = np.linalg.norm(p2 - p1)
    return p2 + l * (p2 - p1) / d


def get_slant_length(l1, l3, L, alpha1, theta1, theta2):
    """
    It is difficult to explain the convention here.
    """
    alpha2 = alpha1 + theta1 - np.pi
    alpha3 = theta2 - alpha2
    l2 = (L - l1*np.cos(alpha1) + l3*np.cos(alpha3)) / np.cos(alpha2)
    return l2


def get_arc_angles(center, start, end):
    """
    Given the start and end points of an arc, find the start and end angles.
    """
    xe = end[:, 0]
    ye = end[:, 1]
    xs = start[:, 0]
    ys = start[:, 1]
    xc = center[:, 0]
    yc = center[:, 1]
    # Calculate the start and end angles
    theta_s = np.arctan2(ys - yc, xs - xc)
    theta_s = np.where(theta_s < 0.0, 2*np.pi + theta_s, theta_s)
    theta_e = np.arctan2(ye - yc, xe - xc)
    theta_e = np.where(theta_e < 0.0, 2*np.pi + theta_e, theta_e)
    theta_e = np.where(theta_e < theta_s, 2*np.pi + theta_e, theta_e)
    return theta_s, theta_e


def validate_arcs(center, start, end):
    """
    Given the center, start and end points of circular arcs, validate them.
    """
    theta_s, theta_e = get_arc_angles(center, start, end)
    # Ensure that theta_e is always counter-clockwise ahead of theta_s
    arc_spans = np.where(theta_e < theta_s, 2*np.pi +
                         theta_e - theta_s, theta_e - theta_s)
    if np.any(arc_spans > np.pi):
        raise ValueError('Bad quality sketch detected:'
                         'An arc span is greater than 180 degrees.')


def get_arc_points(center, start, end, r, numpoints=11):
    """
    Calculate points on a circular arc from center,
    start point, and end point.
    """
    theta_s, theta_e = get_arc_angles(center, start, end)
    theta = np.linspace(theta_s, theta_e, numpoints)
    xc = center[:, 0]
    yc = center[:, 1]
    arc_x = xc + r * np.cos(theta)
    arc_y = yc + r * np.sin(theta)
    return arc_x, arc_y


def get_offset_line(p1, p2, t, ccw=True):
    """
    Given endpoints of a line, get endpoints of an offset line
    where the offset distance is t. `ccw` controls the direction
    of offset.
    """
    angle = 0.5*np.pi if ccw else -0.5*np.pi
    q1 = get_polar_point(p2, p1, t,  angle)
    q2 = get_polar_point(p1, p2, t, -angle)
    return q1, q2


def solve_cross_section(L, A, R, slant_length=False):
    """
    Given cross-section dimension constraints, find the coordinates
    of all points needed to construct arcs and lines.
    L : array of all lengths
    A : array of all angles
    R : array of all radii
    slant_length: if True, return a tuple of l1, l2, l3, l4.
    """
    theta = A * np.pi / 360.0
    r = R / np.tan(theta)
    h = R / np.sin(theta)

    # Get all the slant lengths
    l1 = get_slant_length(r[1], r[0] + L[0], L[5], 2*(theta[2] -
                          theta[3]) + 0.5*np.pi, 2*theta[1], 2*theta[0]) - r[1] - r[0]
    l2 = get_slant_length(r[3], L[1] + r[2], L[5] - L[7],
                          0.5*np.pi, 2*theta[3], 2*theta[2]) - r[3] - r[2]
    l3 = get_slant_length(r[4], r[5] + L[3], (L[8] + L[6] - L[7]),
                          0.5*np.pi, 2*theta[4], 2*theta[5]) - r[4] - r[5]
    l4 = get_slant_length(r[6], r[7] + L[4], L[6], 0.5*np.pi + 2 *
                          (theta[5] - theta[4]), 2*theta[6], 2*theta[7]) - r[6] - r[7]

    # We will set the bottom point of L[2] as the origin as it is guaranteed to be vertical.
    P10 = np.array([0.0, 0.0])
    P9 = np.array([0.0, L[2]])
    C7 = np.array([0.0, L[2] + r[3]])
    C8 = get_polar_point(P9, C7, h[3], theta[3])
    P8 = get_polar_point(P9, C7, r[3], 2*theta[3])
    P7 = get_extended_point(C7, P8, l2)
    C5 = get_extended_point(P8, P7, r[2])
    C6 = get_polar_point(P7, C5, h[2], -theta[2])
    P6 = get_polar_point(P7, C5, r[2], -2*theta[2])
    P5 = get_extended_point(C5, P6, L[1])
    C3 = get_extended_point(P6, P5, r[1])
    C4 = get_polar_point(P5, C3, h[1], -theta[1])
    P4 = get_polar_point(P5, C3, r[1], -2*theta[1])
    P3 = get_extended_point(C3, P4, l1)
    C1 = get_extended_point(P4, P3, r[0])
    C2 = get_polar_point(P3, C1, h[0], theta[0])
    P2 = get_polar_point(P3, C1, r[0], 2*theta[0])
    P1 = get_extended_point(C1, P2, L[0])

    # Solve the lower part of the cross-section
    C9 = get_extended_point(P9, P10, r[4])
    C10 = get_polar_point(P10, C9, h[4], -theta[4])
    P11 = get_polar_point(P10, C9, r[4], -2*theta[4])
    P12 = get_extended_point(C9, P11, l3)
    C11 = get_extended_point(P11, P12, r[5])
    C12 = get_polar_point(P12, C11, h[5], theta[5])
    P13 = get_polar_point(P12, C11, r[5], 2*theta[5])
    P14 = get_extended_point(C11, P13, L[3])
    C13 = get_extended_point(P13, P14, r[6])
    C14 = get_polar_point(P14, C13, h[6], theta[6])
    P15 = get_polar_point(P14, C13, r[6], 2*theta[6])
    P16 = get_extended_point(C13, P15, l4)
    C15 = get_extended_point(P15, P16, r[7])
    C16 = get_polar_point(P16, C15, h[7], -theta[7])
    P17 = get_polar_point(P16, C15, r[7], -2*theta[7])
    P18 = get_extended_point(C15, P17, L[4])

    P = np.vstack([P1, P2, P3, P4, P5, P6, P7, P8, P9, P10,
                  P11, P12, P13, P14, P15, P16, P17, P18])
    C = np.vstack([C1, C2, C3, C4, C5, C6, C7, C8, C9,
                  C10, C11, C12, C13, C14, C15, C16])

    # Translate all the points such that the point P18 is at (0, 0)
    P -= P18
    C -= P18

    if slant_length:
        return P, C, (l1, l2, l3, l4)
    else:
        return P, C


def get_profile(P, C, R, t, numpoints=11, validate=True):
    """
    Given the key points of the cross-section, calculate points on the
    cross-section profile.
    """
    if validate:
        # Validate the lines
        drop_in_y = P[1::2, 1] - P[0::2, 1]
        if np.any(drop_in_y > t):
            raise ValueError(
                'Bad quality sketch detected: An edge is pointing upwards.')
    arc_centers = C[1::2, :]
    arc_starts = np.vstack(
        [P[1], P[4], P[6], P[7],  P[9], P[12], P[14], P[15]])
    arc_ends = np.vstack([P[2], P[3], P[5], P[8], P[10], P[11], P[13], P[16]])
    if validate:
        # Validate the arcs
        validate_arcs(arc_centers, arc_starts, arc_ends)
    arc_xs, arc_ys = get_arc_points(
        arc_centers, arc_starts, arc_ends, R, numpoints)
    # Put all the points in a sequence
    profile = np.vstack([P[0],
                         np.c_[arc_xs[:, 0], arc_ys[:, 0]],
                         np.c_[arc_xs[::-1, 1], arc_ys[::-1, 1]],
                         np.c_[arc_xs[::-1, 2], arc_ys[::-1, 2]],
                         np.c_[arc_xs[:, 3], arc_ys[:, 3]],
                         np.c_[arc_xs[:, 4], arc_ys[:, 4]],
                         np.c_[arc_xs[::-1, 5], arc_ys[::-1, 5]],
                         np.c_[arc_xs[::-1, 6], arc_ys[::-1, 6]],
                         np.c_[arc_xs[:, 7], arc_ys[:, 7]],
                         P[17]])
    return profile


def get_offset_profile(t, P, C, R, numpoints=11, validate=True):
    """
    Get the right-offset profile.
    """
    if np.any(R < t):
        raise ValueError("Offset failed: R < t")

    Q0,  Q1 = get_offset_line(P[0],   P[1], t, ccw=True)
    Q2,  Q3 = get_offset_line(P[2],   P[3], t, ccw=True)
    Q4,  Q5 = get_offset_line(P[4],   P[5], t, ccw=True)
    Q6,  Q7 = get_offset_line(P[6],   P[7], t, ccw=False)
    Q8,  Q9 = get_offset_line(P[8],   P[9], t, ccw=True)
    Q10, Q11 = get_offset_line(P[10], P[11], t, ccw=True)
    Q12, Q13 = get_offset_line(P[12], P[13], t, ccw=True)
    Q14, Q15 = get_offset_line(P[14], P[15], t, ccw=False)
    Q16, Q17 = get_offset_line(P[16], P[17], t, ccw=True)

    Q = np.vstack([Q0, Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8, Q9,
                   Q10, Q11, Q12, Q13, Q14, Q15, Q16, Q17])

    right_R = R.copy()
    right_R[[0, 3, 4, 7]] -= t
    right_R[[1, 2, 5, 6]] += t
    right_profile = get_profile(Q, C, right_R, t, numpoints, validate)
    return right_profile


def get_cross_section_face(L, A, R, t, numpoints=11, validate=True):
    """
    Given the dimensions, get the cross-section closed-face.
    """
    # Calculate the points
    P, C = solve_cross_section(L, A, R)
    left_profile = get_profile(P, C, R, t, numpoints, validate)
    right_profile = get_offset_profile(t, P, C, R, numpoints, validate)[::-1]
    face = np.vstack([left_profile, right_profile, left_profile[0, :]])
    return face


def get_cross_section_area_volume(L, A, R, t):
    """
    Given the parameters of the cross-section, calculate its area.
    Also, calculate the volume of the full beam
    """
    # Get the point coordinates, and slant lengths
    P, C, slant_lengths = solve_cross_section(L, A, R, slant_length=True)
    # Calculate the arc start and end angles to calculate the spans
    arc_centers = C[1::2, :]
    arc_starts = np.vstack(
        [P[1], P[4], P[6], P[7],  P[9], P[12], P[14], P[15]])
    arc_ends = np.vstack([P[2], P[3], P[5], P[8], P[10], P[11], P[13], P[16]])
    theta_s, theta_e = get_arc_angles(arc_centers, arc_starts, arc_ends)
    # Calculate area of the linear segments
    linear_area = t * (np.sum(L[0:5]) + sum(slant_lengths))
    # Calculate area of the curved regions
    R1 = R[[0,3,4,7]]
    r2 = R[[1,2,5,6]]
    r1 = R1 - t
    R2 = r2 + t
    theta_1 = (theta_e - theta_s)[[0,3,4,7]]
    theta_2 = (theta_e - theta_s)[[1,2,5,6]]
    curved_area = np.sum(0.5 * theta_1 * (R1**2 - r1**2)) + np.sum(0.5 * theta_2 * (R2**2 - r2**2))
    area = linear_area + curved_area
    volume = area * L[-1]
    return area, volume


def plot_cross_section_from_features(L, A, R, t, numpoints=11, validate=True):
    """
    Given a JSON file with cross-section features, plot it.
    """
    face = get_cross_section_face(L, A, R, t, numpoints, validate)
    # Plot the points
    fig1, ax1 = plt.subplots()
    ax1.set_aspect('equal')
    ax1.set_axis_off()
    ax1.fill(face[:, 0], face[:, 1], facecolor='gray', edgecolor='k')
    fig1.set_tight_layout(True)
    plt.show()


def get_features_from_json(filename):
    """
    Given a JSON file, read the features into NumPy arrays.
    """
    with open(filename, 'r') as jfile:
        data = json.load(jfile)

    L = np.empty(10)
    for i in range(9):
        L[i] = data['cross_section'][f'L{i+1}']

    L[9] = data['cross_section']['L']

    A = np.empty(8)
    for i in range(8):
        A[i] = data['cross_section'][f'A{i+1}']

    R = np.empty(8)
    for i in range(8):
        R[i] = data['cross_section'][f'R{i+1}']

    t = data['cross_section']['t']

    return L, A, R, t


def plot_cross_section_from_json(filename, numpoints=11, validate=True):
    """
    Given a JSON file with cross-section features, plot it.
    """
    L, A, R, t = get_features_from_json(filename)
    plot_cross_section_from_features(L, A, R, t, numpoints, validate)


if __name__ == "__main__":
    # Calculate area
    L, A, R, t = get_features_from_json('biw_single_part/sample_data/input_dh_73.json')
    area, volume = get_cross_section_area_volume(L, A, R, t)
    print('Area calculated by our code is', area)
    print('Volume calculated by our code is', volume)

    # Plot the cross section in matplotlib
    plot_cross_section_from_json('biw_single_part/sample_data/input_dh_73.json')