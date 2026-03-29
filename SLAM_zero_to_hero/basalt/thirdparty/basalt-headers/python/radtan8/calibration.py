#
# BSD 3-Clause License
#
# This file is part of the Basalt project.
# https://gitlab.com/VladyslavUsenko/basalt.git
#
# Copyright (c) 2022, Collabora Ltd.
# All rights reserved.
#
# Author: Mateo de Mayo <mateo.demayo@collabora.com>
#

"""
Sets CALIB for global calibration parameters.
"""

ODYPLUS_CALIB = {
    "cx": 324.3333053588867,
    "cy": 245.22674560546875,
    "fx": 269.0600776672363,
    "fy": 269.1679859161377,
    "k1": 0.6257319450378418,
    "k2": 0.46612036228179932,
    "k3": 0.0041795829311013222,
    "k4": 0.89431935548782349,
    "k5": 0.54253977537155151,
    "k6": 0.06621214747428894,
    "p2": -4.2882973502855748e-5,
    "p1": -0.00018502399325370789,
    "metric_radius": 2.7941114902496338,
    "width": 640,
    "height": 480,
}

EUROC_CALIB = {
    "cx": 367.215,
    "cy": 248.375,
    "fx": 458.654,
    "fy": 457.296,
    "k1": -0.28340811,
    "k2": 0.07395907,
    "k3": 0,
    "k4": 0,
    "k5": 0,
    "k6": 0,
    "p2": 1.76187114e-05,
    "p1": 0.00019359,
    "width": 752,
    "height": 480,
}

CALIB = ODYPLUS_CALIB
# CALIB = EUROC_CALIB # Fails because of numerical precision issues

FX = CALIB["fx"]
FY = CALIB["fy"]
CX = CALIB["cx"]
CY = CALIB["cy"]
K1 = CALIB["k1"]
K2 = CALIB["k2"]
P1 = CALIB["p1"]
P2 = CALIB["p2"]
K3 = CALIB["k3"]
K4 = CALIB["k4"]
K5 = CALIB["k5"]
K6 = CALIB["k6"]
WIDTH = CALIB["width"]
HEIGHT = CALIB["height"]
METRIC_RADIUS = CALIB.get("metric_radius", 0)
