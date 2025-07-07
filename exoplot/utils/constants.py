LABELS = [r"$R_p / R_s$",
          r"Inclination (deg)", 
          r"$a/R_s$", 
          r"$t_0$"]

BOUNDS = [
    (0.001, 0.2),      # rp/rs
    (83, 89.9),      # inclination
    (5, 15),         # a/rs
    (-0.04, 0.04),   # t0
]

X0 = [0.28, 87, 8, 0.0]
