# DatasetHamil.py  (nuevo)
import numpy as np
import pandas as pd

# === Geometría y masas fijas (puedes ajustarlas) ===
L  = 0.2355
l  = 0.15
m_base = 20.0
m_w    = 1.0
r_w    = 0.0475
M_CONST = m_base + 4.0*m_w  # masa total base (sin carga)
# Izz base como en tu código de prueba
Izz_chasis = (1.0/3.0)*m_base*(L**2 + l**2)
Izz_rueda  = m_w*(L**2 + l**2) + 0.25*m_w*(r_w**2)
Izz_const  = Izz_chasis + 4.0*Izz_rueda

def true_H(ux, uy, uth, m_tot=M_CONST, Izz=Izz_const):
    return 0.5*m_tot*(ux**2 + uy**2) + 0.5*Izz*(uth**2)

def make_dataset(
    n_samples=500000,
    ux_range=(-1.5, 1.5),
    uy_range=(-1.5, 1.5),
    uth_range=(-1.5, 1.5),
    seed=42,
    out_csv="hamiltonian_dataset.csv",
):
    rng = np.random.default_rng(seed)
    ux  = rng.uniform(*ux_range,  size=n_samples)
    uy  = rng.uniform(*uy_range,  size=n_samples)
    uth = rng.uniform(*uth_range, size=n_samples)
    H   = true_H(ux, uy, uth)
    df = pd.DataFrame({"ux": ux, "uy": uy, "uth": uth, "H": H})
    df.to_csv(out_csv, index=False)
    print(f"[OK] Dataset guardado en: {out_csv}  |  muestras: {len(df)}")

if __name__ == "__main__":
    make_dataset()
