# -*- coding: utf-8 -*-
"""
Physics-Informed Neural Network (PINN) for Hamiltonian Prediction.
Part of the "PID Adaptativo Bajo Demanda Mediante Redes Neuronales" framework.

This script trains a constrained linear model to learn the kinetic energy 
coefficients mapping robot velocities [u_x, u_y, u_theta] to the canonical 
Hamiltonian (H). By explicitly engineering squared velocity features and 
applying non-negative weight constraints, the network is forced to adhere 
to physical energy laws.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# --- Configuration ---
DATA_PATH = "hamiltonian_dataset.csv"
SAVE_MODEL_PATH = "hamiltonian_tf_model.keras"

def main():
    # 1. Load and Prepare Dataset
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Dataset not found at: {DATA_PATH}")
        return

    print("[INFO] Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    
    X = df[["ux", "uy", "uth"]].values.astype("float32")
    y = df["H"].values.astype("float32")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=7
    )

    # 2. Physics-Informed Model Architecture
    inputs = keras.Input(shape=(3,), name="u_velocities")  # [ux, uy, uth]
    
    # Extract individual velocity components
    ux  = layers.Lambda(lambda z: z[:, 0:1], name="extract_ux")(inputs)
    uy  = layers.Lambda(lambda z: z[:, 1:2], name="extract_uy")(inputs)
    uth = layers.Lambda(lambda z: z[:, 2:3], name="extract_uth")(inputs)

    # Apply physical transformations (Kinetic Energy is proportional to v^2)
    ux2 = layers.Multiply(name="ux_squared")([ux, ux])
    uy2 = layers.Multiply(name="uy_squared")([uy, uy])
    
    v2 = layers.Add(name="translational_v2")([ux2, uy2])      # ux^2 + uy^2
    w2 = layers.Multiply(name="rotational_w2")([uth, uth])    # uth^2

    features = layers.Concatenate(name="phys_features")([v2, w2])  # Shape: (N, 2)

    # Linear layer representing: H = a * (ux^2 + uy^2) + b * (uth^2)
    # Weights 'a' and 'b' represent equivalent mass/inertia parameters and must be >= 0.
    out = layers.Dense(
        1, 
        use_bias=False, 
        name="H_prediction",
        kernel_constraint=keras.constraints.NonNeg()
    )(features)

    model = keras.Model(inputs, out, name="hamiltonian_physlinear_nomass")
    
    # 3. Model Compilation
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3), 
        loss="mse",
        metrics=[
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
            tf.keras.metrics.MeanAbsoluteError(name="mae")
        ]
    )

    model.summary()

    # 4. Training Callbacks
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            SAVE_MODEL_PATH, monitor="val_loss", save_best_only=True, verbose=1
        )
    ]

    # 5. Model Training
    print("[INFO] Starting model training...")
    model.fit(
        X_train, y_train, 
        validation_data=(X_test, y_test),
        epochs=200, 
        batch_size=1024,
        callbacks=callbacks,
        verbose=2
    )

    # 6. Evaluation and Weight Extraction
    print("\n[INFO] Training complete. Extracting physical parameters...")
    weights = model.get_layer("H_prediction").get_weights()[0].ravel()
    
    print("-" * 40)
    print(f"Learned Translational Coefficient (a) ≈ {weights[0]:.6f}")
    print(f"Learned Rotational Coefficient (b)    ≈ {weights[1]:.6f}")
    print("-" * 40)
    print(f"[OK] Model successfully saved to: {SAVE_MODEL_PATH}")

if __name__ == "__main__":
    main()