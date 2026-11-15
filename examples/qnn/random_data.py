from quantum_models.QNNModel_PL import *
from DataPreprocessing import fit_transform_target, inverse_transform_target

def create_sample_data():
    """Create sample regression data."""
    np.random.seed(42)

    X = np.random.randn(100, 20).astype(np.float32)
    y = (
        np.sin(X[:, 2]) *
        sum(X[:, i] for i in range(X.shape[1]))
    ).astype(np.float32).reshape(-1, 1)

    return X, y


# ==========================================================
# =================== DATA PREPARATION =====================
# ==========================================================

X_train, y_train = create_sample_data()

y_train_scaled, target_scaler = fit_transform_target(y_train)

num_features = X_train.shape[1]

print(f"Data shapes: X={X_train.shape}, y={y_train.shape}")
print(f"Original target range: {y_train.min():.4f}, {y_train.max():.4f}")


# ==========================================================
# =================== MODEL ARCHITECTURES ==================
# ==========================================================

# Models 1 and 2: Quantum circuit + classical Dense output
architecture_dense = {
    "pre_quantum": True,
    "quantum": {
        "read_in": {"two_features_per_qbit": True},
        "ansatz": "aspen"
    },
    "post_quantum": True
}

# Model 3: Quantum circuit with a single measurement
architecture_quantum = {
    "pre_quantum": True,
    "quantum": {
        "read_in": {"two_features_per_qbit": True},
        "ansatz": "aspen",
        "measure": [0]
    },
    "post_quantum": False
}


# ==========================================================
# =================== TRAINING FUNCTION ====================
# ==========================================================

def train_model(architecture, X, y, name):
    """
    Create, compile and train an independent QNN model.
    """
    print(f"\nTraining {name}...")

    model = QNNModel(num_features, architecture)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
        loss="mae",
        metrics=["mae"]
    )

    history = model.fit(
        X, y,
        epochs=100,
        batch_size=5,
        validation_split=0.2,
        shuffle=False
    )

    return model, history


# ==========================================================
# =================== MODEL TRAINING =======================
# ==========================================================

# Model 1: Original targets + Dense layer
model1, history1 = train_model(
    architecture_dense,
    X_train,
    y_train,
    "Model 1: Unscaled + Dense"
)

# Model 2: Scaled targets + Dense layer
model2, history2 = train_model(
    architecture_dense,
    X_train,
    y_train_scaled,
    "Model 2: Scaled + Dense"
)

# Model 3: Scaled targets + quantum-only output
model3, history3 = train_model(
    architecture_quantum,
    X_train,
    y_train_scaled,
    "Model 3: Scaled + No Dense"
)


# ==========================================================
# =================== POSTPROCESSING =======================
# ==========================================================

# Model 1: Already in original units
y_pred1 = model1.predict(X_train).reshape(-1, 1)

# Model 2: Inverse transform scaled predictions
y_pred2_scaled = model2.predict(X_train).reshape(-1, 1)
y_pred2 = inverse_transform_target(
    y_pred2_scaled, target_scaler
)

# Model 3: Inverse transform quantum measurements
y_pred3_scaled = model3.predict(X_train).reshape(-1, 1)
y_pred3 = inverse_transform_target(
    y_pred3_scaled, target_scaler
)


# ==========================================================
# =================== MODEL EVALUATION =====================
# ==========================================================

print("\nModel comparison (original target scale):")

for name, prediction in [
    ("Model 1: Unscaled + Dense", y_pred1),
    ("Model 2: Scaled + Dense", y_pred2),
    ("Model 3: Scaled + No Dense", y_pred3)
]:
    mae = np.mean(np.abs(y_train - prediction))
    mse = np.mean((y_train - prediction) ** 2)

    print(f"\n{name}")
    print(f"  MAE: {mae:.6f}")
    print(f"  MSE: {mse:.6f}")
    print(f"  Prediction range: "
          f"[{prediction.min():.4f}, {prediction.max():.4f}]")


# ==========================================================
# =================== RESULTS PLOTTING =====================
# ==========================================================

fig, axes = plt.subplots(3, 1, figsize=(12, 12))

predictions = [y_pred1, y_pred2, y_pred3]

titles = [
    "Model 1: Unscaled Targets + Dense",
    "Model 2: Scaled Targets + Dense + Inverse Scaling",
    "Model 3: Scaled Targets + No Dense + Inverse Scaling"
]

for ax, prediction, title in zip(axes, predictions, titles):
    ax.plot(
        y_train.flatten(),
        label="Actual",
        color="black"
    )
    ax.plot(
        prediction.flatten(),
        label="QNN Predictions",
        color="green"
    )

    ax.set_title(title)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Target value")
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()


# ==========================================================
# =================== TRAINING LOSS ========================
# ==========================================================

plt.figure(figsize=(12, 6))

# Model 1 uses original target units
plt.plot(
    history1.history["loss"],
    label="Model 1: Unscaled + Dense"
)

# Models 2 and 3 use normalized target units
plt.plot(
    history2.history["loss"],
    label="Model 2: Scaled + Dense"
)

plt.plot(
    history3.history["loss"],
    label="Model 3: Scaled + No Dense"
)

plt.title("QNN Training Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("MAE (different target scales)")
plt.legend()
plt.grid(True)
plt.show()
