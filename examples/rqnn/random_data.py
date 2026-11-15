"""Small forecasting and recurrent-memory diagnostic for PlainQRNNModel.

Run beside PlainQRNNModel.py
"""
import numpy as np
import tensorflow as tf
from quantum_models.PlainQRNNModel import PlainQRNNModel, make_forecasting_windows


def memory_test(model):
    """Test whether the final prediction depends on the first time step."""
    a = np.array([[-0.8, 0.1, 0.2, -0.1, 0.3, 0.4, 0.2]], dtype=np.float64)[..., None]
    b = a.copy()
    b[0, 0, 0] = 0.8
    pa, pb = model.probabilities(a).numpy()[0, 0], model.probabilities(b).numpy()[0, 0]
    print(f'Memory diagnostic: first input changed, last fixed: {pa:.8f} -> {pb:.8f}; delta={abs(pb-pa):.3e}')
    return abs(pb-pa)


def main():
    """Generate smooth temporal data, fit forecasting model and print metrics."""
    np.random.seed(42)
    tf.random.set_seed(42)
    t = np.arange(38, dtype=np.float64)
    series = 0.5 * np.sin(0.35*t) + 0.2 * np.cos(0.12*t)

    train_series = series[:30]
    X_train, y_train, scale = make_forecasting_windows(train_series, n_steps=7)
    X_all, y_all, _ = make_forecasting_windows(series, n_steps=7, target_range=scale)
    X_test, y_test = X_all[len(X_train):], y_all[len(y_train):]
    print('Training windows:', X_train.shape, 'test windows:', X_test.shape)
    print('Training data scale:', scale)
    print('First input:', X_train[0, :, 0], 'target:', y_train[0, 0])

    architecture = {'quantum': {'data_register': {'units': 3},
                                'hidden_state': {'units': 3},
                                'ansatz': 'paper_readout'}}
    model = PlainQRNNModel(n_feat=1, n_steps=7,
                           architecture=architecture, target_range=scale)
    print('Trainable quantum parameters:', model.block.n_weights)
    memory_test(model)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.03)
    X = tf.constant(X_train, dtype=tf.float64)
    y = tf.constant(y_train, dtype=tf.float64)

    for epoch in range(3):
        with tf.GradientTape() as tape:
            predictions = model(X)
            loss = tf.reduce_mean(tf.square(predictions - y))
        gradient = tape.gradient(loss, model.quantum_weights)
        if gradient is None:
            raise RuntimeError('No quantum gradients: inspect QNode differentiation.')
        optimizer.apply_gradients([(gradient, model.quantum_weights)])
        print(f'Epoch {epoch+1}: training MSE={float(loss):.6f}; active gradients={int(np.sum(np.abs(gradient.numpy()) > 1e-8))}/{model.block.n_weights}')

    predicted = model(tf.constant(X_test, dtype=tf.float64)).numpy()
    print('Test MAE:', float(np.mean(np.abs(predicted-y_test))))
    print('Test MSE:', float(np.mean((predicted-y_test)**2)))
    for actual, estimate in zip(y_test[:, 0], predicted[:, 0]):
        print(f'actual={actual:+.5f}, predicted={estimate:+.5f}')


if __name__ == '__main__':
    main()