
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # pyright: ignore[reportMissingImports]

def build_mlp(input_dim: int, hidden_layers=[128, 64], dropout=0.2):
    """
    Construye un MLP simple:
    input_dim -> Dense(hidden_layers...) -> Output(sigmoid)
    """
    inputs = keras.Input(shape=(input_dim,), name="input")
    x = inputs
    for i, units in enumerate(hidden_layers):
        x = layers.Dense(units, activation="relu", name=f"dense_{i}")(x)
        if dropout and dropout > 0:
            x = layers.Dropout(dropout, name=f"dropout_{i}")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="mlp_model")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss="binary_crossentropy",
                  metrics=["accuracy", keras.metrics.AUC(name="auc")])
    return model
