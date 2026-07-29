"""
=========================================================
Enterprise Deep Learning Platform
Artificial Neural Network (ANN)
Model Architecture
=========================================================
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2


def build_ann(
    input_dim: int,
    output_dim: int = 1,
    learning_rate: float = 0.001
):
    """
    Builds and compiles an Artificial Neural Network.

    Parameters
    ----------
    input_dim : int
        Number of input features.

    output_dim : int
        Number of output neurons.

    learning_rate : float
        Optimizer learning rate.

    Returns
    -------
    tensorflow.keras.Model
    """

    model = Sequential(name="Enterprise_ANN")

    # Input Layer
    model.add(Dense(
        256,
        activation="relu",
        kernel_regularizer=l2(0.001),
        input_shape=(input_dim,)
    ))
    model.add(BatchNormalization())
    model.add(Dropout(0.30))

    # Hidden Layer 1
    model.add(Dense(
        128,
        activation="relu",
        kernel_regularizer=l2(0.001)
    ))
    model.add(BatchNormalization())
    model.add(Dropout(0.30))

    # Hidden Layer 2
    model.add(Dense(
        64,
        activation="relu",
        kernel_regularizer=l2(0.001)
    ))
    model.add(Dropout(0.20))

    # Hidden Layer 3
    model.add(Dense(
        32,
        activation="relu"
    ))

    # Output Layer
    if output_dim == 1:
        activation = "sigmoid"
        loss = "binary_crossentropy"
    else:
        activation = "softmax"
        loss = "categorical_crossentropy"

    model.add(Dense(
        output_dim,
        activation=activation
    ))

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate
    )

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall()
        ]
    )

    return model


if __name__ == "__main__":

    model = build_ann(
        input_dim=20,
        output_dim=1
    )

    model.summary()