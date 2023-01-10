import tensorflow as tf
from tqdm.notebook import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from sklearn.metrics import *
from sklearn.preprocessing import OneHotEncoder


class ImageClassifierCNN:
    def __init__(
        self,
        X_test,
        y_test,
        img_width=60,
        img_height=60,
        img_depth=3,
        batch_size=32,
        epochs=100,
        show_arch=False,
    ):
        self.X_test = X_test
        self.y_test = y_test
        self.img_width = img_width
        self.img_height = img_height
        self.img_depth = img_depth
        self.batch_size = batch_size
        self.epochs = epochs
        self.show_arch = show_arch

        self.model = self._build_model()

    def _model(self):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    filters=32,
                    kernel_size=3,
                    padding="same",
                    activation="relu",
                    input_shape=(self.img_width, self.img_height, self.img_depth),
                ),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, padding="same", activation="relu"
                ),
                tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
                tf.keras.layers.Conv2D(
                    filters=128, kernel_size=3, padding="same", activation="relu"
                ),
                tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
                tf.keras.layers.Conv2D(
                    filters=256, kernel_size=3, padding="same", activation="relu"
                ),
                tf.keras.layers.Conv2D(
                    filters=512, kernel_size=3, padding="same", activation="relu"
                ),
                tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
                tf.keras.layers.Conv2D(
                    filters=1024, kernel_size=3, padding="same", activation="relu"
                ),
                tf.keras.layers.Conv2D(
                    filters=2048, kernel_size=3, padding="same", activation="relu"
                ),
                tf.keras.layers.Conv2D(
                    filters=4096, kernel_size=3, padding="same", activation="relu"
                ),
                tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(units=1024, activation="relu"),
                tf.keras.layers.Dense(units=512, activation="relu"),
                tf.keras.layers.Dense(units=256, activation="relu"),
                tf.keras.layers.Dense(units=2, activation="softmax"),
            ]
        )

        return model

    def _build_model(self):
        model = self._model()
        if self.show_arch:
            print("Model Arch:")
            model.summary()
        return model

    def compile_model(self, optimizer, loss, metrics):
        print("Compiling model")
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
        )

    def train(self, X_train, y_train, X_valid, y_valid):
        print("Unsing data augmentation.")
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.15,
            height_shift_range=0.15,
            shear_range=0.15,
            zoom_range=0.15,
            horizontal_flip=True,
        )
        print("Star model training")
        self.history = self.model.fit(
            train_datagen.flow(
                X_train.reshape(
                    X_train.shape[0], self.img_width, self.img_height, self.img_depth
                ),
                y_train,
                batch_size=self.batch_size,
            ),
            validation_data=(
                X_valid.reshape(
                    X_valid.shape[0], self.img_width, self.img_height, self.img_depth
                ),
                y_valid,
            ),
            epochs=self.epochs,
        )
        print("Train successfully completed")
        return self.history

    def predict(self, return_type: str):
        predictions = self.model.predict(self.X_test)
        if return_type.lower() == "coverted":
            return tf.math.argmax(predictions, axis=1)
        elif return_type.lower() == "raw":
            return prediction
        else:
            raise ('Unsupported type, you better use "raw" or "converted" ')

    def evaluate(self, metric: str):
        y_pred = self.predict(self.X_test, return_type="converted")
        y_pred = y_pred.numpy()
        if metrics.lower() == "mse":
            return mean_squared_error(tf.math.argmax(self.y_test, axis=1), y_pred)
        elif metrics.lower() == "accuracy_score":
            return accuracy_score(tf.math.argmax(self.y_test, axis=1), y_pred)
        elif metrics.lower() == "f1_score":
            return f1_score(tf.math.argmax(self.y_test, axis=1), y_pred)
        else:
            raise (
                'Unsupported type, you better use "mse" or "accuracy_score" or "f1_score"'
            )


if __name__ == "__main__":
    X = np.load("/dbfs/FileStore/dataimage.npy")
    y = np.load("/dbfs/FileStore/y.npy")

    enc = OneHotEncoder(handle_unknown="ignore")
    enc.fit(y)
    encoded_y = enc.transform(y).toarray()

    X_train, X_valid, y_train, y_valid = train_test_split(X, encoded_y, test_size=0.3)

    optims = [
        optimizers.Nadam(learning_rate=0.001),
        "adam",
        optimizers.experimental.Adadelta(0.001),
        optimizers.experimental.Adagrad(0.001),
        optimizers.experimental.Adamax(0.001),
        optimizers.Ftrl(0.001),
        optimizers.experimental.RMSprop(0.001),
    ]
    loss_functions = [
        tf.keras.losses.SparseCategoricalCrossentropy(),
        tf.keras.losses.CategoricalCrossentropy(),
        tf.keras.losses.BinaryCrossentropy(),
        tf.keras.losses.Huber(delta=1.3),
    ]
    _metrics = [
        [tf.keras.metrics.MeanSquaredError()],
        [
            tf.keras.metrics.MeanSquaredError(),
            "accuracy",
            "AUC",
            tf.keras.metrics.Recall(),
        ],
    ]
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_mean_squared_error',
        factor=0.8,
        patience=7,
        min_lr=1e-7,
        verbose=1,
    )

    callbacks = [
        lr_scheduler,
    ]

    if tf.test.is_gpu_available():
        with tf.device("GPU:0"):
            classifier = ImageClassifierCNN(X_test=None, y_test=None, show_arch=True)

            classifier.compile_model(
                optimizer=optims[1],
                loss=loss_functions[3],
                metrics=_metrics[1],
            )
            history = model.fit_generator(
                train_datagen.flow(
                    X_train.reshape(X_train.shape[0], img_width, img_height, img_depth),
                    y_train,
                    batch_size=batch_size,
                ),
                validation_data=(
                    X_valid.reshape(X_valid.shape[0], img_width, img_height, img_depth),
                    y_valid,
                ),
                steps_per_epoch=len(X_train) / batch_size,
                epochs=epochs,
                callbacks=callbacks,
            )
    
