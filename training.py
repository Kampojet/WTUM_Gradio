import tensorflow as tf
import matplotlib.pyplot as plt


def calculate_f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall + 1e-10)


def plot_metrics(history, name):
    epochs = range(1, len(history.history['categorical_accuracy']) + 1)

    # Plotting
    plt.figure(figsize=(12, 8))

    # Categorical Accuracy
    plt.subplot(2, 3, 1)
    plt.plot(epochs, history.history['categorical_accuracy'], label='Train')
    plt.plot(epochs, history.history['val_categorical_accuracy'], label='Test')
    plt.title('Categorical Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Categorical Accuracy')
    plt.legend()
    plt.xticks(epochs)

    # Loss
    plt.subplot(2, 3, 2)
    plt.plot(epochs, history.history['loss'], label='Train')
    plt.plot(epochs, history.history['val_loss'], label='Test')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.xticks(epochs)

    plt.tight_layout()
    plt.savefig(name + '.jpg', format='jpg')


def preprocessing(root_dir):
    img_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=90,
        horizontal_flip=False,
        vertical_flip=False,
        channel_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        rescale=1. / 255,
        brightness_range=(0.5, 1),
        validation_split=0.3)

    img_generator_flow_train = img_generator.flow_from_directory(
        directory=root_dir,
        target_size=(224, 224),
        batch_size=32,
        shuffle=True,
        subset="training")

    img_generator_flow_valid = img_generator.flow_from_directory(
        directory=root_dir,
        target_size=(224, 224),
        batch_size=32,
        shuffle=True,
        subset="validation")

    return img_generator_flow_train, img_generator_flow_valid


def training_resnet(img_generator_flow_train, img_generator_flow_valid, iftest=False):
    base_model = tf.keras.applications.ResNet50V2(input_shape=(224, 224, 3),
                                                  include_top=False,
                                                  weights="imagenet"
                                                  )
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4, activation="softmax")  # 4 classes
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.CategoricalAccuracy()])
    if iftest:
        model.fit(img_generator_flow_train,
                  validation_data=img_generator_flow_valid,
                  steps_per_epoch=2, epochs=2)
    else:
        model.fit(img_generator_flow_train,
                  validation_data=img_generator_flow_valid,
                  epochs=20)
    return model


def training_inception(img_generator_flow_train, img_generator_flow_valid, iftest=False):
    base_model = tf.keras.applications.InceptionV3(input_shape=(224, 224, 3),
                                                   include_top=False,
                                                   weights="imagenet"
                                                   )
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4, activation="softmax")  # 4 classes
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.CategoricalAccuracy()])

    if iftest:
        model.fit(img_generator_flow_train,
                  validation_data=img_generator_flow_valid,
                  steps_per_epoch=2, epochs=2)
    else:
        model.fit(img_generator_flow_train,
                  validation_data=img_generator_flow_valid,
                  epochs=20)

    return model


if __name__ == '__main__':
    IS_TESTING = False
    train_data, valid_data = preprocessing('Dog Emotions')
    model_r = training_resnet(train_data, valid_data, IS_TESTING)
    plot_metrics(model_r.history, "ResNet50V2")
    if not IS_TESTING:
        model_r.save("ResNet50V2.keras")
    model_i = training_inception(train_data, valid_data, IS_TESTING)
    plot_metrics(model_i.history, "InceptionV3")
    if not IS_TESTING:
        model_i.save("InceptionV3.keras")
