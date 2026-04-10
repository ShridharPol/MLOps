import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    # Load the Wine dataset (13 features, 3 classes)
    wine = datasets.load_wine()
    X, y = wine.data, wine.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features (important for neural networks)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Build a TensorFlow model for 13 input features and 3 output classes
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, input_shape=(13,), activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

    model.save('my_model.keras')
    print("Wine model was trained and saved as my_model.keras")