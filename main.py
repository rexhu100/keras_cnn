import pandas as pd
import numpy as np

from process_data import process_data
from setup_model import setup_model


def main():
    # Read data from csv
    mnist_data = pd.read_csv("data/train.csv")
    train_label = np.array(mnist_data["label"])
    train_image = np.array(mnist_data.iloc[:, 1:])

    # Format data
    train_image, train_label = process_data(train_image, train_label)
    print(train_image.shape, train_label.shape)
    mnist_model = setup_model()

    # Train model
    mnist_model.fit(train_image, train_label, batch_size=32, epochs=20)

    # Read test data
    test_data = pd.read_csv("data/test.csv")
    test_image = np.array(test_data)
    test_image, _ = process_data(test_image)
    print(np.argmax(mnist_model.predict(test_image), axis=-1))

    # Format and save submission data
    test_out = pd.DataFrame()
    test_out["ImageId"] = [x for x in range(1, test_data.shape[0]+1)]
    test_out["Label"] = np.argmax(mnist_model.predict(test_image), axis=-1)

    test_out.to_csv("data/submission.csv", index=False)


if __name__ == "__main__":
    main()
