import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import math
import os
import scipy.ndimage
import shutil
from collections import Counter
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, img_to_array, save_img
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from PIL import Image


def pad_image(image, target_height, target_width):
        height, width = image.shape
        pad_top = (target_height - height) // 2
        pad_bottom = target_height - height - pad_top
        pad_left = (target_width - width) // 2
        pad_right = target_width - width - pad_left
        padded_image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT,value=255)
        return padded_image

def trained_model():
    # Function to preprocess image
    def preprocess_image(image_path, num_rows):
        # Cut the image into rows of height pixels
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        height, width = image.shape
        row_height = height // num_rows
        quarter_width = (width - 350) // 7
        rows = []
        for i in range(0, height, row_height):
            if i <= height - row_height:
                row = image[i:i+row_height, :]  # Extract the row

                # Split the row into segments
                for j in range(350, width, quarter_width):
                    if j <= width - quarter_width:
                        segment = row[:, j:j+quarter_width]
                        rows.append(segment)
            if i > height - row_height:
                break
        return rows

    ### Settings ###
    img_height , img_width = 255 , (3010 - 350)
    quarter_img_width = img_width // 7

    ### Creating Data ###
    row1 = preprocess_image('Test_Dataset\hw1-test.png', 8)
    row1_padded = [pad_image(line, img_height, quarter_img_width) for line in row1]
    row2 = preprocess_image('Test_Dataset\hw2-test.png', 8)
    row2_padded = [pad_image(line, img_height, quarter_img_width) for line in row2]
    row3 = preprocess_image('Test_Dataset\hw3-test.png', 8)
    row3_padded = [pad_image(line, img_height, quarter_img_width) for line in row3]
    row4 = preprocess_image('Test_Dataset\hw4-test.png', 8)
    row4_padded = [pad_image(line, img_height, quarter_img_width) for line in row4]

    ### Create training & testing data ###
    train_data = np.concatenate((row1_padded[:42],row2_padded[:42],row3_padded[:42],row4_padded[:42]), axis=0)
    valid_data = np.concatenate((row1_padded[42:56],row2_padded[42:56],row3_padded[42:56],row4_padded[42:56]), axis=0)

    ### Add a 1 channel dimension to the data created ###
    train_data = np.expand_dims(train_data, axis=-1)
    valid_data = np.expand_dims(valid_data, axis=-1)

    ### Create labels for training & test data ###
    labels_train = np.array([0] * 42 + [1] * 42 + [2] * 42 + [3] * 42)
    categorical_labels_train = to_categorical(labels_train, num_classes=4)
    labels_validation = np.array([0] * 14 + [1] * 14 + [2] * 14 + [3] * 14)
    categorical_labels_validation = to_categorical(labels_validation, num_classes=4)

    ### Augmentation ###
    train_datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.05, horizontal_flip=True, zoom_range=0.2, brightness_range=[0.8, 1.2] ) # Random rotation in degrees # Shear transformation
    train_datagen.fit(train_data)

    ### Create augmented data ###
    aug_data = []
    aug_labels = []
    for batch1, batch2 in train_datagen.flow(train_data, categorical_labels_train, batch_size=len(train_data)):
        aug_data.append(batch1)
        aug_labels.append(batch2)
        if len(aug_data) >= 1:  # Only need one batch since batch_size = len(training_data)
            break
    
    augmented_data = np.concatenate(aug_data)
    augmented_labels = np.concatenate(aug_labels)

    ### Combine original & augmented data ###
    combined_data = np.concatenate((train_data, augmented_data), axis=0)
    combined_labels = np.concatenate((categorical_labels_train, augmented_labels), axis=0)

    ### Define the CNN Model ###
    model = Sequential([

        Conv2D(64, (2, 2), activation='relu', input_shape=(img_height, quarter_img_width, 1)),
        MaxPooling2D((2, 2)),
        #BatchNormalization(),
        Conv2D(32, (2, 2), activation='relu'),
        MaxPooling2D((2, 2)),
        #BatchNormalization(),
        Conv2D(32, (2, 2), activation='relu'),
        MaxPooling2D((2, 2)),
        #BatchNormalization(),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(4, activation='softmax')  # 4 classes for 4 handwriting styles  # Classification
    ])

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    ### Training Model ###
    epochs = 10
    batch_size = 4
    model.fit(
            combined_data, combined_labels,
            epochs=epochs,
            validation_data=(valid_data, categorical_labels_validation),
            batch_size=batch_size,
        )
    
    model.save('handwriting_model.keras')
    return model

### Test Function ###
def test(path, model):
    def pad_image(image, goal_height, goal_width):
       height, width = image.shape
       pad_top = (goal_height - height) // 2
       pad_bottom = goal_height - height - pad_top
       pad_left = (goal_width - width) // 2
       pad_right = goal_width - width - pad_left
       padded_image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT,value=255)
       return padded_image
    
    def resize_image(image_path, factor):
        # Open the image
        img = Image.open(image_path)
        print(f"Resizing image: {image_path}")

        # Get current dimensions
        width, height = img.size

        # Calculate new dimensions
        new_width = int(width * factor)
        new_height = int(height * factor)

        # Resize the image
        resized_img = img.resize((new_width, new_height))

        # Save the resized image back to the same path
        resized_img.save(image_path)
    
    def most_common_prediction(a, b, c, d, e, f, g):
        parameters = [a, b, c, d, e, f ,g]
        counts = Counter(parameters)
        most_common = counts.most_common(1)[0][0]
        return most_common

    goal_height , goal_width = 255 , (3010 - 350)

    ### imread the test_image (loading test images) ###
    testing_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    ### Adjusting Image ###
    height, width = testing_image.shape
    x = goal_height / height
    y = goal_width / width
    if y < 1:
        resize_image(path, y)
        testing_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        height, width = testing_image.shape
        x = goal_height / height
        if x < 1:
            resize_image(path, x)
    elif y > 1:
        resize_image(path, x)
        testing_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        height, width = testing_image.shape
        y = goal_width / width
        if y < 1:
            resize_image(path, y)

    testing_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    ### Padding test_images ###
    pad_testing_image = pad_image(testing_image, goal_height, goal_width)
    quarter_width = goal_width // 7
    quarter_image0 = pad_testing_image[:,:quarter_width]
    quarter_image1 = pad_testing_image[:,quarter_width:2*quarter_width]
    quarter_image2 = pad_testing_image[:,2*quarter_width:3*quarter_width]
    quarter_image3 = pad_testing_image[:,3*quarter_width:4*quarter_width]
    quarter_image4 = pad_testing_image[:,4*quarter_width:5*quarter_width]
    quarter_image5 = pad_testing_image[:,5*quarter_width:6*quarter_width]
    quarter_image6 = pad_testing_image[:,6*quarter_width:]
    
    ### Adding channel & batch ###
    extend_image0 = np.expand_dims(quarter_image0, axis=-1)  # Adding channel dimension
    extend_image0 = np.expand_dims(extend_image0, axis=0)  # Adding batch dimension
    extend_image1 = np.expand_dims(quarter_image1, axis=-1)  
    extend_image1 = np.expand_dims(extend_image1, axis=0)  
    extend_image2 = np.expand_dims(quarter_image2, axis=-1)  
    extend_image2 = np.expand_dims(extend_image2, axis=0)  
    extend_image3 = np.expand_dims(quarter_image3, axis=-1)  
    extend_image3 = np.expand_dims(extend_image3, axis=0)  
    extend_image4 = np.expand_dims(quarter_image4, axis=-1)  
    extend_image4 = np.expand_dims(extend_image4, axis=0)  
    extend_image5 = np.expand_dims(quarter_image5, axis=-1)  
    extend_image5 = np.expand_dims(extend_image5, axis=0)  
    extend_image6 = np.expand_dims(quarter_image6, axis=-1)  
    extend_image6 = np.expand_dims(extend_image6, axis=0)  

    ### Predictions(Probability) ###
    image_prediction0 = model.predict(extend_image0)
    image_prediction1 = model.predict(extend_image1)
    image_prediction2 = model.predict(extend_image2)
    image_prediction3 = model.predict(extend_image3)
    image_prediction4 = model.predict(extend_image4)
    image_prediction5 = model.predict(extend_image5)
    image_prediction6 = model.predict(extend_image6)

    ### Getting predicted class index & prints ###
    predict_class0 = np.argmax(image_prediction0, axis=1)[0] + 1
    predict_class1 = np.argmax(image_prediction1, axis=1)[0] + 1
    predict_class2 = np.argmax(image_prediction2, axis=1)[0] + 1
    predict_class3 = np.argmax(image_prediction3, axis=1)[0] + 1
    predict_class4 = np.argmax(image_prediction4, axis=1)[0] + 1
    predict_class5 = np.argmax(image_prediction5, axis=1)[0] + 1
    predict_class6 = np.argmax(image_prediction6, axis=1)[0] + 1
    print(os.path.basename(path))
    print("Predicted Square Number 1:", predict_class0)
    print("Predicted Square Number 2:", predict_class1)
    print("Predicted Square Number 3:", predict_class2)
    print("Predicted Square Number 4:", predict_class3)
    print("Predicted Square Number 5:", predict_class4)
    print("Predicted Square Number 6:", predict_class5)
    print("Predicted Square Number 7:", predict_class6)

    ### Use "most_common_prediction" function to choose the most common prediction ###
    predict_class = most_common_prediction(predict_class0, predict_class1, predict_class2, predict_class3, predict_class4, predict_class5, predict_class6)
    print("Predicted class of the test image is:", predict_class)

### running model ###
#modeling = trained_model() #uncomment to run the model

### Load the saved model ###
model = load_model('Final_Top_Model.keras') # change to 'handwriting_model.keras' if u want to load a new model

# Enter file path here
#for i in range (0,10,1):
#test("Screenshot 1.1.png", model)
#test("Screenshot 1.2.png", model)
#test("Screenshot 2.1.png", model)
#test("Screenshot 2.2.png", model)
#test("Screenshot 3.1.png", model)
#test("Screenshot 3.2.png", model)
#test("Screenshot 4.1.png", model)
#test("Screenshot 4.2.png", model)
test("Relative Path", model)