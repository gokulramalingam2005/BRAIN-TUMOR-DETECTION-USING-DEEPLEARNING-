from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from data_preprocessing import load_dataset

# Define the model architecture
def build_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model

# Load the dataset
train_images, train_labels = load_dataset('path/to/training_data', target_size=(128, 128))

# Define the model
model = build_model(input_shape=(128, 128, 3))

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Define callbacks (optional)
checkpoint = ModelCheckpoint('model_weights.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

# Train the model
history = model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_split=0.2, callbacks=[checkpoint])

# Save the trained model
model.save('brain_tumor_detection_cnn_model.h5')

print("Training completed and model saved.")
