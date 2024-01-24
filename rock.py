import tensorflow as tf
import cv2
import numpy as np

model = tf.keras.models.load_model("keras_model.h5")
vid = cv2.VideoCapture(0)

while True:
    suc, frame = vid.read()
    resize_image = cv2.resize(frame,(224,224))
    test_image = np.array(resize_image, dtype=np.float32)
    test_image = np.expand_dims(test_image, axis=0)
    normal_image = test_image/255
    prediction = model.predict(normal_image)
    print("Prediction: ", prediction)
    cv2.imshow("Rock, Paper, Scicors", frame)
    if cv2.waitKey(25) == 32:
        break
    
cv2.destroyAllWindows()