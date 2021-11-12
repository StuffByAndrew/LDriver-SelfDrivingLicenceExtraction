from typing import NamedTuple
from tensorflow.keras import layers, models, optimizers
import string
import cv2
import numpy as np

ALL_LETTERS = (list(string.ascii_uppercase)) + list(map(str,range(0,10)))

class LicenceOCR:
    #list of bbox locations [x1, x2, y1, y2]
    lbbox = (
        [26, 153, 116, 192],
        [147, 292, 115, 189],
        [23, 82, 217, 250],
        [75, 127, 217, 250],
        [170, 222, 217, 250],
        [221, 269, 217, 250]
    )
    img_shape = (50, 50, 3)

    def __init__(self, vtesting=False):
        self.vtest = vtesting
        self.model = self.load_weights()

    def read_licence(self, img):
        letter_rois = [img[y1:y2,x1:x2] for x1,x2,y1,y2 in self.lbbox]
        resized_letters = np.array(list(map(lambda img : cv2.resize(img, self.img_shape[:2]), letter_rois)))
        self.vshow(resized_letters)
        preds_oh = self.model.predict(resized_letters)
        preds = [ALL_LETTERS[np.argmax(p)] for p in preds_oh]
        print(preds)

    def vshow(self, imgs):
        if self.vtest:
            for img in imgs:
                cv2.imshow('vtest', img)
                cv2.waitKey(0)

    def load_weights(self):
        # Create a new model instance
        model = self.create_model()
        # Restore the weights
        model.load_weights('./weights/best')
        return model
    
    def create_model(self):
        conv_model = models.Sequential()
        conv_model.add(layers.Conv2D(32, (3, 3), activation='relu',
                                    input_shape=self.img_shape))
        conv_model.add(layers.MaxPooling2D((2, 2)))
        conv_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        conv_model.add(layers.MaxPooling2D((2, 2)))
        conv_model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        conv_model.add(layers.MaxPooling2D((2, 2)))
        conv_model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        conv_model.add(layers.MaxPooling2D((2, 2)))
        conv_model.add(layers.Flatten())
        conv_model.add(layers.Dropout(0.5))
        conv_model.add(layers.Dense(512, activation='relu'))
        conv_model.add(layers.Dense(len(ALL_LETTERS), activation='softmax'))

        LEARNING_RATE = 1e-4
        conv_model.compile(loss='categorical_crossentropy',
                            optimizer=optimizers.RMSprop(lr=LEARNING_RATE),
                            metrics=['acc'])
        return conv_model

if __name__ == '__main__':
    from detection import find_licence
    import glob
    ocr = LicenceOCR(vtesting=True)
    for img in glob.glob('experiments/test_images/*.png'):
        orig_img = cv2.imread(img)
        # cv2.imshow('vtest', orig_img)
        # cv2.waitKey(0)

        # orig_img = cv2.imread('./experiments/test_images/74.png')
        try:
            licence = find_licence(orig_img)
            ocr.read_licence(licence)
        except:
            print('no licence')
        