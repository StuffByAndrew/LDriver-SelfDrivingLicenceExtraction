from os import stat
from tensorflow.keras import layers, models, optimizers
import string
import cv2
import numpy as np
from utils import sharpen
from matplotlib import pyplot as plt

ALL_LETTERS = list(string.ascii_uppercase) + list(map(str,range(0,10)))

class LicenceOCR:
    img_shape = (50, 50, 1)

    def __init__(self, vtesting=False, experimental=False):
        self.vtest = vtesting
        self.exper = experimental
        self.model = self.load_weights()

    def read_licence(self, licence):
        letter_rois = licence.letters
        
        resized_letters = self.process_letters(letter_rois)

        # Reshape
        a = self.img_shape
        resized_letters = resized_letters.reshape(resized_letters.shape[0], a[0], a[1], a[2])
        self.vshow(resized_letters)
        preds_oh = self.model.predict(resized_letters)
        preds = [ALL_LETTERS[np.argmax(p)] for p in preds_oh]
        print(preds)
        return preds

    @classmethod
    def process_letters(cls, letters):
        return np.array(list(map(lambda letters : cv2.resize(letters, cls.img_shape[:2]), letters)))

    def vshow(self, imgs):
        if self.vtest:
            for img in imgs:
                img = np.squeeze(img, axis=2)
                cv2.imshow('vtest', img)
                cv2.waitKey(0)
                if self.exper:
                    ret,th1 = cv2.threshold(img,60,255,cv2.THRESH_BINARY_INV)
                    th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,15,2)
                    th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,15,2)
                    titles = ['Original Image', 'Global Thresholding (v = 60  )', 'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
                    images = [img, th1, th2, th3]
                    for i in range(4):
                        plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
                        plt.title(titles[i])
                        plt.xticks([]),plt.yticks([])
                    plt.show()
                    self .histogram(img)

    def load_weights(self):
        # Create a new model instance
        model = self.create_model()
        # Restore the weights
        model.load_weights('weights/best')
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

    @staticmethod
    def histogram(gray_img):
        n, bins, patches = plt.hist(gray_img.flatten(), bins=50)
        plt.show()

if __name__ == '__main__':
    from detection import LicencePlate
    import glob
    ocr = LicenceOCR(vtesting=True, experimental=False)
    for img in glob.glob('experiments/test_images/*.png'):
        orig_img = cv2.imread(img)
        # cv2.imshow('vtest', orig_img)
        # cv2.waitKey(0)
  
        # orig_img = cv2.imread('./experiments/test_images/74.png')
        # found, licence = LicencePlate.find_licence(orig_img)
        # if found:
        #     ocr.read_licence(licence)

        lp = LicencePlate(orig_img)
        if lp.valid:
            ocr.read_licence(lp)