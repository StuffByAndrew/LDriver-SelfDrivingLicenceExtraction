import ldriver.data.licence
from importlib_resources import files, as_file
import cv2
import os

source = files(ldriver.data.licence).joinpath('P.png')
print(os.path.isfile(str(source)))
img = cv2.imread(str(source))
cv2.imshow('a', img)
cv2.waitKey(0)