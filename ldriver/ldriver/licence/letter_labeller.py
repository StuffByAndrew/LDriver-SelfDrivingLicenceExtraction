import glob
import argparse
import cv2
from ocr import ALL_LETTERS
import json
import re
import os
import logging

def has_all_numbers(dct=None):
    if not dct:
        if os.path.isfile('./plate_data/labels.json'):
            with open('./plate_data/labels.json', 'r') as f:
                labels = json.load(f)
        else:
            return
    else:
        labels = dct
    
    letters = set(map(str,labels.values()))
    print('dataset includes {} letters out of {}'.format(len(letters), len(ALL_LETTERS)))
    print('letters:{}'.format(letters))
    print('missing: {}'.format(set(ALL_LETTERS)-letters))

def main(source):
    cv_letter_offset = 97
    cv_number_offset = 48

    def is_cv_number(n):
        return 48 <= n <= 57

    def is_cv_alpha(n):
        return 97 <= n <= 122

    labels = {}    
    for imgf in glob.glob('./plate_data/*.png'):
        img = cv2.imread(imgf, cv2.IMREAD_UNCHANGED)
        cv2.imshow('labeller', img)
        key = cv2.waitKey(0)
        logging.debug(key)
        if key == 59:
            os.remove(imgf)
        else:
            for _ in range(100):
                try:
                    label = ALL_LETTERS[key-cv_letter_offset] if is_cv_alpha(key) else ALL_LETTERS[key-cv_number_offset+26]
                    print(label)                
                    labels[re.findall(r'\d+', imgf)[0]] = label
                    break
                except IndexError:
                    print('invalid keyboard input')
                    continue

    with open('./plate_data/labels.json', 'w+') as f:
        json.dump(labels, f)
    has_all_numbers(labels)

    

if __name__ == '__main__':
    has_all_numbers()
    # parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_argument('--source', dest='source', action='store_const',
    #                 const=sum, default=max,
    #                 help='sum the integers (default: find the max)')

    # args = parser.parse_args()
    main(None)