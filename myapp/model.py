import os
import pickle
import numpy as np
from keras.models import load_model


class Model:
    def __init__(self):
        model_path = os.path.join('myapp', 'model.h5')
        # your code here
        self.model = load_model(model_path)

    def predict(self, x):
        '''
        Parameters
        ----------
        x : np.ndarray
            Входное изображение -- массив размера (28, 28)
        Returns
        -------
        pred : str
            Символ-предсказание 
        '''
        # your code here
        labels_dict = {
            0: '0',
            1: '1',
            2: '2',
            3: '3',
            4: '4',
            5: '5',
            6: '6',
            7: '7',
            8: '8',
            9: '9',
            10: 'A',
            11: 'B',
            12: 'C',
            13: 'D',
            14: 'E',
            15: 'F',
            16: 'G',
            17: 'H',
            18: 'I',
            19: 'J',
            20: 'K',
            21: 'L',
            22: 'M',
            23: 'N',
            24: 'O',
            25: 'P',
            26: 'Q',
            27: 'R',
            28: 'S',
            29: 'T',
            30: 'U',
            31: 'V',
            32: 'W',
            33: 'X',
            34: 'Y',
            35: 'Z',
            36: 'a',
            37: 'b',
            38: 'd',
            39: 'e',
            40: 'f',
            41: 'g',
            42: 'h',
            43: 'n',
            44: 'q',
            45: 'r',
            46: 't'
        }

        img_rows = 28
        img_cols = 28
        x = x.reshape(1, img_rows, img_cols, 1)
        x = x.astype('float32')
        x /= 255 # Нормализация

        pred = self.model.predict(x)

        return labels_dict[np.argmax(pred)]

