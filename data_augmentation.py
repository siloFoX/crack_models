from preprocess import DataCollect
import cv2
import numpy as np

class DataAugmentation :
    '''
        data augmentation with cv2(opencv-python==4.1.0) & DataCollect(class in preprocess.py)
        imported numpy(1.16.2)
    '''
    
    def __init__ (self, data_store = False) : 

        self.data_store = data_store

        dc = DataCollect()
        self.pre_noise, self.pre_original = dc.preprocessed_set()

        if data_store :
            self.noise_path = dc.noise_path
            self.original_path = dc.original_path

            self.noise_list = dc.noise_list
            self.original_list = dc.original_list
        
    def rotation (self, data) :

        tmp = []
        angles = [90, 180, -90]
        scale_of_data = 1

        length_data = data.shape[0]
        
        if not self.data_store :

            for angle in angles :

                for idx in range(length_data) :

                    datum = data[idx]

                    rotation_Matrix = cv2.getRotationMatrix2D((125, 125), angle, scale_of_data)
                    img_rotated = cv2.warpAffine(datum, rotation_Matrix, (250, 250))
                    img_rotated = np.reshape(img_rotated, newshape = (250, 250, 1))

                    tmp.append(img_rotated.tolist())

        data = data.tolist()

        for datum in tmp :
            data.append(datum)

        return np.array(data)

    def rotated_data (self) :

        rotated_noise = self.rotation(self.pre_noise)
        rotated_original = self.rotation(self.pre_original)

        return rotated_noise, rotated_original