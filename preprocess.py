import numpy as np
import cv2
import os

class DataCollect :
    '''
        class collecting data(gray scale) : assert(num_of_noise == num_of_original)
        should be imported numpy(1.16.2), os, cv2(opencv-python==4.1.0)
        data_set returned two data set of ndarray 
    '''

    def __init__ (self, noise_path = "crack_data\\noise\\", original_path = "crack_data\\original\\") :
        '''
            initializer included data_path(noise, original both)
            and, also use status dictionary
        '''

        self.noise_path = noise_path
        self.original_path = original_path
        
        if(self.noise_path[len(self.noise_path) - 1] != "\\") :
            self.noise_path + "\\"
            
        if(self.original_path[len(self.original_path) - 1] != "\\") :
            self.original_path + "\\"
        
        self.noise_list = np.array(os.listdir(noise_path))
        self.original_list = np.array(os.listdir(original_path))
        
        self.num_of_data = self.noise_list.shape[0]
        
        assert(self.num_of_data == self.original_list.shape[0])
        
        self.status = {"collected" : False, "processed" : False}
        
    def raw_data_set (self) :
        '''
            raw_data with no processed data - but data's shape is (num_of_data, 250, 250)
        '''

        noise_data = []
        original_data = []
        
        for i in range(self.num_of_data) :
            noise_path_tmp = self.noise_path + self.noise_list[i]
            original_path_tmp = self.original_path + self.original_list[i]
            
            noise_tmp = cv2.imread(noise_path_tmp, cv2.IMREAD_GRAYSCALE)
            original_tmp = cv2.imread(original_path_tmp, cv2.IMREAD_GRAYSCALE)
            
            noise_tmp_np = np.array(noise_tmp).astype(np.float32)
            original_tmp_np = np.array(original_tmp).astype(np.float32)
            
            if(noise_tmp_np.shape[0] != 250 or noise_tmp_np.shape[1] != 250) :
                noise_tmp = cv2.resize(noise_tmp, dsize = (250, 250), interpolation = cv2.INTER_AREA)
                noise_tmp_np = np.array(noise_tmp).astype(np.float32)
                
            if(original_tmp_np.shape[0] != 250 or original_tmp_np.shape[1] != 250) :
                original_tmp = cv2.resize(original_tmp, dsize = (250, 250), interpolation = cv2.INTER_AREA)
                original_tmp_np = np.array(original_tmp).astype(np.float32)
                
            noise_tmp = noise_tmp_np.tolist()
            original_tmp = original_tmp_np.tolist()
            
            noise_data.append(noise_tmp)
            original_data.append(original_tmp)
        
        noise_data = np.array(noise_data)
        original_data = np.array(original_data)
        
        self.raw_noise_data = noise_data
        self.raw_original_data = original_data
        
        self.status["collected"] = True
        
        return noise_data, original_data
    
    def preprocessed_set (self) :
        '''
            shape = (num_of_data, 250, 250, 1), range = 0 - 1
        '''

        if not self.status["collected"] :
            raw_noise, raw_original = self.raw_data_set()
        else :
            raw_noise = self.raw_noise_data
            raw_original = self.raw_original_data
            
        raw_noise /= 255.
        raw_original /= 255.
        
        processed_noise = np.reshape(raw_noise, newshape = (-1, 250, 250, 1))
        processed_original = np.reshape(raw_original, newshape = (-1, 250, 250, 1))
        
        self.processed_noise = processed_noise
        self.processed_original = processed_original
        
        self.status["processed"] = True
        
        return processed_noise, processed_original
    
    def sample_show (self, sample_num = 0) :
        '''
            sample image showing
        '''

        if(sample_num > self.num_of_data - 1) :
            print("There's no image in {} index" .format(sample_num))
            
            return 0
        
        if not self.status["processed"] :
            self.preprocessed_set()
            

        cv2.imshow("noise_sample_show", self.processed_noise[sample_num])
        cv2.imshow("sample_show", self.processed_original[sample_num])
        cv2.waitKey(0)
        cv2.destroyAllWindows() 