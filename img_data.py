import bmp
import random
import numpy as np

class Data:
    
    def __init__(self):
        self.train = []
        for word_index in range(1, 15):
            expect = np.zeros(14)
            expect[word_index-1] = 1
            for img_index in range(200):
                file_name = './TRAIN/' + str(word_index) + '/' + str(img_index) + '.bmp'
                inputs = bmp.parse(file_name)
                self.train.append([inputs, expect])

        self.test = []
        for word_index in range(1, 15):
            expect = np.zeros(14)
            expect[word_index-1] = 1
            for img_index in range(200, 256):
                file_name = './TRAIN/' + str(word_index) + '/' + str(img_index) + '.bmp'
                inputs = bmp.parse(file_name)
                self.test.append([inputs, expect])
        
        print("train data ready")
            
    def train_set(self, data_num=10):
        inputs = []
        expects = []
        for _ in range(data_num):
            rand = random.randint(0, len(self.train) - 1)
            tmp = self.train[rand]
            inputs.append(tmp[0])
            expects.append(tmp[1])
        
        return (inputs, expects)

    def test_set(self, data_num=10):
        inputs = []
        expects = []
        for _ in range(data_num):
            rand = random.randint(0, len(self.test) - 1)
            tmp = self.test[rand]
            inputs.append(tmp[0])
            expects.append(tmp[1])
        
        return (inputs, expects)

if __name__ == "__main__":
    data = Data()
    print(data.train_set(10))