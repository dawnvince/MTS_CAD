import os
import numpy as np
import csv

data_path = "./data"
if not os.path.exists(data_path):
    os.makedirs(data_path)

def gen_SMD(path="./datasets/ServerMachineDataset"):
    SMD_machine = ["machine-1-1","machine-1-2","machine-1-3","machine-1-4","machine-1-5","machine-1-6","machine-1-7","machine-1-8","machine-2-1","machine-2-2","machine-2-3","machine-2-4","machine-2-5","machine-2-6","machine-2-7","machine-2-8","machine-2-9","machine-3-1","machine-3-2","machine-3-3","machine-3-4","machine-3-5","machine-3-6","machine-3-7","machine-3-8","machine-3-9","machine-3-10","machine-3-11"]
    
    for m in SMD_machine:
        train_path = os.path.join(path, "train", "%s.txt"%m)
        test_path = os.path.join(path, "test", "%s.txt"%m)
        label_path = os.path.join(path, "test_label", "%s.txt"%m)
        traindata = np.loadtxt(open(train_path), delimiter=',')
        testdata = np.loadtxt(open(test_path), delimiter=',')
        labels = np.genfromtxt(label_path, delimiter='\n')
        
        np.save("%s/%s_train.npy"%(data_path, m), traindata)
        np.save("%s/%s_test.npy"%(data_path, m), testdata)
        np.save("%s/%s_label.npy"%(data_path, m), labels)
        
def gen_SWaT(path="./datasets/SWaT"):
    train_path = os.path.join(path, "SWaT_Dataset_Normal_v1.csv")
    test_path = os.path.join(path, "SWaT_Dataset_Attack_v0.csv")
    with open(train_path, 'r')as file:
        csv_reader = csv.reader(file, delimiter=',')
        res_train = [row[1:-1] for row in csv_reader][2:]
        row_train = len(res_train)
        traindata = np.array(res_train, dtype=np.float32)[21600:]
        print(traindata.shape)
    
    # clean some constant series
    epsilo = 0.001
    data_min = np.min(traindata, axis=0)
    data_max = np.max(traindata, axis=0)+epsilo
    for i in range(len(data_max)):
        if data_max[i] - data_min[i] < 10 * epsilo:
            data_min[i] = data_max[i]
            data_max[i] = 1 + data_max[i]
            
    # MinMax Scaler & clip
    train_data = (traindata - data_min)/(data_max - data_min)
    train_data = np.clip(train_data, a_min=-1.0, a_max=3.0)
    
    with open(test_path, 'r')as file:
        csv_reader = csv.reader(file, delimiter=',')
        res_test = [row[1:-1] for row in csv_reader][1:]
        row_test = len(res_test)
        testdata = np.array(res_test, dtype=np.float32)
        print(testdata.shape)
    
    test_data = (testdata - data_min)/(data_max - data_min)
    test_data = np.clip(test_data, a_min=-1.0, a_max=3.0)
    
    with open(test_path, 'r')as file:
        csv_reader = csv.reader(file, delimiter=',')
        res = [row[-1]for row in csv_reader][1:]
        label_ = [0 if i == "Normal" else 1 for i in res]
        label = np.array(label_)
    
    np.save("%s/SWaT_train.npy"%(data_path), train_data)
    np.save("%s/SWaT_test.npy"%(data_path), test_data)
    np.save("%s/SWaT_label.npy"%(data_path), label)
  
    
def gen_WADI(path="./datasets/WADI"):
    train_path = os.path.join(path, "WADI_14days_new.csv")
    test_path = os.path.join(path, "WADI_attackdataLABLE.csv")
    nan_cols = []
    
    with open(train_path, "r") as file:
        csv_reader = csv.reader(file, delimiter=',')
        res_train = [row[3:] for row in csv_reader][1:]
        res_train = np.array(res_train)[21600:]
        row_train, col_train = len(res_train), len(res_train[0])
        for j in range(res_train.shape[1]):
            for i in range(res_train.shape[0]):
                if res_train[i][j] == "1.#QNAN" or res_train[i][j] == '':
                    nan_cols.append(j)
                    break
        res_train = np.delete(res_train, nan_cols, axis=1)
        res_train = res_train.astype(np.float32)
                    
    traindata = res_train
    epsilo = 0.001
    data_min = np.min(traindata, axis=0)
    data_max = np.max(traindata, axis=0)+epsilo
    for i in range(len(data_max)):
        if data_max[i] - data_min[i] < 10 * epsilo:
            data_min[i] = data_max[i]
            data_max[i] = 1 + data_max[i]
    traindata = (traindata - data_min)/(data_max - data_min)
    train_data = np.clip(traindata, a_min=-1.0, a_max=2.0)
    
    with open(test_path, 'r') as file:
        csv_reader = csv.reader(file, delimiter=',')
        res_test = [row[3:-1] for row in csv_reader][2:]
        res_test = np.array(res_test)
        row_test, col_test = len(res_test), len(res_test[0])
        for i in range(row_test):
            for j in range(col_test):
                if res_test[i][j] == '':
                    res_test[i][j] = 0
        res_test = np.delete(res_test, nan_cols, axis=1)
        res_test = res_test.astype(np.float32)            
    test_data = (res_test - data_min)/(data_max - data_min)
    test_data = np.clip(test_data, a_min=-1.0, a_max=2.0)
    
    with open(test_path, 'r') as file:
        csv_reader = csv.reader(file, delimiter=',')
        res = [row[-1] for row in csv_reader][2:]
        label = np.array(res, dtype=np.float32)
        for i in range(len(label)):
            if label[i] <= 0:
                label[i] = 1
            else:
                label[i] = 0
                
    np.save("%s/WADI_train.npy"%(data_path), train_data)
    np.save("%s/WADI_test.npy"%(data_path), test_data)
    np.save("%s/WADI_label.npy"%(data_path), label)
    
if __name__ == "__main__":
    gen_SMD("../datasets/ServerMachineDataset")
    gen_SWaT("../datasets/SWAT")
    gen_WADI("../datasets/WADIA2")
    
    
    