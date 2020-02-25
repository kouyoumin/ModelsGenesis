import os
import shutil

class models_genesis_config:
    model = "Unet3D"
    suffix = "genesis_chest_ct"
    exp_name = model + "-" + suffix
    
    # data
    data = "/mnt/dataset/shared/zongwei/LUNA16/Self_Learning_Cubes"
    train_fold=[0,1,2,3,4]
    valid_fold=[5,6]
    test_fold=[7,8,9]
    hu_min = -1000.0
    hu_max = 1000.0
    scale = 32
    input_rows = 64
    input_cols = 64 
    input_deps = 32
    nb_class = 1
    
    # model pre-training
    verbose = 1
    weights = None
    batch_size = 6
    optimizer = "sgd"
    workers = 10
    max_queue_size = workers * 4
    save_samples = "png"
    nb_epoch = 10000
    patience = 50
    lr = 2.0

    # image deformation
    nonlinear_rate = 0.9
    paint_rate = 0.9
    outpaint_rate = 0.8
    inpaint_rate = 1.0 - outpaint_rate
    local_rate = 0.5
    flip_rate = 0.4
    
    # logs
    model_path = "pretrained_weights"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    logs_path = os.path.join(model_path, "Logs")
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    
    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")


class models_genesis_config_mr:
    model = "Unet3D"
    suffix = "genesis_liver_mr"
    sample_path = '/home/kouyoumin/hdd/kouyoumin/ModelsGenesis/pytorch/sample'
    exp_name = model + "-" + suffix
    
    # data
    data = "/home/kouyoumin/Datasets/CGMH"
    train_fold=[1,2,4,5,7,10,11,12,16,17,20,21,22,24,27,28,29,30,32,33,35,36,38,43,44,45,48,50,51,52,53,54,55]
    valid_fold=[56,57,58,59,62,63,64,66,69]
    test_fold=[56,57,58,59,62,63,64,66,69]
    hu_min = -1000.0
    hu_max = 1000.0
    scale = 32
    input_rows = 128
    input_cols = 128 
    input_deps = 32
    nb_class = 1
    
    # model pre-training
    verbose = 1
    weights = None
    #weights = '/home/kouyoumin/hdd/kouyoumin/ModelsGenesis/pytorch/old/pretrained_weights/Genesis_Liver_MR.pt'
    pretrained ='/home/kouyoumin/hdd/kouyoumin/ModelsGenesis/pytorch/old/pretrained_weights/Genesis_Liver_MR.pt'
    batch_size = 6
    optimizer = "sgd"
    workers = 10
    max_queue_size = workers * 4
    save_samples = "png"
    nb_epoch = 10000
    patience = 50
    lr = 0.1

    # image deformation
    nonlinear_rate = 0.9
    paint_rate = 0.9
    outpaint_rate = 0.8
    inpaint_rate = 1.0 - outpaint_rate
    local_rate = 0.5
    flip_rate = 0.4
    
    # logs
    model_path = "pretrained_weights"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    logs_path = os.path.join(model_path, "Logs")
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    
    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
