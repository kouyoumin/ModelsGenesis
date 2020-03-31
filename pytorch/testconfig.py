class models_genesis_config_mr2_test:
    model = "Unet3D"
    
    # data
    data = "/home/kouyoumin/CGMH"
    # Problem folds: 22, 33, 45
    test_fold=[65,66,67,68,69,70]
    nb_class = 2
    
    verbose = 1
    weights = 'hbp2.pt'
