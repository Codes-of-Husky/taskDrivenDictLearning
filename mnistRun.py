from data.mnistData import mnistData
from models.tbLearn import tbLearn
import pdb

path = "/home/slundquist/mountData/datasets/mnist"
dataObj = mnistData(path, flatten=True)

#Parameter class
class TBDictParams(object):
    ##Bookkeeping params
    #Base output directory
    out_dir            = "/home/slundquist/mountData/tbLearn/"
    #Inner run directory
    run_dir            = out_dir + "/mnist/"

    #Save parameters
    save_period        = 100000
    #output plots directory
    plot_period        = 10000
    eval_period        = 1718 # 1 epoch
    #Progress step
    progress           = 100
    #Controls how often to write out to tensorboard
    write_step         = 100
    #Flag for loading weights from checkpoint
    load               = False
    load_file          = ""
    #Device to run on
    device             = "/gpu:1"
    #data params
    image_shape        = dataObj.raw_image_shape #Can be None
    num_classes        = dataObj.num_classes
    num_features       = dataObj.num_features

    #Model params
    num_steps = 40000 #T in paper
    dict_size = 300
    batch_size = 256
    l1_weight = 0.1 #lambda_1 in paper
    l2_weight = 0    #lambda_2 in paper
    weight_decay = 1e-5 #v in paper
    init_weights = dataObj.getPCA(dict_size)

    #LCA params
    sc_lr = 3e-3

    decay_time = num_steps/10 #Time period which learning rate starts annealing
    start_lr = 1.0 #Learning rate

#Initialize params
params = TBDictParams()
tfObj = tbLearn(params)
tfObj.trainModel(dataObj)


