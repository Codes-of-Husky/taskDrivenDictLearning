import matplotlib
matplotlib.use('Agg')
from data.syntheticData import syntheticData
from models.tbLearn import tbLearn


path = "/home/slundquist/mountData/datasets/synthetic"


#save_name = "default"
#save_name = "easy2"
#save_name = "easy3"

save_name = "short2"


if save_name == "default":

    data_params = {
    
        "codeword_length": 10,
    
        "num_mean_vals": 8,
        "num_std_vals": 6,
        "keep_rate": 0.5,
        "num_nonzero_per_class": 1
        }

if save_name == "easy1":
        data_params = {
    
        "codeword_length": 20,
    
        "num_mean_vals": 8,
        "num_std_vals": 6,
        "keep_rate": 0.5,
       "num_nonzero_per_class": 2
    }

if save_name == "easy2":
        data_params = {
    
        "codeword_length": 30,
    
        "num_mean_vals": 10,
        "num_std_vals": 10,
        "keep_rate": 0.2,
       "num_nonzero_per_class": 3
    }
        
    
if save_name == "easy3":
        data_params = {
    
        "codeword_length": 30,
    
        "num_mean_vals": 10,
        "num_std_vals": 3,
        "keep_rate": 0.5,
       "num_nonzero_per_class": 2
    }
    
if save_name == "short2":
        data_params = {
    
        "codeword_length": 10,
    
        "num_mean_vals": 10,
        "num_std_vals": 3,
        "keep_rate": 0.5,
       "num_nonzero_per_class": 3
    }
dataObj = syntheticData(path, params = data_params, save_name = save_name, flatten=True)

#Parameter class
class TBDictParams(object):
    ##Bookkeeping params
    #Base output directory
    out_dir            = "/home/slundquist/mountData/tbLearn/"
    #Inner run directory
    run_dir            = out_dir + "/synth_" + save_name +"/"

    #Save parameters
    save_period        = 10000
    #output plots directory
    plot_period        = 5000
    eval_period        = 1000 # 1 epoch
    #Progress step
    progress           = 100
    #Controls how often to write out to tensorboard
    write_step         = 100
    #Flag for loading weights from checkpoint
    load               = False
    load_file          = ""
    #Device to run on
    device             = "/gpu:0"
    #data params
    image_shape        = dataObj.raw_image_shape #Can be None
    num_classes        = dataObj.num_classes
    num_features       = dataObj.num_features

    #Model params
    #initial_train_W = 3000
    initial_train_W = 0
    num_steps = 40000 + initial_train_W #T in paper
    dict_size = 20
    batch_size = 256
    l1_weight = 0.15 #lambda_1 in paper
    l2_weight = 0    #lambda_2 in paper
    weight_decay = 1e-5 #v in paper
    init_weights = dataObj.getDict(dict_size, alpha=l1_weight)

    #LCA params
    sc_lr = 3e-3

    decay_time = (num_steps-initial_train_W)/10 + initial_train_W #Time period which learning rate starts annealing
    start_lr = .1 #Learning rate

#Initialize params
params = TBDictParams()
tfObj = tbLearn(params)
tfObj.trainModel(dataObj)