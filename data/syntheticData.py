

import numpy as np
import scipy.stats as stats
from sklearn.decomposition import dict_learning
import pickle as pk

import os
class syntheticData(object):
    '''
    classdocs
    '''


    def __init__(self, path, params = None, save_name = "default", flatten=False):
        
        '''
        Constructor
        '''
        
        self.path = path
        self.flatten = flatten
        self.num_classes = 2
        
        
        if not params == None:
            for key in params.keys():
                
                
                self.__setattr__(key, params[key])
        else:
            self.codeword_length = 10
        
            self.num_mean_vals = 8
            self.num_std_vals = 6
            self.keep_rate = 0.5
            self.num_classes = 2
            self.num_nonzero_per_class=2
            
        self.saved_instance_name = save_name

            
        self.num_features = self.codeword_length
        
        self.num_test_samples = 1000
        self.raw_image_shape = (self.codeword_length, 1,1)
        self.num_codewords = int(self.num_mean_vals * self.num_std_vals * self.keep_rate)
        
        self.generate_dictionary_elements()
        self.generate_class_feature_distributions()
        
    

    def generate_dictionary_elements(self):
        
        try:
            saved = pk.load(open("%s/dictElements_%s.pk" % (self.path, self.saved_instance_name), 'rb'))
            self.dictionary_matrix = saved
            print("loaded pre-generated dictionary elements.")
            return
        except: 
            print("no pre-generated dictionary elements found. generating new ones.")
            
        
        """
        Use a set of mean and std values for gaussian centroids as dictionary elements.
        """
        
        dictionary_matrix = np.zeros((int(self.num_mean_vals * self.num_std_vals), self.codeword_length))
        
        codeword_index = 0
        for mean in np.linspace(0, stop = self.codeword_length, num = self.num_mean_vals):
            for std in np.linspace(self.codeword_length/15, stop = self.codeword_length/3, num = self.num_std_vals):
                
                dictionary_matrix[codeword_index] = stats.norm.pdf(np.arange(self.codeword_length), mean, std)
                
                codeword_index += 1
        
        

        num_kept = int(self.num_codewords)
        
        if self.keep_rate < 1.0:
            
            kept = np.random.choice(range(int(self.num_mean_vals * self.num_std_vals)), num_kept, replace = False)
            
            self.dictionary_matrix =  dictionary_matrix[kept,:]
        else: 
            self.dictionary_matrix =  dictionary_matrix

        """
        Codewords have unit norm
        """
        norms = np.sqrt( np.sum(self.dictionary_matrix ** 2, axis = 1))  
        self.dictionary_matrix = self.dictionary_matrix / norms.reshape((-1,1))
    
        try:
            os.mkdir(self.path)
        except: pass
        pk.dump(self.dictionary_matrix, open("%s/dictElements_%s.pk" % (self.path, self.saved_instance_name),'wb'))
                             
        return self.dictionary_matrix
    
    
    
    def generate_class_feature_distributions(self):
    
        
        try:
            saved = pk.load(open("%s/classFeatures_%s.pk" % (self.path, self.saved_instance_name), 'rb'))
            self.class_feature_distributions = saved
            print("loaded pre-generated class feature distributions.")
            return
        except: 
            print("no pre-generated class feature distrs found. generating new ones.")

        
        self.class_feature_distributions = []
        for i in range(self.num_classes):
            mean_vec = np.zeros((self.num_codewords))
            mean_vec_inds = np.random.choice(range(self.num_codewords), size = self.num_nonzero_per_class, replace = False)
            mean_vec[mean_vec_inds] = 1
            mean_vec_inds = np.random.choice(range(self.num_codewords), size = self.num_nonzero_per_class, replace = False)
            mean_vec[mean_vec_inds] = -1
            
            cov_matr = (np.eye(self.num_codewords) + (np.random.rand(self.num_codewords, self.num_codewords) - 0.5)/self.num_codewords) /self.num_codewords**1.5
            cov_matr += np.transpose(cov_matr)
            
            self.class_feature_distributions.append((mean_vec, cov_matr))
        
        try:
            os.mkdir(self.path)
        except: pass
        pk.dump(self.class_feature_distributions, open("%s/classFeatures_%s.pk" % (self.path, self.saved_instance_name),'wb'))
            
        
        
    def getData(self, numExample, return_decoded=True):
        
        labels = np.floor(np.arange(0, self.num_classes, self.num_classes/ numExample))
        
        image_codes =np.vstack([
            
            np.random.multivariate_normal(
                self.class_feature_distributions[i][0], 
                self.class_feature_distributions[i][1], 
                size = int(numExample/self.num_classes))
            for i in range(self.num_classes)]
         )
                           
        if return_decoded:         
            images = decode(image_codes, self.dictionary_matrix)
            images = np.reshape(images, (numExample,) + self.raw_image_shape)
        
        else:
            images = image_codes
            
        random_order = np.arange(numExample)
        np.random.shuffle(random_order)
        
        out_feature_length = self.num_features if return_decoded else self.num_codewords
        return (images[random_order].reshape((numExample, out_feature_length) if self.flatten else (numExample, out_feature_length, 1, 1)), labels[random_order])
    
    
    def getTestData(self):
        
        images, labels = self.getData(numExample = self.num_test_samples, return_decoded=True)
        return images.reshape((self.num_test_samples, self.num_features)), labels
        
    def getNormSample(self, num_sample, return_decoded=True):
        
        dat = self.getData(num_sample, return_decoded)[0].reshape((num_sample,self.num_features)) #features only
        norm_dat = (dat - np.mean(dat , axis=1, keepdims=True))/np.linalg.norm(dat , axis=1, keepdims=True)
        return norm_dat


    def getPCA(self, num_init, num_sample=None):
        assert(num_init <= np.min(self.train_images.shape))
        if(num_sample is None):
            num_sample = num_init * 5

        data = self.getNormSample(num_sample)
        [u, s, v] = np.linalg.svd(np.transpose(data))
        return np.transpose(u[:, :num_init])

    def getDict(self, num_init, alpha, num_sample=None):
        #assert(num_init <= np.min(self.raw_image_shape))
        if(num_sample is None):
            num_sample = 5000
        data = self.getNormSample(num_sample)
        
        
        print("Running sklearn dict_learn for initial dictionary")
        print(type(data), data.shape)
        dictionary = dict_learning(data, num_init, alpha, verbose=2, n_jobs=-1, max_iter=50)[1]
        print("Done")
        return dictionary
    
def decode(codes, dictionary):
    """
    codes is NxC
    dictionary is CxD
    for N = num_elements, C = num_codewords, D = raw data dimension 
    """
    c, d = dictionary.shape
    codes = codes.reshape(-1, c)
    
    return np.dot(codes, dictionary)
    

def plot_dictionary_elements(dictionary_matrix):
    
    num_el, length = dictionary_matrix.shape
    x = np.linspace(0, length, length)
    
    ind = 0
    for i in np.random.choice(range(num_el), 5, replace=False):
        ax = plt.subplot2grid((5,1), (ind,0))
        ax.plot(x, dictionary_matrix[i],c="kyrgb"[ind])
        ind+=1
    plt.show()


if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    from sklearn.decomposition import sparse_encode
    from sklearn.decomposition import PCA


    save_name = "easy1"
    
    if save_name == "easy1":
        data_params = {
    
        "codeword_length": 20,
    
        "num_mean_vals": 8,
        "num_std_vals": 6,
        "keep_rate": 0.5,
       "num_nonzero_per_class": 2
    }

 
    save_name = "easy2"
    
    if save_name == "easy2":
        data_params = {
    
        "codeword_length": 30,
    
        "num_mean_vals": 10,
        "num_std_vals": 10,
        "keep_rate": 0.2,
       "num_nonzero_per_class": 3
    }

    save_name = "easy3"
    
    if save_name == "easy3":
        data_params = {
    
        "codeword_length": 30,
    
        "num_mean_vals": 10,
        "num_std_vals": 3,
        "keep_rate": 0.5,
       "num_nonzero_per_class": 2
    }

    save_name = "short2"
    
    if save_name == "short2":
        data_params = {
    
        "codeword_length": 10,
    
        "num_mean_vals": 10,
        "num_std_vals": 3,
        "keep_rate": 0.5,
       "num_nonzero_per_class": 3
    }
   
    sd = syntheticData("/home/slundquist/mountData/datasets/synthetic", params = data_params, save_name=save_name)
    a = sd.getData(256, return_decoded = True)
    
    plot_dictionary_elements(sd.dictionary_matrix)
    
    pca = PCA(n_components = 2)
    res = pca.fit_transform(a[0].reshape(256,sd.num_features))
    print("res.shape", res.shape)
    colorstring="".join(["b" if a[1][i] else "r" for i in range(len(a[1]))])
    
    
    plt.scatter(res[:,0], res[:,1], c=colorstring, s= 8)
    plt.show()
    
    
    a = sd.getData(256, return_decoded = False)
    colorstring="".join(["b" if a[1][i] else "r" for i in range(len(a[1]))])
    
    pca = PCA(n_components = 2)
    res = pca.fit_transform(a[0].reshape(256,sd.num_codewords))
    plt.scatter(res[:,0], res[:,1], c=colorstring, s = 8)
    plt.show()
    
    
    ax2 = plt.subplot2grid((2,2), (0,1))
    ax1 = plt.subplot2grid((2,2), (0,0))
    ax3 = plt.subplot2grid((2,2), (1,0))
    ax3.set_xlabel("pdf w.r.t class_distr 1")
    ax3.set_ylabel("pdf w.r.t class_distr 2")
    ax3.legend(["class 1", "class 2"])
    
    print(a[1])
    
    print(a[1]==0)
    
    print(a[1]==1)
    
    print(a[0][a[1]==0][0].shape)
    
    a = sd.getData(256, return_decoded = True)
    
    
    axmin = np.min(a[0])-0.1
    axmax = np.max(a[0])+0.1
    for i in range(10):
        
        ax1.plot(np.linspace(start = 0, stop = sd.num_features, num = sd.num_features),a[0][a[1]==0][i].reshape(-1))
        ax1.set_ylim([axmin, axmax])
    print(a[0][a[1]==1].shape)
    
    for i in range(10):
        ax2.plot(np.linspace(start = 0, stop = sd.num_features, num = sd.num_features),a[0][a[1]==1][i].reshape(-1))
        ax2.set_ylim([axmin, axmax])
        
    plt.show()
    
    
    ###
    recoded = sparse_encode(X=decoded, dictionary=dictionary, n_nonzero_coefs= 20, alpha = 0.001)
    print("recoded.shape", recoded.shape)
    for i in range(num_datapoints):
        print("code/recode:")
        for k in range(num_codewords):
            print(round(codes[i,k],3), "   ",  round(recoded[i,k],3))
    ###
    
    
#def generate_codes(length):
    