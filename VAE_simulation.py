'Simulation of training decentralized(local) models'
'Architecture: Variational Autoencoder'

'this code may provide you with information such as:'
' 1) local model performance for training and validation'
' 2) dataset structure'
' 3) local model perfromance across all heterogeneity levels'

from utils import *
#==============================================================================
"initial parameters"

#To run a whole simulation uncomment 2 lines below
#your local model performance will be averaged over the length of array "iterations"
#iterations are used to set a randomizing seed and get reproducible outputs

# iterations = [0, 1, 2, 3, 4, 5, 6 ,7 ,8 , 9]
# heterogeneity = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

#To run a particular simulation uncomment 2 lines below
iterations = [0]
heterogeneity = [50]
    
n_classes = 3 #depends on how many decentralized models you aim to train, 
              #note that your dataset should have same amount of classes
VAE = True

train_model = True # if you have pre-trained models set train_models to False

# norms = np.array([np.arange(40,45), np.arange(50,55), np.arange(60,65)])

'start training and estimating'
for het in heterogeneity:
    
    " *Files "
    paths_to_load = []
    paths_to_save = []
    for clas in range(n_classes):
        paths_to_load.append('unbalanced data/CLASS_'+str(clas+1)+'/het'+str(het)+'%/')
    paths_to_save = 'unbalanced data/RESULTS/het'+str(het)+'%/train_data/'
    
    ' *Create dataframe to save results '
    '.csv files will be updated with your performances...'
    df = pd.read_csv('het'+str(het)+'.csv')
    df = df.set_index(['LOSSES'])
    ' *Load data & Start training '
    for iteration in iterations:      
        ' *set initialization parameters '
        seed(iteration)
        tensorflow.random.set_seed(iteration)
        
        " *Create server "
        main_server = SERVER()
        
        ' *Load server-side data '
        # main_server.load_data('FED_HETERO_CSI/balanced_data/')  
        
        " *Assign classes to a server "
        class_list = [CLASS(num) for num in range(n_classes)]
        main_server.set_classes(class_list)
        
        ' *Assign model to each class '
        for clas in main_server.classes:
            clas.define_vae()

        ' *Load data into each class '
        for clas in main_server.classes:
            print('loading data into class',clas.number)
            print('loading path',paths_to_load[clas.number])
            clas.load_data(paths_to_load[clas.number])

        ' *Train or Load models '
        if train_model is False:
            ' *each model one-by-one '
            for clas in main_server.classes:
                print("Loading vae model for class", clas.number)
                clas.model = keras.models(paths_to_load+'model')
        else:
            for clas in main_server.classes:
                print(clas.number,"class is training")
                
                # for VAE the data is going to be split into real and complex parts, 
                # and will be trained separately
                train_data_real = clas.train_data[:,0,:,:,:]
                train_data_imag = clas.train_data[:,1,:,:,:]
                val_data_real = clas.val_data[:,0,:,:,:]
                val_data_imag = clas.val_data[:,1,:,:,:]
                
                'save history of training real values'
                clas.hist_real = clas.vae_real.fit(train_data_real, epochs=100, batch_size=32)
                clas.hist_imag = clas.vae_imag.fit(train_data_imag, epochs=50, batch_size=32)
                    
    'if you have trained local model by yourself, you might also want to save them'
    # for clas in main_server.classes:
    #     clas.model.save('model'+str(clas.number))
    
    'evaluation using validation data'
    for clas in main_server.classes:
        print('Class number', clas.number)
        if VAE:
            test_data_real = clas.val_data.real.reshape(len(clas.val_data), 64, 100, 1)
            test_data_imag = clas.val_data.imag.reshape(len(clas.val_data), 64, 100, 1)                   
            
            _,_,encoded_real = clas.vae_encoder_real.predict(test_data_real)
            _,_,encoded_imag = clas.vae_encoder_imag.predict(test_data_imag)
                                
            decoded_real =  clas.vae_decoder_real.predict(encoded_real)
            decoded_imag = clas.vae_decoder_imag.predict(encoded_imag)
            
            mse_real = []
            mse_imag = []
            mse_mag = []
            mse_phs = []
            for i in range(len(test_data_real)):
                y_true_real = test_data_real[i].reshape(64,100)
                y_pred_imag = decoded_real[i].reshape(64,100)
                
                y_true_real = test_data_imag[i].reshape(64,100)
                y_pred_imag = decoded_imag[i].reshape(64,100)
                
                mse_real.append(mean_squared_error(y_true_real, y_pred_real))
                mse_imag.append(mean_squared_error(y_true_imag, y_pred_imag))
                
                y_true_compl = y_true_real + 1j*y_true_imag
                y_pred_compl = y_pred_real + 1j*y_pred_imag
                
                y_true_magnitude = np.abs(y_true_compl)
                y_pred_magnitude = np.abs(y_pred_compl)
                
                mse_mag.append(mean_squared_error(y_true_magnitude, y_pred_magnitude))
                
                'check again'
                y_true_phase = np.arctan(np.sqrt(y_true_compl.imag / y_true_compl.real))
                y_pred_phase = np.arctan(np.sqrt(y_pred_compl.imag / y_pred_compl.real))
                
                ''
                mse_phs.append(mean_squared_error(y_true_phase, y_pred_phase))
                
            print(np.mean(mse_real))
            print(np.mean(mse_imag))
            print(np.mean(mse_mag))
            print(np.mean(mse_phs))
    
    
    df.to_csv('VAE_simulation'+str(het)+'.csv')  
