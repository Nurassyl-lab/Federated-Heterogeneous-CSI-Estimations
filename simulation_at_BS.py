# In[0]
'Description:'
'UE - user equipment, BS - base station'

'UE sent encoded features'
'BS has 3 pre-trained decoders'

'UE encoded information about which encoder model was used inside encoded features'
'since encoders and decoders were trained together, this information will tell'
'which decoder model to use'

from utils import *

# In[1]
'Define initial variables'

'define heterogeneity level'
heterogeneity = 100

'define # of classes which is also corresponds to number of local models'
n_classes = 3

'define path from where to load models'
path = 'FED_HETERO_CSI/MODELS/het_'+str(heterogeneity)+'/'

'define UE class'
class base_station():
    def __init__(self):
        _, self.im_decoder1, _, _, _, _ = define_VAE_CSI_MODEL()
        _, _, _, self.rl_decoder1, _, _ = define_VAE_CSI_MODEL()

        _, self.im_decoder2, _, _, _, _ = define_VAE_CSI_MODEL()
        _, _, _, self.rl_decoder2, _, _ = define_VAE_CSI_MODEL()

        _, self.im_decoder3, _, _, _, _ = define_VAE_CSI_MODEL()
        _, _, _, self.im_decoder2, _, _ = define_VAE_CSI_MODEL()

    def load_models(self):
        self.im_decoder1 = keras.models.load_model(path + 'class0_imag_decoder')
        self.rl_decoder1 = keras.models.load_model(path + 'class0_real_decoder')

        self.im_decoder2 = keras.models.load_model(path + 'class1_imag_decoder')
        self.rl_decoder2 = keras.models.load_model(path + 'class1_real_decoder')

        self.im_decoder3 = keras.models.load_model(path + 'class2_imag_decoder')
        self.rl_decoder3 = keras.models.load_model(path + 'class2_real_decoder')
        
# In[2]
'Define UE with 3 encoders and 1 classification model'

BS = base_station()
BS.load_models()
# In[3]
'load encoded features'

CSI = np.load('FED_HETERO_CSI/BS_data_original/het'+str(heterogeneity)+'/eval_true.npy')

CSI_encoded = []
for i in range(len(CSI)):
    CSI_encoded.append(np.load('FED_HETERO_CSI/complex_encoded_features/het'+str(heterogeneity)+'/'+str(i)+'.npy'))

# In[4]
'Run simulation'

BS_out = []
for data in CSI_encoded:
    real_data = data[:40].real
    imag_data = data[:40].imag
    
    if int(data[-1].real) == 1:
        im_pred = BS.im_decoder1.predict(imag_data.reshape(1, 40)).reshape(64, 100, 1)
        rl_pred = BS.rl_decoder1.predict(real_data.reshape(1, 40)).reshape(64, 100, 1)
        BS_out.append(np.stack((rl_pred, im_pred)))
        
    elif int(data[-1].real) == 2:
        im_pred = BS.im_decoder2.predict(imag_data.reshape(1, 40)).reshape(64, 100, 1)
        rl_pred = BS.rl_decoder2.predict(real_data.reshape(1, 40)).reshape(64, 100, 1)
        BS_out.append(np.stack((rl_pred, im_pred)))
        
    elif int(data[-1].real) == 3:
        im_pred = BS.im_decoder3.predict(imag_data.reshape(1, 40)).reshape(64, 100, 1)
        rl_pred = BS.rl_decoder3.predict(real_data.reshape(1, 40)).reshape(64, 100, 1)
        BS_out.append(np.stack((rl_pred, im_pred)))

# In[5]
'evaluate'

mse_real = []
mse_imag = []
mse_mag = []

for i in range(len(CSI)):
    real_orig = CSI[i, 0, :, :, :].reshape(64,100)
    imag_orig = CSI[i, 1, :, :, :].reshape(64,100)
    
    real_rec = BS_out[i][0, :, :, :].reshape(64,100)
    imag_rec = BS_out[i][1, :, :, :].reshape(64,100)
    
    mse_real.append(mean_squared_error(real_orig, real_rec))
    mse_imag.append(mean_squared_error(imag_orig, imag_rec))
    
    comp_orig = real_orig + 1j*imag_orig
    comp_rec = real_rec + 1j*imag_rec
    
    mse_mag.append(mean_squared_error(np.abs(comp_orig), np.abs(comp_rec)))

print(f'mse_real {np.mean(mse_real)}, mse_imag {np.mean(mse_imag)}, mse_mag {np.mean(mse_mag)}')