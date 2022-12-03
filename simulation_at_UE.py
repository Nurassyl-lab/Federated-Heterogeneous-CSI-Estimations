# In[0]
'Description:'
'UE - user equipment, BS - base station'

'UE chooses right encoder for received CSI data'
'3 pre-trained encoders are delivered to the UE along with 1 classification model'
'UE encodes CSI data using 3 encoders respectively getting 3 different outputs(encoded features).'

'UE should only choose 1 of 3 encoded features to send to BS,'
'so there is reason to send a data that was encoded in a proper way.'

'Since 3 encoders encode data in different ways we cannot choose a encoded features to send to BS.'
'Classification model is used exactly for this task.'

from utils import *
from UE_classifier import *

# In[1]
'Define initial variables'

'define heterogeneity level'
heterogeneity = 100

'define # of classes which is also corresponds to number of local models'
n_classes = 3

'define path from where to load models'
path = 'FED_HETERO_CSI/MODELS/het_'+str(heterogeneity)+'/'

'2d array helps to evaluate accuracy of classification model'
normas = np.array([
                    [40, 41, 42, 43, 44, 45],
                    [50, 51, 52, 53, 54, 55],
                    [60, 61, 62, 63, 64, 65]
                ])
    
'define UE class'
class user_equipment():
    def __init__(self):
        self.im_encoder1, _, _, _, _, _ = define_VAE_CSI_MODEL()
        _, _, self.rl_encoder1, _, _, _ = define_VAE_CSI_MODEL()

        self.im_encoder2, _, _, _, _, _ = define_VAE_CSI_MODEL()
        _, _, self.rl_encoder2, _, _, _ = define_VAE_CSI_MODEL()

        self.im_encoder3, _, _, _, _, _ = define_VAE_CSI_MODEL()
        _, _, self.rl_encoder3, _, _, _ = define_VAE_CSI_MODEL()

        self.classifier = def_classifier()

    def load_models(self):
        self.im_encoder1 = keras.models.load_model(path + 'class0_imag_encoder')
        self.rl_encoder1 = keras.models.load_model(path + 'class0_real_encoder')

        self.im_encoder2 = keras.models.load_model(path + 'class1_imag_encoder')
        self.rl_encoder2 = keras.models.load_model(path + 'class1_real_encoder')

        self.im_encoder3 = keras.models.load_model(path + 'class2_imag_encoder')
        self.rl_encoder3 = keras.models.load_model(path + 'class2_real_encoder')

        self.classifier = keras.models.load_model(path + 'classification_model')

# In[2]
'Define UE with 3 encoders and 1 classification model'

UE = user_equipment()
UE.load_models()

# In[3]
'load CSI data'

CSI = np.load('FED_HETERO_CSI/BS_data_original/het'+str(heterogeneity)+'/eval_true.npy')

# In[4]
'Run simulation'

'send input CSI data to UE'
i = 0
accuracy = 0
for data in CSI:
    input_classifier = []
    'UE encodes received CSI using 3 encoders'
    im_mean1, im_var1, imag_enc1 = UE.im_encoder1.predict(data[0,:,:,:].reshape(1, 64, 100, 1))
    rl_mean1, rl_var1, real_enc1 = UE.rl_encoder1.predict(data[1,:,:,:].reshape(1, 64, 100, 1))
    input_classifier.append([np.mean(im_mean1), np.mean(im_var1), np.mean(imag_enc1), np.var(imag_enc1), np.mean(rl_mean1), np.mean(rl_var1), np.mean(real_enc1), np.var(real_enc1)])

    
    im_mean2, im_var2, imag_enc2 = UE.im_encoder2.predict(data[0,:,:,:].reshape(1, 64, 100, 1))
    rl_mean2, rl_var2, real_enc2 = UE.rl_encoder2.predict(data[1,:,:,:].reshape(1, 64, 100, 1))
    input_classifier.append([np.mean(im_mean2), np.mean(im_var2), np.mean(imag_enc2), np.var(imag_enc2), np.mean(rl_mean2), np.mean(rl_var2), np.mean(real_enc2), np.var(real_enc2)])


    im_mean3, im_var3, imag_enc3 = UE.im_encoder3.predict(data[0,:,:,:].reshape(1, 64, 100, 1))
    rl_mean3, rl_var3, real_enc3 = UE.rl_encoder3.predict(data[1,:,:,:].reshape(1, 64, 100, 1))
    input_classifier.append([np.mean(im_mean3), np.mean(im_var3), np.mean(imag_enc3), np.var(imag_enc3), np.mean(rl_mean3), np.mean(rl_var3), np.mean(real_enc3), np.var(real_enc3)])

    pred_clas = np.argmax(UE.classifier.predict(np.array(input_classifier).reshape(1, 3, 8, 1)))

    'find to which class/model received data truly belongs'
    true_class = np.where(normas == np.floor(np.linalg.norm(data[0, :, :, :] + 1j*data[1, :, :, :])))[0][0]
    
    if true_class == pred_clas:
        accuracy += 1
    
    'send encoded features to BS'
    if pred_clas == 0:
        tmp = (real_enc1 + 1j*imag_enc1).reshape(40)
        np.save('FED_HETERO_CSI/complex_encoded_features/het'+str(heterogeneity)+'/'+str(i)+'.npy', np.r_[tmp, 1])#let BS know which encoder was used in order to select a proper decoder
    elif pred_clas == 1:
        tmp = (real_enc2 + 1j*imag_enc2).reshape(40)
        np.save('FED_HETERO_CSI/complex_encoded_features/het'+str(heterogeneity)+'/'+str(i)+'.npy', np.r_[tmp, 2])
    elif pred_clas == 2:
        tmp = (real_enc3 + 1j*imag_enc3).reshape(40)
        np.save('FED_HETERO_CSI/complex_encoded_features/het'+str(heterogeneity)+'/'+str(i)+'.npy', np.r_[tmp, 3])
    i+=1

print(accuracy / i)