"Don't use this code for now"


from utils import *
#==============================================================================
"initial parameters"

#To run whole simulation uncomment 2 lines below
# iterations = [0, 1, 2, 3, 4, 5, 6 ,7 ,8 , 9]
# heterogeneity = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

#To run a particular simulation uncomment 2 lines below
iterations = [0]
heterogeneity = [50]
    

n_classes = 3
VAE = False
cnn = True

load_model = False # if you have pre-trained models set load_model to True


ep = 30
bs = 32

norms = np.array([np.arange(40,45), np.arange(50,55), np.arange(60,65)])

'start training and estimating'
for het in heterogeneity:
    
    " *Files "
    paths_to_load = []
    paths_to_save = []
    for clas in range(n_classes):
        paths_to_load.append('competition data/unbalanced data/CLASS_'+str(clas+1)+'/het'+str(het)+'%/')
    paths_to_save = 'competition data/unbalanced data/RESULTS/het'+str(het)+'%/train_data/'
    
    
    ' *Create dataframes for het. level '
    
    df = pd.read_csv('competition data/het'+str(het)+'.csv')
    df = df.set_index(['LOSSES'])
    ' *Load data & Start training '
    for iteration in iterations:
        # clust_count = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0}
        
        ' *set initialization parameters '
        seed(iteration)
        tensorflow.random.set_seed(iteration)
        
        
        " *Create server "
        main_server = SERVER()
     
        
        ' *Load server-side data '
        # main_server.load_data('FED_HETERO_CSI/balanced_data/')
        
        
        " *Assign classes to server "
        class_list = [CLASS(num) for num in range(n_classes)]
        main_server.set_classes(class_list)
        
        
        ' *Assign model to each class '
        if VAE:
            for clas in main_server.classes:
                clas.define_vae()
        elif cnn:
            for clas in main_server.classes:
                clas.define_cnn()
        else:
            for clas in main_server.classes:
                clas.define_dense_model()


        ' *Load data into each class '
        for clas in main_server.classes:
            print(paths_to_load[clas.number])
            clas.load_data(paths_to_load[clas.number])
        
            
        ' *Train or Load models '
        if load_model:
            
            ' *each model one-by-one '
            for clas in main_server.classes:
                print("Loading data for class", clas.number)
                clas.model = keras.models.load_model(paths_to_load+'model')
                
        else:
            for clas in main_server.classes:
                print(clas.number,"class is training")
                
                if VAE:
                    #for VAE the data is going to be split into real and complex parts and will be trained separately
                    train_data_real = clas.train_data.real.reshape(len(clas.train_data), 64, 100, 1)
                    train_data_imag = clas.train_data.imag.reshape(len(clas.train_data), 64, 100, 1)
                
                
                    'save history of training real values'
                    clas.hist = clas.vae_real.fit(train_data_real, epochs=100, batch_size=32)
                    # clas.vae_imag.fit(train_data_imag, epochs=30, batch_size=32)
                
                elif cnn:
                    clas.hist = clas.model.fit(clas.train_data, clas.train_data, epochs=10, batch_size=32, validation_data = (clas.val_data, clas.val_data))

for clas in main_server.classes:
    clas.model.save('model'+str(clas.number))
                    
for clas in main_server.classes:
    print('Class number', clas.number)
    if VAE:
        test_data_real = clas.val_data.real.reshape(len(clas.val_data), 64, 100, 1)
        test_data_imag = clas.val_data.imag.reshape(len(clas.val_data), 64, 100, 1)                   
        
        _,_,encoded_real = clas.vae_encoder_real.predict(test_data_real)
        _,_,encoded_imag = clas.vae_encoder_real.predict(test_data_imag)
                            
        decoded_real =  clas.vae_decoder_real.predict(encoded_real)
        decoded_imag = clas.vae_decoder_imag.predict(encoded_imag)
        
        mse = []
        for i in range(len(test_data_real)):
            mse.append(mean_squared_error(test_data_real[i].reshape(64,100), decoded_real[i].reshape(64,100)))
        np.mean(mse)
                        
    elif cnn:
        for cl_number in range(len(main_server.classes)):
            out = clas.model.predict(main_server.classes[cl_number].val_data)         
                        
            
            mse_mag = []
            # print('len', len(mse_mag))
            # mse_r = []
            # mse_i = []
            # mse_comp = []
            # normas = [[],[]]
            
            for i in range(len(main_server.classes[cl_number].val_data)):
                clas_data = (main_server.classes[cl_number].val_data[i, 0, :, :, 0] + 1j*main_server.classes[cl_number].val_data[i, 1, :, :, 0]).reshape(64,100)
                out_data = (out[i, 0, :, :, 0] + 1j*out[i, 1, :, :, 0]).reshape(64,100)
                
                mse_mag.append(mean_squared_error(np.abs(clas_data), np.abs(out_data)))
                # mse_r.append(mean_squared_error(clas_data.real, out_data.real))
                # mse_i.append(mean_squared_error(clas_data.imag, out_data.imag))
                # mse_comp.append(mse_r[i])
                # mse_comp.append(mse_i[i])
                # normas[0].append(np.linalg.norm(clas_data))
                # normas[1].append(np.linalg.norm(out_data))
            
            # print(clas.number, cl_number, np.mean(mse_mag))#############################################################################################
            # print((np.mean(mse_r) + np.mean(mse_i))/2)
            # print(np.mean(mse_comp))
            # print(np.min(normas[0]), np.max(normas[0]))
            # print(np.min(normas[1]), np.max(normas[1]))
            df.loc['model'+str(clas.number+1), 'data'+str(cl_number+1)] = np.mean(mse_mag)

def softmax(z):
    assert len(z.shape) == 2


    s = np.max(z, axis=1)
    s = s[:, np.newaxis]
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]
    return e_x / div

'dictionary to track class model choice/selection'
choice = {0:0,1:0,2:0}
right_coice = {0:0,1:0,2:0}

'list to store class model choice/selection'
arr_choice = []
arr_right = []

if cnn:
    'define server output array'
    server_out = []
    server_out_right = []
    for data in clas.val_data:
        'create complex number'
        compl = data[0, :, :, :] + 1j*data[1, :, :, :]
        
        'calculate L2 norm of a complex number'
        'this is done in order to determine which class model will be perfect for this type of output'
        compl = compl.reshape(64,100)
        norm = np.floor(np.linalg.norm(compl))
        
        'track the amount of type each class model is choosen'
        r = np.where(norms == norm)[0][0]
        right_coice[r] += 1
        'keep the model selection in arr_right in order to compare with arr_choice'
        arr_right.append(r)
        
    
        'define list to store outputs from your classes'
        out = []
        
        'define list to store softmax output from your classes'
        soft = []
        
        'get sigmoid and softmax outputs'
        for clas in main_server.classes:
            pred = clas.model.predict(data.reshape(1, 2, 64, 100, 1)).reshape(2,64,100,1)
            tmp_compl = (pred[0,:,:, :] + 1j*pred[1,:,:, :]).reshape(64,100)
            out.append(pred)
            soft.append(softmax(tmp_compl))
            
        'determine argmax values from sigmoid or softmax outputs'
        args = []
        for o in soft:
        # for o in out:
            tmp = o.flatten()
            args.append(tmp[np.argmax(tmp)])
            
        'determine which class model to use'
        clas_m = np.argmax(args)
        
        'add 1 to a class model that was choosen'
        choice[clas_m] += 1
        
        'append choice made in a list'
        arr_choice.append(clas_m)
        
        'append output of choosen class into a server_out'
        server_out.append(out[clas_m].reshape(2, 64, 100, 1))
        
        'append a right model to a server_out_right'
        server_out_right.append(out[r].reshape(2, 64, 100, 1))
    
    np_server_out = np.array(server_out)
    np_server_right_out = np.array(server_out_right)
    
    mse = []
    print('mean MSE loss of server_out')
    for i in range(len(np_server_out)):
        mag_class_data = np.abs((clas.val_data[i, 0, :, :, :] + 1j*clas.val_data[i, 1, :, :, :])).reshape(64,100)
        mag_server_data = np.abs((np_server_out[i, 0, :, :, :] + 1j*np_server_out[i, 1, :, :, :])).reshape(64,100)
        mse.append(mean_squared_error(mag_class_data, mag_server_data))
    print(np.mean(mse))
    
    df.loc['server_soft_select_loss', 'data1'] = np.mean(mse)
    
    mse = []
    print('mean MSE loss of server_right_out')
    for i in range(len(np_server_right_out)):
        mag_class_data = np.abs((clas.val_data[i, 0, :, :, :] + 1j*clas.val_data[i, 1, :, :, :])).reshape(64,100)
        mag_server_data = np.abs((np_server_right_out[i, 0, :, :, :] + 1j*np_server_right_out[i, 1, :, :, :])).reshape(64,100)
        mse.append(mean_squared_error(mag_class_data, mag_server_data))
    print(np.mean(mse))
    
    df.loc['server_norm_select_loss', 'data1'] = np.mean(mse)
    df.loc['server_soft_select_count', 'data1'] = str(choice)
    df.loc['server_norm_select_count', 'data1'] = str(right_coice)
    
'soft'
acc = [0,0]
for a, b in zip(arr_right[0:2000], arr_choice[0:2000]):
    if a==b:
        acc[0] += 1
        

'dictionary to track class model choice/selection'
choice = {0:0,1:0,2:0}
right_coice = {0:0,1:0,2:0}

'list to store class model choice/selection'
arr_choice = []
arr_right = []

if cnn:
    'define server output array'
    server_out = []
    server_out_right = []
    for data in clas.val_data:
        'create complex number'
        compl = data[0, :, :, :] + 1j*data[1, :, :, :]
        
        'calculate L2 norm of a complex number'
        'this is done in order to determine which class model will be perfect for this type of output'
        compl = compl.reshape(64,100)
        norm = np.floor(np.linalg.norm(compl))
        
        'track the amount of type each class model is choosen'
        # r = np.where(norms == norm)[0][0]
        # right_coice[r] += 1
        'keep the model selection in arr_right in order to compare with arr_choice'
        # arr_right.append(r)
        
    
        'define list to store outputs from your classes'
        out = []
        
        'define list to store softmax output from your classes'
        # soft = []
        
        'get sigmoid and softmax outputs'
        for clas in main_server.classes:
            pred = clas.model.predict(data.reshape(1, 2, 64, 100, 1)).reshape(2,64,100,1)
            tmp_compl = (pred[0,:,:, :] + 1j*pred[1,:,:, :]).reshape(64,100)
            out.append(pred)
            # soft.append(softmax(tmp_compl))
            
        'determine argmax values from sigmoid or softmax outputs'
        args = []
        # for o in soft:
        for o in out:
            tmp = o.flatten()
            args.append(tmp[np.argmax(tmp)])
            
        'determine which class model to use'
        clas_m = np.argmax(args)
        
        'add 1 to a class model that was choosen'
        choice[clas_m] += 1
        
        'append choice made in a list'
        arr_choice.append(clas_m)
        
        'append output of choosen class into a server_out'
        server_out.append(out[clas_m].reshape(2, 64, 100, 1))
        
        'append a right model to a server_out_right'
        # server_out_right.append(out[r].reshape(2, 64, 100, 1))
    
    np_server_out = np.array(server_out)
    # np_server_right_out = np.array(server_out_right)
    
    mse = []
    print('mean MSE loss of server_out')
    for i in range(len(np_server_out)):
        mag_class_data = np.abs((clas.val_data[i, 0, :, :, :] + 1j*clas.val_data[i, 1, :, :, :])).reshape(64,100)
        mag_server_data = np.abs((np_server_out[i, 0, :, :, :] + 1j*np_server_out[i, 1, :, :, :])).reshape(64,100)
        mse.append(mean_squared_error(mag_class_data, mag_server_data))
    print(np.mean(mse))
    df.loc['server_tanh_select_loss', 'data1'] = np.mean(mse)
    
    mse = []
    print('mean MSE loss of server_right_out')
    for i in range(len(np_server_right_out)):
        mag_class_data = np.abs((clas.val_data[i, 0, :, :, :] + 1j*clas.val_data[i, 1, :, :, :])).reshape(64,100)
        mag_server_data = np.abs((np_server_right_out[i, 0, :, :, :] + 1j*np_server_right_out[i, 1, :, :, :])).reshape(64,100)
        mse.append(mean_squared_error(mag_class_data, mag_server_data))
    print(np.mean(mse))
    
    # df.loc['server_norm_select_loss', 'data1'] = np.mean(mse)
    df.loc['server_tanh_select_count', 'data1'] = str(choice)
    # df.loc['server_norm_select_count', 'data1'] = str(right_coice)
    
# 'tanh'  
# for a, b in zip(arr_right[0:2000], arr_choice[0:2000]):
#     if a==b:
#         acc[1] += 1

df.to_csv('tmp.csv')  













