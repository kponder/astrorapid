from astropy.table import Table
from astrorapid.prepare_training_set import PrepareTrainingSetArrays
from astrorapid.get_training_data import get_data_from_plasticc
from astrorapid.transformer_model import train_model, train_classifier, train_transformer, train_EncoderFFNmodel, Encoder, classify_ffn
from astrorapid.plot_metrics import plot_metrics
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf

data_path = '/Users/kap146/Projects/desc/plasticc_train/'
fig_dir = '/Users/kap146/Projects/desc/plasticc_train/'
passbands=('u', 'g', 'r', 'i', 'z', 'Y')
num_ex_vs_time=100 
init_day_since_trigger=-25

train = PrepareTrainingSetArrays(
        passbands=passbands,
        contextual_info=(), #('redshift',), # sets known_redshift to True. 
        nobs=50, ## this is my lc length
        mintime=-30, maxtime=120, ## this is based on trigger time
        timestep=3.0,
        reread_data=False, 
        bcut=True,  # glactic latitude cut
        zcut=None, # redshift cut
        ignore_classes=(88,92,65,16,53,6), 
        class_name_map=None,
        nchunks=10000, 
        training_set_dir=data_path, 
        data_dir=data_path,
        save_dir=data_path, 
        get_data_func=get_data_from_plasticc, #None, 
        augment_data=False, #False, 
        redo_processing=True,
        PLAsTiCC=True, calculate_t0=True, #False,
        spline_interp=True, #False,
        single_class=False, #False, # specific thing for preprocessing
        augment_kind=None, #'pelican',
        )

otherchange='minmax_yspline_yt0_multiclass_maybefull'
class_nums = (90, 42,62, 67, 52, 64, 95, 15)
nprocesses = 1
train_size = 0.5

X_train, Xerr_train, X_test, Xerr_test, y_train, y_test, labels_train, labels_test, class_names, class_weights, sample_weights, \
    timesX_train, timesX_test, orig_lc_train, orig_lc_test, objids_train, objids_test, lc_norm_train, lc_norm_test = \
        train.prepare_training_set_arrays(otherchange, class_nums, nprocesses, train_size, normalize=True,)

## temporary weight map
X_wgtmap_train = np.zeros(np.shape(Xerr_train))
X_wgtmap_validation = np.zeros(np.shape(Xerr_test))

X_wgtmap_train[np.where(Xerr_train != 0)] = 1.0/Xerr_train[np.where(Xerr_train != 0)]**2 #np.ones(X_train.shape)
X_wgtmap_validation[np.where(Xerr_test != 0)] = 1.0/Xerr_test[np.where(Xerr_test != 0)]**2 #np.ones(X_test.shape)


# print(np.shape(X_train), np.shape(X_test), np.shape(y_train), np.shape(y_test), np.shape(X_wgtmap_train), np.shape(X_wgtmap_validation))
# print(X_train[0])
# print(labels_train[0])
# #print(orig_lc_train[0])
# #plt.plot(timesX_train[0], X_train[0], 'o')
# #print(orig_lc_test[1]['flux'])
# #print(objids_test)
# wh_train, = np.where(objids_train == 730)
# wh_test, = np.where(objids_test== 730)
# if len(wh_train) > 0:
#     wh = wh_train
#     X2 = X_train
# else:
#     wh = wh_test
#     X2 = X_test
# print('wh: ', wh)
# print('X2: \n\n\n\n\n\n', np.shape(X2[wh]), X2[wh])
# for i, pb in enumerate(passbands):
#     plt.title('730 read in')
#     plt.plot(timesX_test[0], X2[wh, :, i][0], 'o', label=pb)
#     #wh, = np.where(origx_lc_train[0]['passband']==pb) # & (orig_lc_train[0]['photflag'] > 0))
#     #plt.plot(X[0]['time'][wh], orig_lc_train[0]['flux'][wh], 'o', label=pb)
# plt.legend()
# #plt.xlim(-50, 100)
# plt.show()
# Train the neural network model on saved files
# modelt = train_transformer(X_train, X_test, 
#                           X_wgtmap_train, X_wgtmap_validation,
#                           X_train, X_test, ## originall y_train/test but this is the Autoendocer
#                           #sample_weights=sample_weights, # not using for AE
#                           fig_dir=fig_dir,
#                           retrain=True, #retrain_network, 
#                           epochs=2,
#                           num_layers=8, d_model=128, dff=64, num_heads=8, embed=True,
#                           step_size=0.00001,
#                           ae_loss='KLD_RMSE', # spline LCs so no need for KLD
#                           plot_loss=True, 
#                           dropout_rate=0.0,
#                           batch_size=64)

# tf.keras.backend.clear_session()

# aeclass = train_classifier(X_train, X_wgtmap_train, lc_norm_train,
#                     X_test, X_wgtmap_validation, lc_norm_test,
#                     y_train, y_test,
#                     transformer_weights=os.path.join(fig_dir, 'transformer_model.hdf5'),
#                     sample_weights=sample_weights,
#                     #num_class=len(class_nums),
#                     fig_dir=fig_dir, retrain=True,
#                     epochs=3, 
#                     dropout_rate=0.0, 
#                     batch_size=64,
#                     num_layers=8, d_model=128, dff=64, num_heads=8, embed=True,
#                     #step_size=classifier_step_size,
#                     #classifier_loss=classifier_loss,
#                     single_class=False,
#                     plot_loss=True)

# import tensorflow as tf
# Nf = X_train.shape[-1] # update to be based on input data
# target_size = Nf # target output size
# lc_length = X_train.shape[1] # read out of files
# num_layers=8
# d_model=128
# dff=64
# num_heads=8
# num_class = y_test.shape[-1]
# step_size=0.0001

# ## Define the FFN Classifier
# cl_inp = tf.keras.layers.Input(shape=(None, Nf), name='classifier_input')

# classify_encoder = Encoder(num_layers, d_model, num_heads, dff,
#                         lc_length, 0.0, embed=True)
# class_ffn = classify_ffn(num_class, dff, rate=0.0)

# ## The model implemented
# cl_x = classify_encoder(cl_inp)
# cl_x = class_ffn(cl_x)
# aeclass = tf.keras.models.Model(inputs=[cl_inp], outputs=cl_x)

# aeclass_weights = 'ffnclass_model.hdf5'
# aeclass.load_weights(os.path.join(fig_dir, aeclass_weights))

# optimizer = tf.keras.optimizers.Adam(step_size)
# loss_object = tf.keras.losses.CategoricalCrossentropy() 
# aeclass.compile(loss=loss_object,
#                 optimizer=optimizer, 
#                 sample_weights=sample_weights,
#                 metrics=['accuracy'],
#                 )

# # # # y_testm = np.mean(y_test, axis=1)

# # # Plot classification metrics such as confusion matrices
#plot_metrics(class_names, aeclass, X_test, y_test, fig_dir, timesX_test=timesX_test, orig_lc_test=orig_lc_test,
#                    objids_test=objids_test, passbands=passbands, num_ex_vs_time=num_ex_vs_time, init_day_since_trigger=init_day_since_trigger)

