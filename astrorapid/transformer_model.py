import tensorflow as tf
import numpy as np
import os
import pickle

# Positional Encoding
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


# MultiHeaded Attention
def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
       q, k, v must have matching leading dimensions.
       k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
       The mask has different shapes depending on its type(padding or look ahead)
       but it must be broadcastable for addition.

       Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable
              to (..., seq_len_q, seq_len_k). Defaults to None.
      Returns:
       output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
           Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        r = tf.transpose(x, perm=[0, 2, 1, 3])
        return r
  
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights

def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
                                tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
                                tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
                                ])

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
  
    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
        return out2

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)
 
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)
  
    
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output,
                                               out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2

class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 maximum_position_encoding, rate=0.1, embed=False):
        super(Encoder, self).__init__()

        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.maximum_position_encoding = maximum_position_encoding
        self.rate = rate
        self.embed = embed

        if self.embed:
            self.embedding = tf.keras.layers.Dense(self.d_model) # linear embedding

        self.pos_encoding = positional_encoding(self.maximum_position_encoding,
                                                self.d_model)

        self.enc_layers = [EncoderLayer(self.d_model, self.num_heads, self.dff, self.rate)
                           for _ in range(self.num_layers)]

        self.dropout = tf.keras.layers.Dropout(self.rate)

    def call(self, x, training): #, mask):
        mask = create_padding_mask(x[:,:, 0])

        if self.embed:
            x = self.embedding(x)

        seq_len = tf.shape(x)[1]

        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = tf.cast(x, dtype=tf.float32)
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)

    def get_config(self):
        config = super().get_config().copy()
        config.update({'num_layers': self.num_layers,
                       'd_model': self.d_model, 
                       'num_heads': self.num_heads, 
                       'dff': self.dff, 
                       'maximum_position_encoding': self.maximum_position_encoding, 
                       'rate': self.rate,
                       'embed': self.embed
                      })
        return config

class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 maximum_position_encoding, rate=0.1, embed=False):
        super(Decoder, self).__init__()

        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.maximum_position_encoding = maximum_position_encoding
        self.rate = rate
        self.embed = embed
        
        if self.embed:
            self.embedding = tf.keras.layers.Dense(self.d_model) # linear embedding

        self.pos_encoding = positional_encoding(self.maximum_position_encoding, self.d_model)

        self.dec_layers = [DecoderLayer(self.d_model, self.num_heads, self.dff, self.rate)
                           for _ in range(self.num_layers)]

        self.dropout = tf.keras.layers.Dropout(self.rate)

    def call(self, x, enc_output, training, mask): #, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}
        
        if self.embed:
            x = self.embedding(x) 
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = tf.cast(x, dtype=tf.float32)

        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                   mask[0], mask[1])
                                                   #look_ahead_mask, padding_mask)

        attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
        attention_weights['decoder_layer{}_block2'.format(i+1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x #, attention_weights

    def get_config(self):
        config = super().get_config().copy()
        config.update({'num_layers': self.num_layers,
                       'd_model': self.d_model, 
                       'num_heads': self.num_heads, 
                       'dff': self.dff, 
                       'maximum_position_encoding': self.maximum_position_encoding, 
                       'rate': self.rate,
                       'embed': self.embed
                      })
        return config

# Masking
def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0.0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)
  
    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)
  
    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
  
    return enc_padding_mask, combined_mask, dec_padding_mask

def create_decoder_masks(inp, tar):  
    inp = inp[:,:, 0]
    tar = tar[:,:, 0]
    
    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)
  
    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
  
    return combined_mask, dec_padding_mask

class RMSE(tf.keras.losses.Loss):
    def __init__(self, name="rmse"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        mse = tf.math.reduce_mean(tf.square(y_true - y_pred))
        return tf.math.sqrt(mse)

def loss_kld(layer1, layer2, alpha=0.30, kld_rho=1e-5):
    alpha = tf.constant(alpha, dtype=tf.float32)
    layer1 = layer1[0]
    layer1 = tf.math.abs(layer1)
    layer2 = layer2[0]
    layer2 = tf.math.abs(layer2)

    def loss(y_true, y_pred):
        ones = tf.ones(layer1.shape, dtype=tf.float32)
        rhoc = kld_rho
        rho = rhoc*ones

        def kld(layer):
            kld_1 = tf.math.multiply(rhoc, tf.math.log(tf.math.divide_no_nan(rho, layer)))
            kld_2 = tf.math.multiply((1.0 - rhoc), tf.math.divide_no_nan(tf.math.log(ones-rho), tf.math.log(ones-layer)))
            return tf.reduce_sum(kld_1 + kld_2) #kld_1_without_nans + kld_2_without_nans)

        mse = tf.math.reduce_mean(tf.square(y_true - y_pred))
        rmse = tf.math.sqrt(mse)
        return rmse + tf.multiply(alpha, (kld(layer1) + kld(layer2)))
    return loss


def loss_kld_noAE(layer1, alpha=0.3, kld_rho=1e-5):
    alpha = tf.constant(alpha, dtype=tf.float32)
    layer1 = layer1[0]
    layer1 = tf.math.abs(layer1)

    ones = tf.ones(layer1.shape, dtype=tf.float32)
    rhoc = kld_rho
    rho = rhoc*ones

    def kld(layer):
        kld_1 = tf.math.multiply(rhoc, tf.math.log(tf.math.divide_no_nan(rho, layer)))
        kld_2 = tf.math.multiply((1.0 - rhoc), tf.math.divide_no_nan(tf.math.log(ones-rho), tf.math.log(ones-layer)))
        return tf.reduce_sum(kld_1 + kld_2)

    return tf.multiply(alpha, kld(layer1))



def train_transformer(X_train, X_validation, 
                    X_wgtmap_train, X_wgtmap_validation,
                    y_train, y_validation, 
                    fig_dir='.', retrain=True,
                    epochs=25, dropout_rate=0.0, batch_size=64,
                    num_layers=8, d_model=6, dff=64, num_heads=6, embed=True,
                    step_size=0.00001,
                    ae_loss='KLD_RMSE', kld_alpha=0.3, kld_rho=1e-5,
                    plot_loss=True):
    """ Train Transformer AutoEncoder and save model. 
    
    Parameters
    ----------
    X_train : array
        info on this
    X_validation : array
        info
    y_train : array
        info
    y_validation : array
        info
    fig_dir : str
        Location of figures
    retrain : bool
        maybe do not need?
    epochs : int
        Number of iterations
    dropout_rate : float
        Rate of node dropout (regularization)
    batch_size : int
        Batch size. Default is 64. Prefer powers of 2
    num_layers : int
        Transformer number of layers
    d_model : int
        Transformer d_model
    dff : int
        Transformer dff
    num_heads : int
        Transformer number of heads
    step_size : float
        Step size for Adam gradient descent 
    embed : bool
        Expands the input from 6 filters to d_model dimensional inputs through keras.layers.Dense(d_model)
    ae_loss : str
        Choose a loss function from dictionary.
        Default is RMSE with KLD penalty (KLD_RMSE)
        Options are MSE, MSLE, Huber, MAE, LCE, RMSE, KLD_RMSE
    kld_alpha : float
        KLD_RMSE loss. The penalty factor of KLD weight (higher is more regularization)
    kld_rho : float
        KLD_RMSE loss. The number to compare all weights to. Closer to zero is lower weights. 
    classifier_loss : undecided
        Loss function for feed-forward network doing final classification.
    plot_loss : bool
        Plot the loss function (this would need to be 2 different loss functions for transformer)
    """

    Nf = X_train.shape[-1] # update to be based on input data
    target_size = Nf # target output size
    lc_length = X_train.shape[1] # read out of files

    # Determine the mask map
    pre_mask_map = np.ma.masked_values(X_train, 0)
    mask_map = np.ones(np.shape(X_train))
    mask_map[pre_mask_map.mask] = 0.0

    pre_mask_map_validation = np.ma.masked_values(X_validation, 0)
    mask_map_validation = np.ones(np.shape(X_validation))
    mask_map_validation[pre_mask_map_validation.mask] = 0.0

    # my function needs a dataset object to run through the generator. 
    ## weight map needs to be 1/variance
    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train, mask_map, X_wgtmap_train))
    batch_ds = dataset.batch(batch_size)

    dataset_validation = tf.data.Dataset.from_tensor_slices((X_validation, y_validation, mask_map_validation, X_wgtmap_validation))
    batch_ds_validation = dataset_validation.batch(batch_size)

    if embed:
        assert d_model > target_size, f'If embedding, d_model > {Nf}'

    # Define the model
    encoder = Encoder(num_layers, d_model, num_heads, dff,
                        lc_length, dropout_rate, embed)

    decoder = Decoder(num_layers, d_model, num_heads, dff,
                        lc_length, dropout_rate, embed)

    final_layer = tf.keras.layers.Dense(target_size)

    if embed:
        inp = tf.keras.layers.Input(shape=(None,Nf))
        target = tf.keras.layers.Input(shape=(None,Nf))
        mask_map = tf.keras.layers.Input(shape=(None,Nf))
    else:
        inp = tf.keras.layers.Input(shape=(None,None))
        target = tf.keras.layers.Input(shape=(None,None))
        mask_map = tf.keras.layers.Input(shape=(None,None))

    x = encoder(inp)
    x = decoder(target, x, mask=create_decoder_masks(inp, target))
    x = final_layer(x)
    x = tf.keras.layers.Multiply()([x, mask_map])

    transformer = tf.keras.models.Model(inputs=[inp, target, mask_map], outputs=x)


    loss_dict = {'MSE': tf.keras.losses.MeanSquaredError(),
             'MSLE': tf.keras.losses.MeanSquaredLogarithmicError(),
             'Huber': tf.keras.losses.Huber(),
             'MAE': tf.keras.losses.MeanAbsoluteError(),
             'LCE': tf.keras.losses.LogCosh(),
             'RMSE': RMSE(),
             'KLD_RMSE':loss_kld(transformer.get_layer(name='encoder').get_weights(),
                                 transformer.get_layer(name='decoder').get_weights(),
                                 alpha=kld_alpha, kld_rho=kld_rho),
             }
    optimizer = tf.keras.optimizers.Adam(step_size)
    loss_object = loss_dict[ae_loss]


    # Compile and run the Transformer model
    transformer.compile(optimizer=optimizer, loss=loss_object)

    num_batches = 0
    for (batch, _) in enumerate(batch_ds):
        num_batches = batch
    
    val_batches = 0
    for (batch, _) in enumerate(batch_ds_validation):
        val_batches = batch

    def generator(data_set):
        while True:
            for in_batch, tar_batch, mask_batch, wgt_batch in data_set:
                yield ( [in_batch , tar_batch[:, :-1, :],  mask_batch[:, 1:, :], wgt_batch] , tar_batch[:, 1:, :])


    history = transformer.fit_generator(generator(batch_ds),
                        validation_data = generator(batch_ds_validation),
                        epochs=epochs,
                        steps_per_epoch=num_batches,
                        validation_steps=val_batches,
                        verbose=0,
                        )

    print(transformer.summary())
    transformer_filename = os.path.join(fig_dir, "transformer_model.hdf5")
    transformer.save(transformer_filename)

    with open(os.path.join(fig_dir, "transformer_history.pickle"), 'wb') as fp:
        pickle.dump(history.history, fp)

    if plot_loss:
        plot_history(history.history, fig_dir, transformer=True)

    return transformer


## Neural Network for final classifier
def classify_ffn(nclass, dff, rate=0.0, single_class=False):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(dff, activation='relu')) 
    model.add(tf.keras.layers.Dropout(rate))
    model.add(tf.keras.layers.Dense(dff, activation='relu'))
    if single_class:
        model.add(tf.keras.layers.GlobalMaxPool1D())
    model.add(tf.keras.layers.Dense(nclass, activation='softmax'))
    return model

def train_classifier(X_train, X_wgtmap_train, X_norm_train,
                    X_validation, X_wgtmap_validation, X_norm_validation,
                    y_train, y_validation,
                    transformer_weights='transformer_model.hdf5',
                    sample_weights=None,
                    fig_dir='.', retrain=True,
                    epochs=25, dropout_rate=0.0, batch_size=64,
                    num_layers=8, d_model=6, dff=64, num_heads=6, embed=True,
                    step_size=0.0001,
                    classifier_loss='categorical_crossentropy',
                    single_class=True,
                    plot_loss=True):
    """ Train FFN classifier and save model. 
    
    Parameters
    ----------
    X_train : array
        info on this
    X_validation : array
        info
    y_train : array
        info
    y_validation : array
        info
    transformer_model : str
        Name and location of transformer model to read weights from.
    sample_weights : array or None
        info - for classificaiton step
    fig_dir : str
        Location of figures
    retrain : bool
        maybe do not need?
    epochs : int
        Number of iterations
    dropout_rate : float
        Rate of node dropout (regularization)
    batch_size : int
        Batch size. Default is 64. Prefer powers of 2
    num_layers : int
        Transformer number of layers
    d_model : int
        Transformer d_model
    dff : int
        Transformer dff
    num_heads : int
        Transformer number of heads
    embed : bool 
        info
    step_size : float
        Step size for Adam gradient descent 
    classifier_loss : undecided
        Loss function for feed-forward network doing final classification.
    plot_loss : bool
        Plot the loss function
    """
    ## TBD: get data in right format and all the other inputs
    Nf = X_train.shape[-1] # update to be based on input data
    target_size = Nf # target output size
    lc_length = X_train.shape[1] # read out of files

    num_class = y_validation.shape[-1]

    #y_train = np.mean(y_train, axis=1)
    #y_validation = np.mean(y_validation, axis=1)

    X_norm_train = np.reshape([l*lc_length for l in X_norm_train], (len(X_train), lc_length, 2))
    X_norm_validation = np.reshape([l*lc_length for l in X_norm_validation], (len(X_validation), lc_length, 2))

    ## weight map needs to be 1/variance
    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train, X_wgtmap_train, X_norm_train))
    batch_ds = dataset.batch(batch_size)

    dataset_validation = tf.data.Dataset.from_tensor_slices((X_validation, y_validation, X_wgtmap_validation, X_norm_validation))
    batch_ds_validation = dataset_validation.batch(batch_size)

    ## Define the tranformer autoencoder
    # These steps are neccesary since it is a custom model
    # as opposued to only using load_model (missing correct config)
    encoder = Encoder(num_layers, d_model, num_heads, dff,
                       lc_length, dropout_rate, embed=True)
    decoder = Decoder(num_layers, d_model, num_heads, dff,
                        lc_length, dropout_rate, embed=True)
    final_layer = tf.keras.layers.Dense(target_size)

    if embed:
        inp = tf.keras.layers.Input(shape=(None, Nf))
        target = tf.keras.layers.Input(shape=(None, Nf))
        mask = tf.keras.layers.Input(shape=(None, Nf))
    else:
        inp = tf.keras.layers.Input(shape=(None, None))
        target = tf.keras.layers.Input(shape=(None, None))
        mask = tf.keras.layers.Input(shape=(None, None))

    x = encoder(inp)
    x = decoder(target, x, mask=create_decoder_masks(inp, target))
    x = final_layer(x)
    x = tf.keras.layers.Multiply()([x, mask])
    transformer = tf.keras.models.Model(inputs=[inp, target, mask], outputs=x)

    transformer.load_weights(os.path.join(fig_dir, transformer_weights))
    
    ## Define the FFN Classifier
    if embed:
        cl_inp = tf.keras.layers.Input(shape=(None, Nf), name='classifier_input')
    else:
        cl_inp = tf.keras.layers.Input(shape=(None, None))
    norm = tf.keras.layers.Input(shape=(None, 2), dtype=tf.float32)

    classify_encoder = Encoder(num_layers, d_model, num_heads, dff,
                               lc_length, dropout_rate, embed=embed)
    classify_encoder(cl_inp)
    classify_encoder.set_weights(transformer.get_layer(name='encoder').get_weights())

    classify_encoder.trainable = False

    class_ffn = classify_ffn(num_class, dff, rate=dropout_rate, single_class=single_class)

    ## The model implemented
    cl_x = classify_encoder(cl_inp)
    # This layer concatenates the norm information.
    cl_x = tf.keras.layers.Concatenate(axis=-1)([cl_x, norm])
    # This is the final FFN
    cl_x = class_ffn(cl_x)
    aeclass = tf.keras.models.Model(inputs=[cl_inp, norm], outputs=cl_x)

    # Optimizer and compile
    optimizer = tf.keras.optimizers.Adam(step_size)
    loss_object = tf.keras.losses.CategoricalCrossentropy()

    aeclass.compile(loss=loss_object,
                    optimizer=optimizer, 
                    sample_weights=sample_weights,
                    metrics=['accuracy'],
                    )
    num_batches = 0
    for (batch, _) in enumerate(batch_ds):
        num_batches = batch
    
    val_batches = 0
    for (batch, _) in enumerate(batch_ds_validation):
        val_batches = batch

    def generator(data_set):
        while True:
            for in_batch, tar_batch, wgt_batch, norm_batch in data_set:
                yield ( [in_batch, norm_batch, wgt_batch] , tar_batch)

    history = aeclass.fit_generator(generator(batch_ds),
                          validation_data = generator(batch_ds_validation),
                          epochs=epochs,
                          steps_per_epoch = num_batches,
                          validation_steps=val_batches,
                         )
    
    print(aeclass.summary())
    aeclass_filename = os.path.join(fig_dir, "ffnclass_model.hdf5")
    aeclass.save(aeclass_filename)

    with open(os.path.join(fig_dir, "ffnclass_history.pickle"), 'wb') as fp:
        pickle.dump(history.history, fp)

    if plot_loss:
        plot_history(history.history, fig_dir, transformer=False)

    return aeclass

def train_model(X_train, X_wgtmap_train, X_norm_train,
                X_validation, X_wgtmap_validation, X_norm_validation,
                y_train, y_validation, 
                label_train, label_validation,
                fig_dir='.', retrain=True,
                epochs=25, dropout_rate=0.0, batch_size=64,
                num_layers=8, d_model=6, dff=64, num_heads=6, embed=True,
                transformer_step_size=0.00001,
                transformer_loss='KLD_RMSE', kld_alpha=0.3, kld_rho=1e-5,
                classifier_loss='categorical_crossentropy', 
                sample_weights=None, #num_class=4,
                classifier_step_size=0.0001,
                single_class=True,
                plot_loss=True):

    train_transformer(X_train, X_wgtmap_train, 
                                    X_validation, X_wgtmap_validation,
                                    y_train, y_validation,
                                    fig_dir=fig_dir, retrain=retrain,
                                    epochs=epochs, dropout_rate=dropout_rate, batch_size=batch_size,
                                    num_layers=num_layers, d_model=d_model, dff=dff, num_heads=num_heads, embed=embed,
                                    step_size=transformer_step_size,
                                    ae_loss=transformer_loss, kld_alpha=kld_alpha, kld_rho=kld_rho,
                                    plot_loss=plot_loss)
    tf.keras.backend.clear_session()
    ffn_classifier = train_classifier(X_train, X_wgtmap_train, X_norm_train,
                                      X_validation, X_wgtmap_validation, X_norm_validation,
                                      y_train, y_validation,
                                      label_train, label_validation,
                                      transformer_weights=os.path.join(fig_dir, 'transformer_model.hdf5'),
                                      sample_weights=sample_weights, #num_class=num_class,
                                      fig_dir=fig_dir, retrain=retrain,
                                      epochs=epochs, dropout_rate=dropout_rate, batch_size=batch_size,
                                      num_layers=num_layers, d_model=d_model, dff=dff, num_heads=num_heads, embed=embed,
                                      step_size=classifier_step_size,
                                      classifier_loss=classifier_loss,
                                      single_class=single_class,
                                      plot_loss=plot_loss)
    return ffn_classifier

def train_EncoderFFNmodel(X_train, X_wgtmap_train, X_norm_train,
                          X_validation, X_wgtmap_validation, X_norm_validation,
                        y_train, y_validation, 
                        fig_dir='.', retrain=True,
                        epochs=25, dropout_rate=0.0, batch_size=64,
                        sample_weights=None,
                        num_layers=8, d_model=6, dff=64, num_heads=6, embed=True,
                        sparse=True, kld_alpha=0.3, kld_rho=1e-5,
                        classifier_loss=tf.keras.losses.CategoricalCrossentropy(),
                        step_size=0.00001,
                        single_class=True,
                        plot_loss=True):

    Nf = X_train.shape[-1] # update to be based on input data
    #target_size = Nf # target output size
    lc_length = X_train.shape[1] # read out of files

    num_class = y_train.shape[-1]

    X_norm_train = np.reshape([l*lc_length for l in X_norm_train], (len(X_train), lc_length, 2))
    X_norm_validation = np.reshape([l*lc_length for l in X_norm_validation], (len(X_validation), lc_length, 2))

    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train, X_wgtmap_train, X_norm_train))
    batch_ds = dataset.batch(batch_size)

    dataset_validation = tf.data.Dataset.from_tensor_slices((X_validation, y_validation, X_wgtmap_validation, X_norm_validation))
    batch_ds_validation = dataset_validation.batch(batch_size)

    encoder = Encoder(num_layers, d_model, num_heads, dff,
                        lc_length, dropout_rate, embed=embed)

    class_ffn = classify_ffn(num_class, dff, rate=dropout_rate, single_class=single_class)

    if embed:
        inp = tf.keras.layers.Input(shape=(None, Nf))
    else:
        inp = tf.keras.layers.Input(shape=(None, None))
    norm = tf.keras.layers.Input(shape=(None, 2), dtype=tf.float32)

    x = encoder(inp)
    x = tf.keras.layers.Concatenate(axis=-1)([x, norm])
    x = class_ffn(x)

    model = tf.keras.models.Model(inputs=[inp, norm], outputs=x)

    if sparse:
        # Add KLD penalty for sparse light curves
        model.add_loss(lambda x=model.get_layer(name='encoder').get_weights(): loss_kld_noAE(x, kld_alpha, kld_rho))

    optimizer = tf.keras.optimizers.Adam(step_size)
    loss_object = classifier_loss 

    model.compile(loss=loss_object, 
                optimizer=optimizer, 
                metrics=['accuracy'])

    num_batches = 0
    for (batch, _) in enumerate(batch_ds):
        num_batches = batch
    
    val_batches = 0
    for (batch, _) in enumerate(batch_ds_validation):
        val_batches = batch

    def generator(data_set):
        while True:
            for in_batch, tar_batch, wgt_batch, norm_batch in data_set:
                yield ( [in_batch, norm_batch, wgt_batch] , tar_batch)

    history = model.fit_generator(generator(batch_ds),
                    validation_data = generator(batch_ds_validation),
                    epochs=epochs,
                    steps_per_epoch = num_batches,
                    validation_steps = val_batches,
                    )
    print(model.summary())
    enc_ffnclass_filename = os.path.join(fig_dir, "Encoder_ffnclass_model.hdf5")
    model.save(enc_ffnclass_filename)

    with open(os.path.join(fig_dir, "encoder_ffn_history.pickle"), 'wb') as fp:
        pickle.dump(history.history, fp)

    if plot_loss:
        plot_history(history.history, fig_dir, transformer=False)

    return model


def plot_history(history, fig_dir, transformer=False):
    import matplotlib.pyplot as plt
    # Plot loss vs epochs
    plt.figure(figsize=(12,10))
    train_loss = history['loss']
    val_loss = history['val_loss']
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.title(os.path.basename(fig_dir).replace('_', '-'))
    if transformer:
        plt.savefig(os.path.join(fig_dir, "transformer_loss_history.pdf"))
    else:
        plt.savefig(os.path.join(fig_dir, "classifier_loss_history.pdf"))

    if not transformer:
        # Plot accuracy vs figure
        plt.figure(figsize=(12,10))
        if 'accuracy' in history:
            train_acc = history['accuracy']
            val_acc = history['val_accuracy']
        else:
            train_acc = history['acc']
            val_acc = history['val_acc']
        plt.plot(train_acc)
        plt.plot(val_acc)
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.title(os.path.basename(fig_dir).replace('_', '-'))
        plt.savefig(os.path.join(fig_dir, "classifier_accuracy_history.pdf"))