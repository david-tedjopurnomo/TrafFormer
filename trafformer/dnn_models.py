"""This module contains implementations of all models used in our work"""

from tensorflow import keras
from tensorflow.keras import layers 

class TrafFormerSpeed():
    """
    Implementation of our transformer model for the traffic prediction. This
    variation uses only traffic speed data.
    """
    def build_model(self,
                    in_out_shape,
                    head_size,
                    num_heads,
                    ff_dim,
                    num_transformer_blocks,
                    mlp_units,
                    dropout,
                    mlp_dropout):
        inputs = keras.Input(shape=in_out_shape, dtype="float32")
        x = inputs
        h_traf = layers.Lambda(lambda x:x[:,:,:,0])(x)
        d_traf = layers.Lambda(lambda x:x[:,:,:,1])(x)
        h_traf = layers.Masking()(h_traf)
        d_traf = layers.Masking()(d_traf)
        h_traf = layers.LayerNormalization(epsilon=1e-6)(h_traf) 
        d_traf = layers.LayerNormalization(epsilon=1e-6)(d_traf)
        traf_shape = h_traf.shape.as_list()[1:]
        traf_shape.append(1)
        traf_shape = tuple(traf_shape)
        h_traf = layers.Reshape(traf_shape)(h_traf)
        d_traf = layers.Reshape(traf_shape)(d_traf)
        x = layers.Concatenate(axis=-1)([h_traf, d_traf])
        
        for _ in range(num_transformer_blocks):
            x = self.transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

        for dim in mlp_units:
            x = layers.Dense(dim, activation="relu")(x) 
            x = layers.Dropout(mlp_dropout)(x)
        outputs = layers.Dense(1, activation="relu")(x)
        return keras.Model(inputs, outputs)
        
    
    def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):
        # Normalization and Attention
        x = layers.LayerNormalization(epsilon=1e-6)(inputs)
        x = layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(x, x)
        x = layers.Dropout(dropout)(x)
        res = x + inputs

        # Feed Forward Part
        x = layers.LayerNormalization(epsilon=1e-6)(res)
        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        return x + res
        
        
class TrafFormerFull():
    """
    Implementation of our transformer model for the traffic prediction. 
    """
    def build_model(self,
                    in_out_shape,
                    head_size,
                    embed_size,
                    num_heads,
                    ff_dim,
                    num_transformer_blocks,
                    mlp_units,
                    dropout,
                    mlp_dropout):
        inputs = keras.Input(shape=in_out_shape, dtype="float32")
        x = inputs
        
        # Slice the inputs
        # Past hour and past day traffic reading
        h_traf = layers.Lambda(lambda x:x[:,:,:,0])(x)
        d_traf = layers.Lambda(lambda x:x[:,:,:,3])(x)
        h_traf = layers.Masking()(h_traf)
        d_traf = layers.Masking()(d_traf)
        h_traf = layers.LayerNormalization(epsilon=1e-6)(h_traf) 
        d_traf = layers.LayerNormalization(epsilon=1e-6)(d_traf)
        traf_shape = h_traf.shape.as_list()[1:]
        traf_shape.append(1)
        traf_shape = tuple(traf_shape)
        h_traf = layers.Reshape(traf_shape)(h_traf)
        d_traf = layers.Reshape(traf_shape)(d_traf)
        
        
        # Past hour and past day timeslot embedding
        h_time = layers.Lambda(lambda x:x[:,:,:,1])(x)
        d_time = layers.Lambda(lambda x:x[:,:,:,4])(x)
        
        # Past hour and past day day-of-week embedding
        h_day = layers.Lambda(lambda x:x[:,:,:,2])(x)
        d_day = layers.Lambda(lambda x:x[:,:,:,5])(x)
        
        # Create embedding layer for both timeslot and day-of-week embedding
        """
        print("USING LARGER TIME EMBEDDING BLOCKS")
        time_embed = layers.Embedding(96, embed_size)
        day_embed = layers.Embedding(7, embed_size)
        
        # Pass to embedding layers and then merge
        h_time = layers.Lambda(lambda x: x/3.0)(h_time)
        d_time = layers.Lambda(lambda x: x/3.0)(d_time)
        h_time = time_embed(h_time)
        d_time = time_embed(d_time)
        """
        time_embed = layers.Embedding(288, embed_size)
        day_embed = layers.Embedding(7, embed_size)
        
        # Pass to embedding layers and then merge
        h_time = time_embed(h_time)
        d_time = time_embed(d_time)
        h_day = day_embed(h_day)
        d_day = day_embed(d_day)
        
        all_layers = [h_traf, d_traf, h_time, d_time, h_day, d_day]
        x = layers.Concatenate(axis=-1)(all_layers)
        res = x
        
        for _ in range(num_transformer_blocks):
            x = self.transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
        x = x + res

        for dim in mlp_units:
            x = layers.Dense(dim, activation="relu")(x) 
            x = layers.Dropout(mlp_dropout)(x)
        outputs = layers.Dense(1, activation="relu")(x)
        return keras.Model(inputs, outputs)
        
    def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):
        # Normalization and Attention
        x = layers.LayerNormalization(epsilon=1e-6)(inputs)
        x = layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(x, x)
        x = layers.Dropout(dropout)(x)
        res = x + inputs

        # Feed Forward Part
        x = layers.LayerNormalization(epsilon=1e-6)(res)
        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        return x + res
        
        
class TrafFormerSingle():
    """
    Implementation of our transformer model for the traffic prediction. This
    version takes either the day or hour data, not both.
    """
    def build_model(self,
                    in_out_shape,
                    head_size,
                    embed_size,
                    num_heads,
                    ff_dim,
                    num_transformer_blocks,
                    mlp_units,
                    dropout,
                    mlp_dropout):
        inputs = keras.Input(shape=in_out_shape, dtype="float32")
        x = inputs
        
        # Slice the inputs
        # Past hour and past day traffic reading
        traf = layers.Lambda(lambda x:x[:,:,:,0])(x)
        traf = layers.Masking()(traf)
        traf = layers.LayerNormalization(epsilon=1e-6)(traf) 
        traf_shape = traf.shape.as_list()[1:]
        traf_shape.append(1)
        traf_shape = tuple(traf_shape)
        traf = layers.Reshape(traf_shape)(traf)
        
        
        # Past hour and past day timeslot embedding
        time = layers.Lambda(lambda x:x[:,:,:,1])(x)
        day = layers.Lambda(lambda x:x[:,:,:,2])(x)
        
        time_embed = layers.Embedding(288, embed_size)
        day_embed = layers.Embedding(7, embed_size)
        
        # Pass to embedding layers and then merge
        time = time_embed(time)
        day = day_embed(day)
        
        all_layers = [traf, time, day]
        x = layers.Concatenate(axis=-1)(all_layers)
        res = x
        
        for _ in range(num_transformer_blocks):
            x = self.transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
        x = x + res

        for dim in mlp_units:
            x = layers.Dense(dim, activation="relu")(x) 
            x = layers.Dropout(mlp_dropout)(x)
        outputs = layers.Dense(1, activation="relu")(x)
        return keras.Model(inputs, outputs)
        
    
    def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):
        # Normalization and Attention
        x = layers.LayerNormalization(epsilon=1e-6)(inputs)
        x = layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(x, x)
        x = layers.Dropout(dropout)(x)
        res = x + inputs

        # Feed Forward Part
        x = layers.LayerNormalization(epsilon=1e-6)(res)
        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        return x + res
        
        
class TrafFormerCyc():
    """
    Implementation of our transformer model for the traffic prediction that
    uses cyclical features for the daty and time embedding.
    """
    def build_model(self,
                    in_out_shape,
                    head_size,
                    num_heads,
                    ff_dim,
                    num_transformer_blocks,
                    mlp_units,
                    dropout,
                    mlp_dropout):
        inputs = keras.Input(shape=in_out_shape, dtype="float32")
        x = inputs
        
        # Slice the inputs
        # Past hour and past day traffic reading
        h_traf = layers.Lambda(lambda x:x[:,:,:,0])(x)
        d_traf = layers.Lambda(lambda x:x[:,:,:,5])(x)
        h_traf = layers.Masking()(h_traf)
        d_traf = layers.Masking()(d_traf)
        h_traf = layers.LayerNormalization(epsilon=1e-6)(h_traf) 
        d_traf = layers.LayerNormalization(epsilon=1e-6)(d_traf)
        traf_shape = h_traf.shape.as_list()[1:]
        traf_shape.append(1)
        traf_shape = tuple(traf_shape)
        h_traf = layers.Reshape(traf_shape)(h_traf)
        d_traf = layers.Reshape(traf_shape)(d_traf)
        
        # Other data
        i1 = layers.Lambda(lambda x:x[:,:,:,1])(x)
        i2 = layers.Lambda(lambda x:x[:,:,:,2])(x)
        i3 = layers.Lambda(lambda x:x[:,:,:,3])(x)
        i4 = layers.Lambda(lambda x:x[:,:,:,4])(x)
        i5 = layers.Lambda(lambda x:x[:,:,:,6])(x)
        i6 = layers.Lambda(lambda x:x[:,:,:,7])(x)
        i7 = layers.Lambda(lambda x:x[:,:,:,8])(x)
        i8 = layers.Lambda(lambda x:x[:,:,:,9])(x)
        
        i1 = layers.Reshape(traf_shape)(i1)
        i2 = layers.Reshape(traf_shape)(i2)
        i3 = layers.Reshape(traf_shape)(i3)
        i4 = layers.Reshape(traf_shape)(i4)
        i5 = layers.Reshape(traf_shape)(i5)
        i6 = layers.Reshape(traf_shape)(i6)
        i7 = layers.Reshape(traf_shape)(i7)
        i8 = layers.Reshape(traf_shape)(i8)
        
        all_layers = [h_traf, i1, i2, i3, i4, d_traf, i5, i6, i7, i8]
        x = layers.Concatenate(axis=-1)(all_layers)
        
        for _ in range(num_transformer_blocks):
            x = self.transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

        for dim in mlp_units:
            x = layers.Dense(dim, activation="relu")(x) 
            x = layers.Dropout(mlp_dropout)(x)
        outputs = layers.Dense(1, activation="relu")(x)
        return keras.Model(inputs, outputs)
        
    
    def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):
        # Normalization and Attention
        x = layers.LayerNormalization(epsilon=1e-6)(inputs)
        x = layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(x, x)
        x = layers.Dropout(dropout)(x)
        res = x + inputs

        # Feed Forward Part
        x = layers.LayerNormalization(epsilon=1e-6)(res)
        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
        #x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        #x = layers.Dense(inputs.shape[-1], activation="relu")(x)
        return x + res
        
        
class FeedforwardNN():
    """
    Basic feedforward neural network 
    """
    def build_model(self, in_out_shape, embed_size, mlp_units):
        inputs = keras.Input(shape=in_out_shape, dtype="float32")
        x = inputs
        
        # Slice the inputs
        # Past hour and past day traffic reading
        h_traf = layers.Lambda(lambda x:x[:,:,:,0])(x)
        d_traf = layers.Lambda(lambda x:x[:,:,:,3])(x)
        h_traf = layers.Masking()(h_traf)
        d_traf = layers.Masking()(d_traf)
        h_traf = layers.LayerNormalization(epsilon=1e-6)(h_traf) 
        d_traf = layers.LayerNormalization(epsilon=1e-6)(d_traf)
        traf_shape = h_traf.shape.as_list()[1:]
        traf_shape.append(1)
        traf_shape = tuple(traf_shape)
        h_traf = layers.Reshape(traf_shape)(h_traf)
        d_traf = layers.Reshape(traf_shape)(d_traf)
        
        
        # Past hour and past day timeslot embedding
        h_time = layers.Lambda(lambda x:x[:,:,:,1])(x)
        d_time = layers.Lambda(lambda x:x[:,:,:,4])(x)
        
        # Past hour and past day day-of-week embedding
        h_day = layers.Lambda(lambda x:x[:,:,:,2])(x)
        d_day = layers.Lambda(lambda x:x[:,:,:,5])(x)
        
        # Create embedding layer for both timeslot and day-of-week embedding
        time_embed = layers.Embedding(288, embed_size)
        day_embed = layers.Embedding(7, embed_size)
        
        # Pass to embedding layers and then merge
        h_time = time_embed(h_time)
        d_time = time_embed(d_time)
        h_day = day_embed(h_day)
        d_day = day_embed(d_day)
        
        all_layers = [h_traf, d_traf, h_time, d_time, h_day, d_day]
        x = layers.Concatenate(axis=-1)(all_layers)
        
        for cell in mlp_units:
            x = layers.Dense(cell)(x)
        x = layers.Dense(1)(x)
        
        return keras.Model(inputs, x)
        
        
class StackedGRU():
    def build_model(self, in_out_shape, embed_size, mlp_units):
        inputs = keras.Input(shape=in_out_shape, dtype="float32")
        x = inputs
        
        # Slice the inputs
        # Past hour and past day traffic reading
        h_traf = layers.Lambda(lambda x:x[:,:,:,0])(x)
        d_traf = layers.Lambda(lambda x:x[:,:,:,3])(x)
        h_traf = layers.Masking()(h_traf)
        d_traf = layers.Masking()(d_traf)
        h_traf = layers.LayerNormalization(epsilon=1e-6)(h_traf) 
        d_traf = layers.LayerNormalization(epsilon=1e-6)(d_traf)
        traf_shape = h_traf.shape.as_list()[1:]
        traf_shape.append(1)
        traf_shape = tuple(traf_shape)
        h_traf = layers.Reshape(traf_shape)(h_traf)
        d_traf = layers.Reshape(traf_shape)(d_traf)
        
        
        # Past hour and past day timeslot embedding
        h_time = layers.Lambda(lambda x:x[:,:,:,1])(x)
        d_time = layers.Lambda(lambda x:x[:,:,:,4])(x)
        
        # Past hour and past day day-of-week embedding
        h_day = layers.Lambda(lambda x:x[:,:,:,2])(x)
        d_day = layers.Lambda(lambda x:x[:,:,:,5])(x)
        
        # Create embedding layer for both timeslot and day-of-week embedding
        time_embed = layers.Embedding(288, embed_size)
        day_embed = layers.Embedding(7, embed_size)
        
        # Pass to embedding layers and then merge
        h_time = time_embed(h_time)
        d_time = time_embed(d_time)
        h_day = day_embed(h_day)
        d_day = day_embed(d_day)
        
        all_layers = [h_traf, d_traf, h_time, d_time, h_day, d_day]
        x = layers.Concatenate(axis=-1)(all_layers)
        
        
        x = layers.TimeDistributed(layers.GRU(64, activation = 'relu', return_sequences=True))(x)
        x = layers.TimeDistributed(layers.GRU(16, activation = 'relu', return_sequences=True))(x)
        x = layers.TimeDistributed(layers.GRU(1, activation = 'relu', return_sequences=True))(x)
        model = keras.Model(inputs, x)
        return model
        
        
class Seq2Seq():
    def build_model(self, in_out_shape, embed_size, mlp_units):
        inputs = keras.Input(shape=in_out_shape, dtype="float32")
        x = inputs
        
        # Slice the inputs
        # Past hour and past day traffic reading
        h_traf = layers.Lambda(lambda x:x[:,:,:,0])(x)
        d_traf = layers.Lambda(lambda x:x[:,:,:,3])(x)
        h_traf = layers.Masking()(h_traf)
        d_traf = layers.Masking()(d_traf)
        h_traf = layers.LayerNormalization(epsilon=1e-6)(h_traf) 
        d_traf = layers.LayerNormalization(epsilon=1e-6)(d_traf)
        traf_shape = h_traf.shape.as_list()[1:]
        traf_shape.append(1)
        traf_shape = tuple(traf_shape)
        h_traf = layers.Reshape(traf_shape)(h_traf)
        d_traf = layers.Reshape(traf_shape)(d_traf)
        
        
        # Past hour and past day timeslot embedding
        h_time = layers.Lambda(lambda x:x[:,:,:,1])(x)
        d_time = layers.Lambda(lambda x:x[:,:,:,4])(x)
        
        # Past hour and past day day-of-week embedding
        h_day = layers.Lambda(lambda x:x[:,:,:,2])(x)
        d_day = layers.Lambda(lambda x:x[:,:,:,5])(x)
        
        # Create embedding layer for both timeslot and day-of-week embedding
        time_embed = layers.Embedding(288, embed_size)
        day_embed = layers.Embedding(7, embed_size)
        
        # Pass to embedding layers and then merge
        h_time = time_embed(h_time)
        d_time = time_embed(d_time)
        h_day = day_embed(h_day)
        d_day = day_embed(d_day)
        
        all_layers = [h_traf, d_traf, h_time, d_time, h_day, d_day]
        x = layers.Concatenate(axis=-1)(all_layers)
        num_nodes = x.shape[2]
        
        for i in range(len(mlp_units)-1):
            x = layers.TimeDistributed(layers.LSTM(mlp_units[i], activation = 'relu', return_sequences=True))(x)
        x = layers.TimeDistributed(layers.LSTM(mlp_units[-1], activation = 'relu'))(x)
        x = layers.TimeDistributed(layers.RepeatVector(num_nodes))(x)
        for grucell in mlp_units:
            x = layers.TimeDistributed(layers.LSTM(grucell, activation = 'relu', return_sequences=True))(x)
        x = layers.TimeDistributed(layers.Dense(16, activation = 'relu'))(x)
        x = layers.TimeDistributed(layers.Dense(1, activation = 'relu'))(x)
        model = keras.Model(inputs, x)
        return model
        