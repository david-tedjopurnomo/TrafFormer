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