"""This module contains implementations of the transformer model by fchollet"""

from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs):
        attention_output = self.attention(query=inputs, value=inputs, key=inputs)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)


class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, latent_dim, num_heads, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential(
            [layers.Dense(latent_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, encoder_outputs):
        attention_output_1 = self.attention_1(
            query=inputs, value=inputs, key=inputs, attention_mask=None
        )
        attention_output_1 = layers.Attention(causal=True)([inputs, encoder_outputs])
        out_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=None,
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )
        return tf.tile(mask, mult)
        
        
class TrafFormer():
    def get_model(self, shape):
        embed_dim = 256
        latent_dim = 2048
        num_heads = 8
        sequence_length = 12
        day_vocab = 7
        hour_vocab = 288
        
        ##################
        # Encoder inputs #
        ##################
        encoder_inputs = keras.Input(shape=shape, dtype="float16", name="encoder_inputs")
        # Process traffic reading
        e_traf_hour = layers.Lambda(lambda x:x[:,:,:,0])(encoder_inputs)
        e_traf_day = layers.Lambda(lambda x:x[:,:,:,3])(encoder_inputs)
        e_traf_hour = layers.Masking()(e_traf_hour)
        e_traf_day = layers.Masking()(e_traf_day)
        e_traf_shape = e_traf_hour.shape.as_list()[1:]
        e_traf_shape.append(1)
        e_traf_shape = tuple(e_traf_shape)
        e_traf_hour = layers.Reshape(e_traf_shape)(e_traf_hour)
        e_traf_day = layers.Reshape(e_traf_shape)(e_traf_day)
        # Past hour and past day timeslot embedding
        e_h_time = layers.Lambda(lambda x:x[:,:,:,1])(encoder_inputs)
        e_d_time = layers.Lambda(lambda x:x[:,:,:,4])(encoder_inputs)
        # Past hour and past day day-of-week embedding
        e_h_day = layers.Lambda(lambda x:x[:,:,:,2])(encoder_inputs)
        e_d_day = layers.Lambda(lambda x:x[:,:,:,5])(encoder_inputs)
        # Time and day embedding
        timeembed = PositionalEmbedding(sequence_length, hour_vocab, embed_dim)
        dayembed = PositionalEmbedding(sequence_length, day_vocab, embed_dim)
        e_time_hour = timeembed(e_h_time)
        e_time_day = timeembed(e_d_time)
        e_day_hour = dayembed(e_h_day)
        e_day_day = dayembed(e_d_day)
        # Concat and output
        e_concat = layers.Concatenate()([e_traf_hour, e_traf_day, e_time_hour, e_time_day, 
                                       e_day_hour, e_day_day])
        encoder_outputs = TransformerEncoder(embed_dim*4+2, latent_dim, num_heads)(e_concat)
        encoder = keras.Model(encoder_inputs, encoder_outputs)
        
        
        ##################
        # Decoder inputs #
        ##################
        # Process traffic reading
        decoder_inputs = keras.Input(shape=(shape), dtype="float16", name="decoder_inputs")
        encoded_seq_inputs = keras.Input(shape=encoder_outputs.shape[1:].as_list(), name="decoder_state_inputs")
        d_traf_hour = layers.Lambda(lambda x:x[:,:,:,0])(decoder_inputs)
        d_traf_day = layers.Lambda(lambda x:x[:,:,:,3])(decoder_inputs)
        d_traf_hour = layers.Masking()(d_traf_hour)
        d_traf_day = layers.Masking()(d_traf_day)
        d_traf_shape = d_traf_hour.shape.as_list()[1:]
        d_traf_shape.append(1)
        d_traf_shape = tuple(d_traf_shape)
        d_traf_hour = layers.Reshape(d_traf_shape)(d_traf_hour)
        d_traf_day = layers.Reshape(d_traf_shape)(d_traf_day)
        # Past hour and past day timeslot embedding
        d_h_time = layers.Lambda(lambda x:x[:,:,:,1])(decoder_inputs)
        d_d_time = layers.Lambda(lambda x:x[:,:,:,4])(decoder_inputs)
        # Past hour and past day day-of-week embedding
        d_h_day = layers.Lambda(lambda x:x[:,:,:,2])(decoder_inputs)
        d_d_day = layers.Lambda(lambda x:x[:,:,:,5])(decoder_inputs)
        # Time and day embedding
        timeembed = PositionalEmbedding(sequence_length, hour_vocab, embed_dim)
        dayembed = PositionalEmbedding(sequence_length, day_vocab, embed_dim)
        d_time_hour = timeembed(d_h_time)
        d_time_day = timeembed(d_d_time)
        d_day_hour = dayembed(d_h_day)
        d_day_day = dayembed(d_d_day)
        # Concat and output
        d_concat = layers.Concatenate()([d_traf_hour, d_traf_day, d_time_hour, 
                                       d_time_day, d_day_hour, d_day_day])
        decoder_outputs = TransformerDecoder(embed_dim*4+2, latent_dim, num_heads)(d_concat, encoded_seq_inputs)
        
        ####################
        # Full transformer #
        ####################
        decoder_outputs = layers.Dropout(0.5)(decoder_outputs)
        decoder_outputs = layers.Dense(512, activation="relu")(decoder_outputs)
        decoder_outputs = layers.Dropout(0.5)(decoder_outputs)
        decoder_outputs = layers.Dense(64, activation="relu")(decoder_outputs)
        decoder_outputs = layers.Dropout(0.5)(decoder_outputs)
        decoder_outputs = layers.Dense(1, activation="relu")(decoder_outputs)
        decoder = keras.Model([decoder_inputs, encoded_seq_inputs], decoder_outputs)
        decoder_outputs = decoder([decoder_inputs, encoder_outputs])
        
        transformer = keras.Model(
            [encoder_inputs, decoder_inputs], decoder_outputs, name="transformer"
        )
        return transformer