class Dummy_Input(tf.keras.layers.Layer):
    """
    This is the class for a 'classification token' mentioned in the ViT Paper
    
    @inproceedings{50650,
    title	= {An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
    author	= {Alexander Kolesnikov and Alexey Dosovitskiy and Dirk Weissenborn and Georg Heigold and Jakob Uszkoreit and Lucas Beyer and Matthias Minderer and Mostafa Dehghani and Neil Houlsby and Sylvain Gelly and Thomas Unterthiner and Xiaohua Zhai},
    year	= {2021}
    }

    This adds a dummy input to an input layer which acts as a placeholder data structure that’s used to store information 
    that is extracted from other tokens in the sequence. 
    
    By allocating an empty token for this procedure, it seems like the Vision Transformer makes 
    it less likely to bias the final output towards or against any single one of the other individual tokens.  
    """
    def build(self, input_dimensions):
        self.dummy_val = self.add_weight(name="dummy_input",
                                         shape=(1, 1, input_dimensions[-1]), 
                                         initializer=tf.zeros_initializer(),
                                         trainable=True,)

    def call(self, inputs):
        dummy_val_token = tf.tile(self.dummy_val, 
                                  [tf.shape(inputs)[0], 1, 1])
        return tf.concat([dummy_val_token, 
                          inputs], axis=1)

class Inherit_Positional_Embeddings(tf.keras.layers.Layer):
    """
    In Vision Transformers, the input data is typically a 2D image that is flattened into a sequence of patches, which are then fed into the transformer model. 
    Since transformers do not inherently model positional information, additional positional embeddings are added to the input sequence in this code 
    to enable the model to take into account the spatial relationships between the patches.
    """

    def build(self, input_shape):
        self.learned_positional_embeddings = tf.Variable(name="pos_embedding",
                                                         initial_value=tf.initializers.random_normal(stddev=0.06)(shape=(1, input_shape[1], input_shape[2])),
                                                         dtype="float32", 
                                                         trainable=True,)
    def call(self, inputs):
        return inputs + self.learned_positional_embeddings

class MultiHeadSelfAttention(tf.keras.layers.Layer):
    """
    From, 
        
    @inproceedings{50650,
    title	= {An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
    author	= {Alexander Kolesnikov and Alexey Dosovitskiy and Dirk Weissenborn and Georg Heigold and Jakob Uszkoreit and Lucas Beyer and Matthias Minderer and Mostafa Dehghani and Neil Houlsby and Sylvain Gelly and Thomas Unterthiner and Xiaohua Zhai},
    year	= {2021}
    }
    
    Multihead self-attention (MSA) is an extension of SA in which we run k self-attention operations,
    called “heads”, in parallel, and project their concatenated outputs. 
    
    MSA(z) = [SA1(z); SA2(z); · · · ; SAk(z)]Umsa  
    where, Umsa ∈ R k·Dh×D is a learnable weight matrix. Credits to faustomorales the author or vit-keras for simplification of MultiHeadSelfAttention in his library.
    
    """
    def __init__(self, *args, num_heads, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads

    def build(self, input_shape):
        hidden_size = input_shape[-1]
        num_heads = self.num_heads
        self.hidden_size = hidden_size
        self.projection_dim = hidden_size // num_heads
        self.query = tf.keras.layers.Dense(hidden_size, name="query")
        self.key = tf.keras.layers.Dense(hidden_size, name="key")
        self.value = tf.keras.layers.Dense(hidden_size, name="value")
        self.combine_heads = tf.keras.layers.Dense(hidden_size, name="out")

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], score.dtype)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query(inputs)
        key = self.key(inputs)
        value = self.value(inputs)
        query = tf.transpose(tf.reshape(query, (batch_size, -1, self.num_heads, self.projection_dim)), perm=[0, 2, 1, 3])
        key = tf.transpose(tf.reshape(key, (batch_size, -1, self.num_heads, self.projection_dim)), perm=[0, 2, 1, 3])
        value = tf.transpose(tf.reshape(value, (batch_size, -1, self.num_heads, self.projection_dim)), perm=[0, 2, 1, 3])

        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.hidden_size))
        output = self.combine_heads(concat_attention)
        return output, weights


class TransformerBlock(tf.keras.layers.Layer):
    """Implements a Transformer block."""

    def __init__(self, *args, num_heads, mlp_dim, dropout, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout

    def build(self, input_shape):
        self.att = MultiHeadSelfAttention(
            num_heads=self.num_heads,
            name="MultiHeadDotProductAttention_1",
        )
        self.mlpblock = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    self.mlp_dim,
                    activation="linear",
                    name=f"{self.name}/Dense_0",
                ),
                tf.keras.layers.Lambda(
                    lambda x: gelu(x, approximate=False)
                )
                if hasattr(tf.keras.activations, "gelu")
                else tf.keras.layers.Lambda(
                    lambda x: gelu(x, approximate=False)
                ),
                tf.keras.layers.Dropout(self.dropout),
                tf.keras.layers.Dense(input_shape[-1], name=f"{self.name}/Dense_1"),
                tf.keras.layers.Dropout(self.dropout),
            ],
            name="MlpBlock_3",
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name="LayerNorm_0"
        )
        self.layernorm2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name="LayerNorm_2"
        )
        self.dropout_layer = tf.keras.layers.Dropout(self.dropout)

    def call(self, inputs, training):
        x = self.layernorm1(inputs)
        x, weights = self.att(x)
        x = self.dropout_layer(x, training=training)
        x = x + inputs
        y = self.layernorm2(x)
        y = self.mlpblock(y)
        return x + y, weights

def load_ViT_model_with_no_weights(): 
    x = Input(shape=(64, 64, 3)) 
    y = Conv2D(filters=768, kernel_size=32,strides=32,padding="valid", name="embedding",)(x)
    y = Reshape((y.shape[1] * y.shape[2], 768))(y)
    # y = Reshape((5, 768))(y)

    y = Dummy_Input(name="dummy_input")(y)
    y = Inherit_Positional_Embeddings(name="Transformer/posembed_input")(y)
    
    for n in range(12):
        y, _ = TransformerBlock(
            num_heads=12,
            mlp_dim=3072,
            dropout=0.1,
            name=f"Transformer/encoderblock_{n}",)(y)
    y = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, name="Transformer/encoder_norm")(y)
    y = tf.keras.layers.Lambda(lambda v: v[:, 0], name="ExtractToken")(y)
    y = Dense(768, name="pre_logits", activation="tanh")(y)
    model = Model(inputs=x, outputs=y)
    return model

load_vit = load_ViT_model_with_no_weights()
def get_encoder():
    top_layers = tf.keras.models.Sequential([ 
        BatchNormalization(),
        Dense(32, activation = gelu),
        Dropout(0.2),
        BatchNormalization(),
        Dense(16, activation = gelu),
        BatchNormalization(),
    ], name='modified_vit_top')
    model = Sequential([
        load_vit,
        top_layers
    ], name='modified_vit')
    return model