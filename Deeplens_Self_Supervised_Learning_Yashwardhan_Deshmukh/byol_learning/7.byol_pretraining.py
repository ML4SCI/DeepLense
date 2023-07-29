class ProjectionHead(tf.keras.Model):
    def __init__(self):
        super(ProjectionHead, self).__init__()
        self.fc1 = tf.keras.layers.Dense(units=64)
        self.bn = tf.keras.layers.BatchNormalization()
        self.fc2 = tf.keras.layers.Dense(units=32)

    def call(self, inp, training=False):
        x = self.fc1(inp)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        x = self.fc2(x)
        return x

def byol_loss(p, z):
    """
    It calculates the mean similarity between the prediction p and the target z, 
    which are both normalized to unit length with tf.math.l2_normalize. It then returns 2 - 2 * mean_similarity, 
    which is equivalent to 2(1 - mean_similarity), so that the loss is minimized when the cosine similarity between p and z is maximized.
    """
    p = tf.math.l2_normalize(p, axis=1)  
    z = tf.math.l2_normalize(z, axis=1)  

    similarities = tf.reduce_sum(tf.multiply(p, z), axis=1)
    return 2 - 2 * tf.reduce_mean(similarities)

# Instantiate networks
f_online = get_encoder()
g_online = ProjectionHead()
q_online = ProjectionHead()

f_target = get_encoder()
g_target = ProjectionHead()


# Initialize the weights of the networks
x = tf.random.normal((batch_size, 64, 64, 3))
h = f_online(x, training=False)
print('Initializing online networks...')
print('Shape of h:', h.shape)
z = g_online(h, training=False)
print('Shape of z:', z.shape)
p = q_online(z, training=False)
print('Shape of p:', p.shape)

h = f_target(x, training=False)
print('Initializing target networks...')
print('Shape of h:', h.shape)
z = g_target(h, training=False)
print('Shape of z:', z.shape)

num_params_f = tf.reduce_sum([tf.reduce_prod(var.shape) for var in f_online.trainable_variables])    
print('The encoders have {} trainable parameters each.'.format(num_params_f))

# Define optimizer
lr = 1e-4
opt = tf.keras.optimizers.legacy.Adam(learning_rate=lr)
print('Using Adam optimizer with learning rate {}.'.format(lr))
    
class BYOLModel(tf.keras.Model):
    def __init__(self, f_online, g_online, q_online, f_target, g_target, beta=0.99, **kwargs):
        super().__init__(**kwargs)
        self.f_online = f_online
        self.g_online = g_online
        self.q_online = q_online
        self.f_target = f_target
        self.g_target = g_target
        self.beta = beta
        
    def compile(self, optimizer, **kwargs):
        super().compile(**kwargs)
        self.optimizer = optimizer
    
    def train_step(self, data):
        x1, x2 = data
        # Forward pass
        with tf.GradientTape(persistent=True) as tape:
            h_target_1 = self.f_target(x1, training=True)
            z_target_1 = self.g_target(h_target_1, training=True)

            h_target_2 = self.f_target(x2, training=True)
            z_target_2 = self.g_target(h_target_2, training=True)

            h_online_1 = self.f_online(x1, training=True)
            z_online_1 = self.g_online(h_online_1, training=True)
            p_online_1 = self.q_online(z_online_1, training=True)
            
            h_online_2 = self.f_online(x2, training=True)
            z_online_2 = self.g_online(h_online_2, training=True)
            p_online_2 = self.q_online(z_online_2, training=True)
            
            p_online = tf.concat([p_online_1, p_online_2], axis=0)
            z_target = tf.concat([z_target_2, z_target_1], axis=0)
            loss = byol_loss(p_online, z_target)

        # Calculate gradients and update online networks
        trainable_vars = self.f_online.trainable_variables + self.g_online.trainable_variables + self.q_online.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        # Update target networks
        for online_var, target_var in zip(self.f_online.variables, self.f_target.variables):
            target_var.assign(self.beta * target_var + (1 - self.beta) * online_var)
        
        for online_var, target_var in zip(self.g_online.variables, self.g_target.variables):
            target_var.assign(self.beta * target_var + (1 - self.beta) * online_var)
        return {"loss": loss}
    
byol_model = BYOLModel(f_online, g_online, q_online, f_target, g_target)
byol_model.compile(optimizer=opt)

byol_model.fit(data.pretrain_generator(), epochs=num_epochs, steps_per_epoch=steps_per_epoch)