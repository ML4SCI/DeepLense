model_supervised = tf.keras.models.Sequential([
    get_encoder(),
    tf.keras.layers.Dense(3, activation='softmax'),
])
model_supervised.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate= 1e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
    metrics=['acc', tf.keras.metrics.AUC(name='auc')]
)

history = model_supervised.fit(
    data.train_generator(),
    steps_per_epoch=steps_per_epoch,
    validation_data=data.test_generator(),
    validation_steps=test_steps,
    epochs=num_epochs
)