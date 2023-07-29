pretrained_encoder = byol_model.f_online
finetuning_model = tf.keras.models.Sequential([pretrained_encoder, tf.keras.layers.Dense(3, activation='softmax')])
finetuning_model.compile(optimizer=keras.optimizers.Adam(learning_rate= 1e-4),loss=keras.losses.CategoricalCrossentropy(from_logits=False),metrics=['acc', metrics.AUC(name='auc')])
finetuning_model.summary()

history_finetuning = finetuning_model.fit(
    data.train_generator(),
    steps_per_epoch=steps_per_epoch,
    validation_data=data.test_generator(),
    validation_steps=test_steps,
    epochs=num_epochs
)