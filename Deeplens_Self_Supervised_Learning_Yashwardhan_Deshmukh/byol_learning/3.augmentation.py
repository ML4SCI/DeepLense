def random_flip_rotate_resize(image):
    # Horizontal flipping
    horizontal_flip = tf.random.uniform(shape=[])
    if horizontal_flip < 0.5:
        image = tf.image.flip_left_right(image)
    # Vertical flipping
    vertical_flip = tf.random.uniform(shape=[])
    if vertical_flip < 0.5:
        image = tf.image.flip_up_down(image)
    # Random rotation
    image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    # Resizing to original size
    image = tf.image.resize(image, size=[64, 64])
    
    return image


@tf.function
def augment_image_pretraining(image):
    image = random_flip_rotate_resize(image)
    return image