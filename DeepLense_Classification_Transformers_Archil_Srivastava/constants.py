NUM_CLASSES = 3
LABELS = ['axion', 'cdm', 'no_sub']
LABEL_MAP = {'axion': 0, 'cdm': 1, 'no_sub': 2}

TIMM_IMAGE_SIZE = {
    'vit_base_patch16_224': 224,
    'convit_small': 224,
    'convit_tiny': 224,
    'vit_small_r26_s32_224': 224,
    'vit_base_r26_s32_224': 224,
    'vit_tiny_r_s16_p8_224': 224,
    'botnet26t_256': 256,
    'crossvit_small_240': 240,
    'crossvit_base_240': 240,
    'crossvit_9_dagger_240': 240,
    'levit_128': 128,
    'levit_192': 192,
    'levit_256': 256,
    'twins_svt_base': 224,
    'twins_svt_small': 224,
    'swinv2_small_window16_256': 256,
    'swinv2_small_window8_256': 256,
    'swinv2_tiny_window16_256': 256,
    'swinv2_tiny_window8_256': 256,
    'swin_small_patch4_window7_224': 224,
    'swin_tiny_patch4_window7_224': 224,
    'coatnet_1_rw_224': 224,
    'coatnet_0_rw_224': 224,
    'coatnet_bn_0_rw_224': 224,
    'coatnet_nano_rw_224': 224,
    'coat_lite_small': 224,
    'efficientformer_l3': 224,
    'efficientformer_l7': 224,
    'efficientnet_b1': 50
}