from .cvt import CvT
from .hybrid_swin import Hybrid

def get_transformer_model(name, dropout=0., image_size=150, num_classes=3):
    if name == 'cvt':
        return CvT(num_classes=num_classes, dropout=dropout)
    elif name == 'swin_hybrid':
        return Hybrid(image_size=image_size, num_classes=num_classes)
    else:
        return None
