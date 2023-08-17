def get_timm_model(
    name,
    complex=True,
    dropout_rate=0.3,
    in_chans=1,
    num_classes=3,
    pretrained=True,
    tune=False,
):
    """
    Selects the simple or complex Timm Model class based on "complex" parameter.
    """
    if complex:
        from .timm_model import TimmModelComplex

        return TimmModelComplex(
            name,
            dropout_rate=dropout_rate,
            in_chans=in_chans,
            num_classes=num_classes,
            pretrained=pretrained,
            tune=tune,
        )
    else:
        from .timm_model import TimmModelSimple

        return TimmModelSimple(
            name,
            in_chans=in_chans,
            num_classes=num_classes,
            pretrained=pretrained,
            tune=tune,
        )
