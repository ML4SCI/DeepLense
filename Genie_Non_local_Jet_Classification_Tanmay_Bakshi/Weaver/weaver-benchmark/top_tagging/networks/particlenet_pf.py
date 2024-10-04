import torch
import torch.nn as nn
from utils.nn.model.ParticleNet import ParticleNet, FeatureConv


class ParticleNetTagger1Path(nn.Module):

    def __init__(self,
                 pf_features_dims,
                 num_classes,
                 conv_params=[(7, (32, 32, 32)), (7, (64, 64, 64))],
                 fc_params=[(128, 0.1)],
                 use_fusion=True,
                 use_fts_bn=True,
                 use_counts=True,
                 pf_input_dropout=None,
                 for_inference=False,
                 **kwargs):
        super(ParticleNetTagger1Path, self).__init__(**kwargs)
        self.pf_input_dropout = nn.Dropout(pf_input_dropout) if pf_input_dropout else None
        self.pf_conv = FeatureConv(pf_features_dims, 32)
        self.pn = ParticleNet(input_dims=32,
                              num_classes=num_classes,
                              conv_params=conv_params,
                              fc_params=fc_params,
                              use_fusion=use_fusion,
                              use_fts_bn=use_fts_bn,
                              use_counts=use_counts,
                              for_inference=for_inference)

    def forward(self, pf_points, pf_features, pf_mask):
        if self.pf_input_dropout:
            pf_mask = (self.pf_input_dropout(pf_mask) != 0).float()
            pf_points *= pf_mask
            pf_features *= pf_mask

        return self.pn(pf_points, self.pf_conv(pf_features * pf_mask) * pf_mask, pf_mask)


def get_model(data_config, **kwargs):
    # conv_params = [
    #     (16, (64, 64, 64)),
    #     (16, (128, 128, 128)),
    #     (16, (256, 256, 256)),
    #     ]
    ec_k = kwargs.get('ec_k', 16)
    ec_c1 = kwargs.get('ec_c1', 64)
    ec_c2 = kwargs.get('ec_c2', 128)
    ec_c3 = kwargs.get('ec_c3', 256)
    fc_c, fc_p = kwargs.get('fc_c', 256), kwargs.get('fc_p', 0.1)
    conv_params = [
        (ec_k, (ec_c1, ec_c1, ec_c1)),
        (ec_k, (ec_c2, ec_c2, ec_c2)),
        (ec_k, (ec_c3, ec_c3, ec_c3)),
        ]
    fc_params = [(fc_c, fc_p)]
    use_fusion = True

    pf_features_dims = len(data_config.input_dicts['pf_features'])
    num_classes = len(data_config.label_value)
    model = ParticleNetTagger1Path(pf_features_dims, num_classes,
                                   conv_params, fc_params,
                                   use_fusion=use_fusion,
                                   use_fts_bn=kwargs.get('use_fts_bn', False),
                                   use_counts=kwargs.get('use_counts', True),
                                   pf_input_dropout=kwargs.get('pf_input_dropout', None),
                                   for_inference=kwargs.get('for_inference', False)
                                   )
    model_info = {
        'input_names':list(data_config.input_names),
        'input_shapes':{k:((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names':['softmax'],
        'dynamic_axes':{**{k:{0:'N', 2:'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax':{0:'N'}}},
        }

    return model, model_info


def get_loss(data_config, **kwargs):
    return torch.nn.CrossEntropyLoss()