import torch
from . import coordinates, networks, transformers

class ET(object):
    def __init__(
        self,
        tfs=[],
        coords=coordinates.identity_grid,
        net=None,
        equivariant=True,
        downsample=1,
        tf_opts={},
        net_opts={},
        load_path=None,
    ):
        if load_path is not None:
            spec = torch.load(load_path)
            tfs = spec["tfs"]
            coords = spec["coords"]
            net = spec["net"]
            equivariant = spec["equivariant"]
            downsample = spec["downsample"]
            tf_opts = spec["tf_opts"]
            net_opts = spec["net_opts"]

        if net is None:
            raise ValueError("net parameter must be specified")

        if len(tfs) > 0:
            pose_module = (
                networks.EquivariantPosePredictor
                if equivariant
                else networks.DirectPosePredictor
            )
            tfs = [getattr(transformers, tf) if type(tf) is str else tf for tf in tfs]
            seq = transformers.TransformerSequence(
                *[tf(pose_module, **tf_opts) for tf in tfs]
            )
        else:
            seq = None

        if type(coords) is str:
            if hasattr(coordinates, coords):
                coords = getattr(coordinates, coords)
            elif hasattr(coordinates, coords + "_grid"):
                coords = getattr(coordinates, coords + "_grid")
            else:
                raise ValueError("Invalid coordinate system: " + coords)

        if type(net) is str:
            net = getattr(networks, net)
        network = net(**net_opts)
        self.tfs = tfs
        self.coords = coords
        self.downsample = downsample
        self.net = net
        self.equivariant = equivariant
        self.tf_opts = tf_opts
        self.net_opts = net_opts
        self.model = self._build_model(
            net=network, transformer=seq, coords=coords, downsample=downsample
        )

        if load_path is not None:
            self.model.load_state_dict(spec["state_dict"])

    def _build_model(self, net, transformer, coords, downsample):
        return networks.TransformerCNN(
            net=net, transformer=transformer, coords=coords, downsample=downsample
        )

    def _save(self, path, **kwargs):
        spec = {
            "tfs": [tf.__name__ for tf in self.tfs],
            "coords": self.coords.__name__,
            "net": self.net.__name__,
            "equivariant": self.equivariant,
            "downsample": self.downsample,
            "tf_opts": self.tf_opts,
            "net_opts": self.net_opts,
            "state_dict": self.model.state_dict(),
        }
        spec.update(kwargs)
        torch.save(spec, path)