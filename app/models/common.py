import torch
from torch import nn
from torchvision import transforms

unnormalize = transforms.Compose(
    [
        transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
        ),
    ]
)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def intensity_loss(gt, pred):
    return torch.mean(torch.abs(gt - pred))


def imshow(img):
    import matplotlib.pyplot as plt

    img = unnormalize(img)
    plt.imshow(img.detach().cpu().numpy().transpose(1, 2, 0))


class LoggingMixin:
    def log_rgb(self, tag, img):
        logger = self.logger.experiment
        logger.add_image(
            tag,
            torch.clip(unnormalize(img), 0, 1),
            global_step=self.trainer.global_step,
        )

    def log_mask(self, tag, img):
        self.logger.experiment.add_image(
            tag, img, dataformats="hw", global_step=self.trainer.global_step
        )

    def log_image(self, *args, **kwargs):
        self.logger.experiment.add_image(
            *args, global_step=self.trainer.global_step, **kwargs
        )
