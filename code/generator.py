import torch
import cv2 as cv
import torch.nn as nn
import salience

# region Hyperparameters
# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 128

# The number of classes
num_classes = 4

# Number of channels in the training images
nc = 3

# Size of the image
image_size = 32

# GPU vs CPU
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

# Dimensions and padding
dim = [32, 24]
padding = int((dim[0]-dim[1])/2)

# One hot labels
onehot = torch.zeros(num_classes, num_classes)
onehot = onehot.scatter_(1, torch.LongTensor(list(range(num_classes))).view(
    num_classes, 1), 1).view(num_classes, num_classes, 1, 1)
# endregion


class Generator(nn.Module):
    """Generator model."""

    def __init__(self):
        super(Generator, self).__init__()
        self.deconv_z = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf*4, 4, 1, 0),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(0.2),
            nn.Dropout(0.4),
        )

        self.deconv_label = nn.Sequential(
            nn.ConvTranspose2d(num_classes, ngf*4, 4, 1, 0),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(0.2),
            nn.Dropout(0.4),
        )

        self.main = nn.Sequential(
            nn.ConvTranspose2d((ngf*4)*2, ngf*4, 4, 2, 1),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(0.2),
            nn.Dropout(0.4),

            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(0.2),

            nn.ConvTranspose2d(ngf*2, nc, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z, labels):
        # Create label embedding
        z = self.deconv_z(z)
        labels = self.deconv_label(labels)
        x = torch.cat([z, labels], dim=1)

        return self.main(x)


def load_checkpoint():
    """Load the pretrained model from the saved checkpoint."""
    checkpoint = torch.load("generator.pth", map_location=torch.device(device))
    netG.load_state_dict(checkpoint)


# region Create the generator
netG = Generator().to(device)
load_checkpoint()
# endregion


def generate_layout(label):
    """Generate a layout conditioned on a text proportion label [0,3]."""
    # Generate noise and labels
    noise = torch.randn(nz).repeat(4)
    noise = noise.view(-1, nz, 1, 1).to(device)
    labels = torch.tensor([0, 1, 2, 3])
    labels = onehot[labels].to(device)

    # Generate image
    image = netG(noise, labels)[label]
    image = image[:, :, padding:padding+dim[1]]

    return image


def cvt_img_torch_to_opencv(img):
    """Convert an image from pytorch format to opencv format."""
    img = img.permute(1, 2, 0).cpu().detach().numpy()
    return cv.cvtColor(img, cv.COLOR_RGB2BGR)


def show_gen(gen):
    """Show a generated layout image."""
    cv.imshow("gen", salience.scale_image(cvt_img_torch_to_opencv(gen), 10))
    cv.waitKey(0)
