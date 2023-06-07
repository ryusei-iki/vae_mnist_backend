from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import base64
from PIL import Image
import io
# from torch import nn
# import torch


app = FastAPI()
data_num = 1000
x = np.zeros((data_num, 32 * 32))
for i in range(data_num):
    x[i] = np.array(Image.open('datasets/real_images/{}.png'.format(i)).convert('L')).reshape(-1)
    x[i] = x[i] / 255
z = np.load('parameter/middle.npy')
width = 500
height = 500
z[:, 1] = z[:, 1] * (-1)
kari_x = (z[:, 0] - np.min(z)) / (np.max(z) - np.min(z)) * width
kari_y = (z[:, 1] - np.min(z)) / (np.max(z) - np.min(z)) * height

z[:, 0] = kari_x
z[:, 1] = kari_y


# class Encoder(nn.Module):
#     def __init__(self, in_channels, latent_dim, hidden_dims, img_size):
#         super(Encoder, self).__init__()
#         self.latent_dim = latent_dim[0]

#         modules = []
#         if hidden_dims is None:
#             hidden_dims = [32, 64, 128, 256, 512]

#         # Build Encoder
#         for h_dim in hidden_dims:
#             modules.append(
#                 nn.Sequential(
#                     nn.Conv2d(in_channels, out_channels=h_dim,
#                               kernel_size=3, stride=2, padding=1),
#                     nn.BatchNorm2d(h_dim),
#                     nn.LeakyReLU())
#             )
#             in_channels = h_dim

#         self.encoder = nn.Sequential(*modules)
#         self.fc_mu = nn.Linear(hidden_dims[-1] * int(((img_size) / (2**len(hidden_dims)))**2), latent_dim[0])
#         self.fc_var = nn.Linear(hidden_dims[-1] * int(((img_size) / (2**len(hidden_dims)))**2), latent_dim[0])

#     def forward(self, input):
#         result = self.encoder(input)
#         result = torch.flatten(result, start_dim=1)

#         # Split the result into mu and var components
#         # of the latent Gaussian distribution
#         # print(result.size())
#         mu = self.fc_mu(result)
#         log_var = self.fc_var(result)

#         return [mu, log_var]


# class Decoder(nn.Module):
#     def __init__(self, in_channels, latent_dim, hidden_dims, img_size):
#         super(Decoder, self).__init__()
#         self.decode_first = Decoder_first(in_channels, latent_dim, hidden_dims, img_size)
#         self.decode_last = Decoder_last(in_channels, latent_dim, hidden_dims, img_size)

#     def forward(self, z):
#         result = self.decode_last(self.decode_first(z))
#         return result


# class Decoder_first(nn.Module):
#     def __init__(self, in_channels, latent_dim, hidden_dims, img_size):
#         super(Decoder_first, self).__init__()
#         if hidden_dims is None:
#             hidden_dims = [32, 64, 128, 256, 512]
#         self.decoder_input = nn.Linear(latent_dim[0], latent_dim[1])

#     def forward(self, z):
#         result = self.decoder_input(z)
#         return result


# class Decoder_last(nn.Module):
#     def __init__(self, in_channels, latent_dim, hidden_dims, img_size):
#         super(Decoder_last, self).__init__()
#         modules = []
#         self.hidden_dims = hidden_dims
#         if hidden_dims is None:
#             hidden_dims = [32, 64, 128, 256, 512]
#         self.decoder_input = nn.Linear(latent_dim[1], hidden_dims[-1] * int(((img_size) / (2**len(hidden_dims))) ** 2))
#         hidden_dims.reverse()

#         for i in range(len(hidden_dims) - 1):
#             modules.append(
#                 nn.Sequential(
#                     nn.ConvTranspose2d(hidden_dims[i],
#                                        hidden_dims[i + 1],
#                                        kernel_size=3,
#                                        stride=2,
#                                        padding=1,
#                                        output_padding=1),
#                     nn.BatchNorm2d(hidden_dims[i + 1]),
#                     nn.LeakyReLU())
#             )

#         self.decoder = nn.Sequential(*modules)

#         self.final_layer = nn.Sequential(
#             nn.ConvTranspose2d(
#                 hidden_dims[-1],
#                 hidden_dims[-1],
#                 kernel_size=3,
#                 stride=2,
#                 padding=1,
#                 output_padding=1),
#             nn.BatchNorm2d(hidden_dims[-1]),
#             nn.LeakyReLU(),
#             nn.Conv2d(
#                 hidden_dims[-1],
#                 out_channels=in_channels,
#                 kernel_size=3,
#                 padding=1),
#             nn.Tanh())

#     def forward(self, result):
#         result = self.decoder_input(result)
#         result = result.view(-1, self.hidden_dims[0], int(((img_size) / (2**len(hidden_dims)))), int(((img_size) / (2**len(hidden_dims)))))
#         result = self.decoder(result)
#         result = self.final_layer(result)
#         return result


# class VAE(nn.Module):
#     def __init__(self, in_channels, latent_dim, hidden_dims, img_size):
#         super(VAE, self).__init__()
#         self.encode = Encoder(in_channels, latent_dim, hidden_dims, img_size)
#         self.decode = Decoder(in_channels, latent_dim, hidden_dims, img_size)

#     def reparameterize(self, mu, logvar):
#         """
#         Reparameterization trick to sample from N(mu, var) from
#         N(0,1).
#         :param mu: (Tensor) Mean of the latent Gaussian [B x D]
#         :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
#         :return: (Tensor) [B x D]
#         """
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return eps * std + mu

#     def forward(self, input):
#         mu, log_var = self.encode(input)
#         z = self.reparameterize(mu, log_var)
#         return [self.decode(z), input, mu, log_var]

# in_channels = 1
# latent_dim = [2, 124]
# hidden_dims = [32, 64, 128, 256]
# img_size = 32
# device = 'cpu'
# epoch = 100
# model = VAE(in_channels, latent_dim, hidden_dims, img_size).to(device)
# model.load_state_dict(torch.load('parameter/{}.pth'.format(epoch), map_location=torch.device('cpu')))


origins = [
    "http://localhost:3000",
    "https://mnist-ukr.web.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class ClickPosition(BaseModel):
    x: float
    y: float
# @app.post('/click')
# def create_user(clickposition: ClickPosition):

#     input_z = np.array([[clickposition.x/500 * (np.max(z) - np.min(z)) + np.min(z), ((clickposition.y * -1) / 500 + 1) * (np.max(z) - np.min(z)) + np.min(z)]])

#     nearnum = np.argmin(np.sum((input_z - z)**2, axis=1))
#     nearnum = int(nearnum)

#     input_z = torch.tensor(input_z, dtype=torch.float32)
#     out = model.decode(input_z)
#     out = (out - torch.min(out)) / (torch.max(out) - torch.min(out))
#     out = out.to('cpu').detach().numpy().copy()
#     out = out.reshape(img_size, img_size) * 255
#     pil_img = Image.fromarray(out.astype(np.uint8))
#     stream = io.BytesIO()
#     pil_img.save(stream, format='PNG')  # 画像をPNG形式で保存する場合
#     image_base64 = base64.b64encode(stream.getvalue()).decode('utf-8')

#     near_img = x[nearnum].reshape(32, 32)
#     near_img = near_img * 255

#     near_img = Image.fromarray(near_img.astype(np.uint8))
#     stream = io.BytesIO()
#     near_img.save(stream, format='PNG')  # 画像をPNG形式で保存する場合
#     near_image_base64 = base64.b64encode(stream.getvalue()).decode('utf-8')

#     return {"image": image_base64, 'nearImage': near_image_base64, 'nearNum': nearnum}
@app.get("/return_z")
def Hello():
    z = np.load('parameter/middle.npy')
    width = 500
    height = 500
    z[:, 1] = z[:, 1] * (-1)
    kari_x = (z[:, 0] - np.min(z)) / (np.max(z) - np.min(z)) * width
    kari_y = (z[:, 1] - np.min(z)) / (np.max(z) - np.min(z)) * height

    z[:, 0] = kari_x
    z[:, 1] = kari_y

    z = z.tolist()
    return {"z": z}

@app.get("/return_labels")
def Haello():
    labels = np.load('parameter/labels.npy')
    labels = labels.tolist()
    return {"labels": labels}

@app.get("/return_x")
def Haello():
    data_num = 1000
    x = np.zeros((data_num, 32 * 32))
    for i in range(data_num):
        x[i] = np.array(Image.open('datasets/real_images/{}.png'.format(i)).convert('L')).reshape(-1)
        x[i] = x[i] / 255
    x = x.tolist()
    return {'x': x}
