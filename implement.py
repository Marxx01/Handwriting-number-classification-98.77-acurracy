import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import pygame
import sys
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

class canNet(nn.Module):
    def __init__(self):
        super(canNet, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.drop = nn.Dropout(p = 0.5)

        self.act = nn.LeakyReLU()
        
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 3, stride = 1, padding = 1, bias = True)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1, padding = 1, bias = True)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1, bias = True)

        self.linear1 = nn.Linear(in_features = 1152, out_features = 512, bias = True)
        self.linear2 = nn.Linear(in_features = 512, out_features = 256, bias = True)
        self.linear3 = nn.Linear(in_features = 256, out_features = 10, bias = True)

        self.batch1 = nn.BatchNorm2d(32)
        self.batch2 = nn.BatchNorm2d(64)
        self.batch3 = nn.BatchNorm2d(128)

    def forward(self, x):
        out = self.conv1(x)
        out = self.batch1(out)
        out = self.act(out)
        out = self.pool(out)
        
        out = self.conv2(out)
        out = self.batch2(out)
        out = self.act(out)
        out = self.pool(out)

        out = self.conv3(out)
        out = self.batch3(out)
        out = self.act(out)
        out = self.pool(out)

        out = out.view(-1, 1152)
        
        out = self.linear1(out)
        out = self.act(out)

        out = self.linear2(out)
        out = self.act(out)

        out = self.linear3(out)
        out = F.softmax(out, dim = -1)

        return out
    
model = canNet()

model = torch.load('./model.9897.pth', map_location=torch.device('cpu'))

trans = transforms.Grayscale(num_output_channels=1)
resize = transforms.Resize((28, 28))


def draw_and_save_image():
    pygame.init()

    
    width, height = 280, 320
    drawing_height = 280  
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Dibuja un número")

    black = (0, 0, 0)
    white = (255, 255, 255)
    gray = (200, 200, 200)


    screen.fill(black)

    def draw_buttons():
        pygame.draw.rect(screen, gray, (10, 290, 80, 20)) 
        pygame.draw.rect(screen, gray, (190, 290, 80, 20)) 
        font = pygame.font.SysFont(None, 24)
        save_text = font.render('Valorar', True, black)
        clear_text = font.render('Borrar', True, black)
        screen.blit(save_text, (20, 290))
        screen.blit(clear_text, (200, 290))

    draw_buttons()
    drawing = False
    last_pos = None

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    x, y = event.pos
                    if 10 <= x <= 90 and 290 <= y <= 310: 
                        # Crear una nueva superficie que solo contenga el área de dibujo
                        drawing_surface = screen.subsurface((0, 0, width, drawing_height)).copy()

                        # Convertir la superficie a un array de NumPy
                        drawing_array = pygame.surfarray.array3d(drawing_surface)

                        #print(drawing_array.shape)
                        drawing_array = np.moveaxis(drawing_array, 2, 0)
                        drawing_array = np.moveaxis(drawing_array, 1, 2)   # Cambiar de (anchura, altura, canales) a (altura, anchura, canales)
                        #print(drawing_array.shape)

                        drawing_tensor = torch.tensor(drawing_array, dtype=torch.float32)# / 255.0  # Normalizar a [0, 1]
                        drawing_tensor = trans(drawing_tensor)
                        drawing_tensor = resize(drawing_tensor)
                        #print(drawing_tensor.shape)

                        # Convertir el array a un tensor de PyTorch
                        #drawing_array = np.moveaxis(drawing_array, 0, -1)  # Cambiar de (anchura, altura, canales) a (altura, anchura, canales)
                        #drawing_tensor = torch.tensor(drawing_array, dtype=torch.float32) / 255.0  # Normalizar a [0, 1]

                        #print("Tensor generado:")
                        #print(drawing_tensor)

                        return drawing_tensor
                    elif 190 <= x <= 270 and 290 <= y <= 310:
            
                        screen.fill(black)
                        draw_buttons()
                    else:
                        drawing = True
                        last_pos = event.pos
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1: 
                    drawing = False
                    last_pos = None
            elif event.type == pygame.MOUSEMOTION:
                if drawing:
                    mouseX, mouseY = event.pos
                    if mouseY < drawing_height: 
                        if last_pos is not None:
                            pygame.draw.circle(screen, white, last_pos, 10)
                        last_pos = (mouseX, mouseY)

        pygame.display.update()
while True:
    image_tensor = draw_and_save_image()

    result = model(image_tensor.unsqueeze(0))
    print(result.argmax().item())



