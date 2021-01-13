import matplotlib.image as mpimg
import torchvision.transforms as transforms
from PIL import Image
import torch

#imageSize = 224
#loader = transforms.Compose([transforms.Resize((imageSize, imageSize)), transforms.ToTensor()])
device = torch.device("cuda" if torch.cuda.is_available else "cpu")

#The constants for normalizing data. Doesn't seem to affect the data much so we did't use it
#IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
#IMAGENET_STD_NEUTRAL = [1, 1, 1]

#if the image is larger than 500*500, my computer cannot take it so we reduce the size
def setLength(x, curMax=500):
    return min(x, curMax)

#loads both images (content and style) as tensors
#we take in both as input because we need the style photo to be of the same size as the content image
def loadImage(imgPathContent, imgPathStyle):
    imgContent = Image.open(imgPathContent)
    loader = transforms.Compose([transforms.Resize((setLength(imgContent.size[0]), setLength(imgContent.size[1]))), transforms.ToTensor()])
    imgContent = loader(imgContent).unsqueeze(0)
    loader = transforms.Compose([transforms.Resize((setLength(imgContent.shape[2]), setLength(imgContent.shape[3]))), transforms.ToTensor()])
    imgStyle = Image.open(imgPathStyle)
    imgStyle = loader(imgStyle).unsqueeze(0)
    return (imgContent.to(device), imgStyle.to(device))
