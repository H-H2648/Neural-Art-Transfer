import torch
from VGG import VGGModel
from ImageLoader import loadImage
import torch.optim as optim
from torchvision.utils import save_image


device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
VModel = VGGModel().to(device)

contentImgDirectory = r'C:\Users\hhong\Documents\Neural Art Transfer\content-images\cat2.jpg'
styleImgDirectory = r'C:\Users\hhong\Documents\Neural Art Transfer\style-images\smile.jpg'
contentImg , styleImg = loadImage(contentImgDirectory, styleImgDirectory)
currentImg = contentImg.clone().requires_grad_(True)
optimizer = optim.Adam([currentImg], lr= 0.001)
alpha = 100000
beta = 30000
gamma=1

#our loss function is alpha*contentLoss + beta*styleLoss + gamma*totalVariance

#totalVariance refers to the overall difference in pixel value (we want the image to be rather smooth)
for step in range(3000):
    currentImgFeatures = VModel(currentImg)
    contentImgFeatures = VModel(contentImg)
    styleImgFeatures = VModel(styleImg)
    #for content loss, just like the paper, we only look at conv4_1
    contentLoss = torch.nn.MSELoss(reduction='mean')(currentImgFeatures[3], contentImgFeatures[3])
    styleLoss = 0
    for currentFeatures, contentFeatures, styleFeatures in zip(currentImgFeatures, contentImgFeatures, styleImgFeatures):
        #Gram Matrix
        batchSize, channel, height, width = currentFeatures.shape
        G = currentFeatures.view(channel, height*width).mm(currentFeatures.view(channel, height*width).t())
        A = styleFeatures.view(channel, height*width).mm(styleFeatures.view(channel, height*width).t())
        #we look at the difference between the two gram matrices for all layers
        styleLoss += torch.nn.MSELoss(reduction='mean')(G, A)
    #distance in pixel values of neighbouring pixels
    variationLoss =  torch.sum(torch.abs(currentImg[:, :, :, :-1] - currentImg[:, :, :, 1:])) + torch.sum(torch.abs(currentImg[:, :, :-1, :] - currentImg[:, :, 1:, :]))
    totalLoss = alpha*contentLoss + beta*styleLoss + gamma*variationLoss
    optimizer.zero_grad()
    totalLoss.backward()
    optimizer.step()
    if step % 200 == 0:
        print(totalLoss)
        save_image(currentImg, "C:\\Users\\hhong\\Documents\\Neural Art Transfer\\Result8\\currentImg_{0}.jpg".format(int(step/200)))
save_image(currentImg, r"C:\Users\hhong\Documents\Neural Art Transfer\Result8\currentImg_FINAL.jpg")
    