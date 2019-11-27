import torch
import urllib
import numpy as np
from PIL import Image
from torchvision import transforms
import cv2

def create_model():
    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch',
                        'unet', in_channels=3, out_channels=1,
                        init_features=32, pretrained=True)
    return model
    pass

def download_sample_image():
    url, filename = ("https://github.com/mateuszbuda/brain-segmentation-pytorch/raw/master/assets/TCGA_CS_4944.png", "TCGA_CS_4944.png")
    try: urllib.URLopener().retrieve(url, filename)
    except: urllib.request.urlretrieve(url, filename)
    return filename
    pass

if __name__ == "__main__":
    model = create_model()

    filename = download_sample_image()
    input_image = Image.open(filename)
    m, s = np.mean(input_image, axis=(0, 1)), np.std(input_image, axis=(0, 1))
    preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=m, std=s),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model = model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)
    output = torch.round(output[0])
    print(output)
    output = output.to('cpu').numpy().astype(int)
    output = output.transpose(1, 2, 0)
    cv2.imwrite('output/mask.jpg', output * 255)
    pass