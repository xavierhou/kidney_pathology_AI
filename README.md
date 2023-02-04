# kidney_pathology_AI
This is the code for Our KI paper: Artificial intelligence assists identification and pathologic classification of glomerular lesions in patients with diabetic nephropathy, which mainly focusing on glomeruli classification and cell/mesangial area segmentation tasks.

1. Introduction about the glomeruli classification network - efficientnet

    EfficientNets are a family of image classification models, which achieve state-of-the-art accuracy, yet being an order-of-magnitude smaller and faster than previous models. We develop EfficientNets based on AutoML and Compound Scaling. In particular, we first use AutoML Mobile framework to develop a mobile-size baseline network, named as EfficientNet-B0; Then, we use the compound scaling method to scale up this baseline to obtain EfficientNet-B1 to B7.

    1.1 Load a pretrained EfficientNet:

        from efficientnet_pytorch import EfficientNet
        model = EfficientNet.from_pretrained('efficientnet-b0')

    1.2 Example: how to build a pipeline for classification

        Below is a simple, complete example.

        We assume that in your current directory, there is a img.jpg file and a labels_map.txt file (ImageNet class names). These are both included in examples/simple.

        import json
        from PIL import Image
        import torch
        from torchvision import transforms

        from efficientnet_pytorch import EfficientNet
        model = EfficientNet.from_pretrained('efficientnet-b0')

        # Preprocess image
        tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
        img = tfms(Image.open('img.jpg')).unsqueeze(0)
        print(img.shape) # torch.Size([1, 3, 224, 224])

        # Load ImageNet class names
        labels_map = json.load(open('labels_map.txt'))
        labels_map = [labels_map[str(i)] for i in range(1000)]

        # Classify
        model.eval()
        with torch.no_grad():
            outputs = model(img)

        # Print predictions
        print('-----')
        for idx in torch.topk(outputs, k=5).indices.squeeze(0).tolist():
            prob = torch.softmax(outputs, dim=1)[0, idx].item()
            print('{label:<75} ({p:.2f}%)'.format(label=labels_map[idx], p=prob*100))

2. Introduction about the Cell/Mesangial Area Segmentation network - UNet
    we tested U-Net, R2U-Net, Attention U-Net, Attention R2U-Net on Celland Mesangial Area Segmentation, refered to https://github.com/LeeJunHyun/Image_Segmentation implementation. And found that the Attention U-Net is the most suitable network for our specfic task. The readers can refer to the following papers for details. 
    U-Net: Convolutional Networks for Biomedical Image Segmentation

    https://arxiv.org/abs/1505.04597

    Recurrent Residual Convolutional Neural Network based on U-Net (R2U-Net) for Medical Image Segmentation

    https://arxiv.org/abs/1802.06955

    Attention U-Net: Learning Where to Look for the Pancreas

    https://arxiv.org/abs/1804.03999

    Attention R2U-Net : Just integration of two recent advanced works (R2U-Net + Attention U-Net)

3. Introduction about the Glomeruli Segmentation network - VNet
   we implemented various vnet varients to tackling the glomeruli segmentation task. In the vnet folder, you can see 2.5d、3、fine_grained vnet network architecture, and different backbones used for vnet, and many losses used during training the network. The readers can easyily combine these together. We used keras framework to implement these things. 