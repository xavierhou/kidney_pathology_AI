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