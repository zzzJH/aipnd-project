
# coding: utf-8

# # 开发 AI 应用
# 
# 未来，AI 算法在日常生活中的应用将越来越广泛。例如，你可能想要在智能手机应用中包含图像分类器。为此，在整个应用架构中，你将使用一个用成百上千个图像训练过的深度学习模型。未来的软件开发很大一部分将是使用这些模型作为应用的常用部分。
# 
# 在此项目中，你将训练一个图像分类器来识别不同的花卉品种。可以想象有这么一款手机应用，当你对着花卉拍摄时，它能够告诉你这朵花的名称。在实际操作中，你会训练此分类器，然后导出它以用在你的应用中。我们将使用[此数据集](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)，其中包含 102 个花卉类别。你可以在下面查看几个示例。 
# 
# <img src='assets/Flowers.png' width=500px>
# 
# 该项目分为多个步骤：
# 
# * 加载和预处理图像数据集
# * 用数据集训练图像分类器
# * 使用训练的分类器预测图像内容
# 
# 我们将指导你完成每一步，你将用 Python 实现这些步骤。
# 
# 完成此项目后，你将拥有一个可以用任何带标签图像的数据集进行训练的应用。你的网络将学习花卉，并成为一个命令行应用。但是，你对新技能的应用取决于你的想象力和构建数据集的精力。例如，想象有一款应用能够拍摄汽车，告诉你汽车的制造商和型号，然后查询关于该汽车的信息。构建你自己的数据集并开发一款新型应用吧。
# 
# 首先，导入你所需的软件包。建议在代码开头导入所有软件包。当你创建此 notebook 时，如果发现你需要导入某个软件包，确保在开头导入该软件包。

# In[1]:


# Imports here
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models

import helper

import os
from PIL import Image
import numpy as np


# ## 加载数据
# 
# 在此项目中，你将使用 `torchvision` 加载数据（[文档](http://pytorch.org/docs/master/torchvision/transforms.html#)）。数据应该和此 notebook 一起包含在内，否则你可以[在此处下载数据](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz)。数据集分成了三部分：训练集、验证集和测试集。对于训练集，你需要变换数据，例如随机缩放、剪裁和翻转。这样有助于网络泛化，并带来更好的效果。你还需要确保将输入数据的大小调整为 224x224 像素，因为预训练的网络需要这么做。
# 
# 验证集和测试集用于衡量模型对尚未见过的数据的预测效果。对此步骤，你不需要进行任何缩放或旋转变换，但是需要将图像剪裁到合适的大小。
# 
# 对于所有三个数据集，你都需要将均值和标准差标准化到网络期望的结果。均值为 `[0.485, 0.456, 0.406]`，标准差为 `[0.229, 0.224, 0.225]`。这样使得每个颜色通道的值位于 -1 到 1 之间，而不是 0 到 1 之间。

# In[2]:


train_dir = 'train'
valid_dir = 'valid'
test_dir = 'test'


# In[3]:


# TODO: Define your transforms for the training, validation, and testing sets
data_transforms = {
    'training': transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
#                                   transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])]),
    'validation': transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])]),
    'testing': transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
}

# TODO: Load the datasets with ImageFolder
image_datasets = {
    'train': datasets.ImageFolder('./flowers/' + train_dir, transform=data_transforms['training']), # 6552
    'valid': datasets.ImageFolder('./flowers/' + valid_dir, transform=data_transforms['validation']), # 818
    'test': datasets.ImageFolder('./flowers/' + test_dir, transform=data_transforms['testing']) # 819
}

# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = {
    'trainloader': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
    'validloader': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32),
    'testloader': torch.utils.data.DataLoader(image_datasets['test'], batch_size=32)
}


# ### 标签映射
# 
# 你还需要加载从类别标签到类别名称的映射。你可以在文件 `cat_to_name.json` 中找到此映射。它是一个 JSON 对象，可以使用 [`json` 模块](https://docs.python.org/2/library/json.html)读取它。这样可以获得一个从整数编码的类别到实际花卉名称的映射字典。

# In[4]:


import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
print(cat_to_name['21'])


# # 构建和训练分类器
# 
# 数据准备好后，就开始构建和训练分类器了。和往常一样，你应该使用 `torchvision.models` 中的某个预训练模型获取图像特征。使用这些特征构建和训练新的前馈分类器。
# 
# 这部分将由你来完成。如果你想与他人讨论这部分，欢迎与你的同学讨论！你还可以在论坛上提问或在工作时间内咨询我们的课程经理和助教导师。
# 
# 请参阅[审阅标准](https://review.udacity.com/#!/rubrics/1663/view)，了解如何成功地完成此部分。你需要执行以下操作：
# 
# * 加载[预训练的网络](http://pytorch.org/docs/master/torchvision/models.html)（如果你需要一个起点，推荐使用 VGG 网络，它简单易用）
# * 使用 ReLU 激活函数和丢弃定义新的未训练前馈网络作为分类器
# * 使用反向传播训练分类器层，并使用预训练的网络获取特征
# * 跟踪验证集的损失和准确率，以确定最佳超参数
# 
# 我们在下面为你留了一个空的单元格，但是你可以使用多个单元格。建议将问题拆分为更小的部分，并单独运行。检查确保每部分都达到预期效果，然后再完成下个部分。你可能会发现，当你实现每部分时，可能需要回去修改之前的代码，这很正常！
# 
# 训练时，确保仅更新前馈网络的权重。如果一切构建正确的话，验证准确率应该能够超过 70%。确保尝试不同的超参数（学习速率、分类器中的单元、周期等），寻找最佳模型。保存这些超参数并用作项目下个部分的默认值。

# In[31]:


# TODO: Build and train your network
model = models.vgg19(pretrained=True)
print(model)


# In[32]:


for param in model.parameters():
    param.requires_grad = False

from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 4096)),
                          ('relu', nn.ReLU()),
                          ('dropout1', nn.Dropout(0.5)),
#                           ('fc2', nn.Linear(4096, 1024)),
#                           ('relu', nn.ReLU()),
#                           ('dropout2', nn.Dropout(0.5)),
                          ('fc3', nn.Linear(4096, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
model.classifier = classifier

print(model)


# In[33]:


criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
print(optimizer)


# In[34]:


epochs = 2
steps = 0
running_loss = 0
print_every = 20

cuda = torch.cuda.is_available()
print('is cuda avaliable', cuda)
if (cuda):
    model.cuda()
else:
    model.cpu()


# In[35]:


for e in range(epochs):
    model.train()
    for images, labels in iter(dataloaders['trainloader']):
        steps += 1
        
        images, labels = Variable(images), Variable(labels)
        optimizer.zero_grad()
        
        if cuda:
            images, labels = images.cuda(), labels.cuda()
            
        outputs = model.forward(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.data[0] 
        
        if steps % print_every == 0:
            model.eval()
            accuracy = 0
            test_loss = 0
            for ii, (images, labels) in enumerate(dataloaders['testloader']):
                images, labels = Variable(images, volatile=True), Variable(labels, volatile=True)

                if cuda:
                    images, labels = images.cuda(), labels.cuda()

                output = model.forward(images)
                test_loss += criterion(output, labels).data[0]

                ps = torch.exp(output).data
                equality = (labels.data == ps.max(1)[1])
                accuracy += equality.type_as(torch.FloatTensor()).mean()
                
            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Test Loss: {:.3f}.. ".format(test_loss/len(dataloaders['testloader'])),
                  "Test Accuracy: {:.3f}".format(accuracy/len(dataloaders['testloader'])))

            running_loss = 0
            model.train()
print('train finish')


# ## 测试网络
# 
# 建议使用网络在训练或验证过程中从未见过的测试数据测试训练的网络。这样，可以很好地判断模型预测全新图像的效果。用网络预测测试图像，并测量准确率，就像验证过程一样。如果模型训练良好的话，你应该能够达到大约 70% 的准确率。

# In[18]:


# TODO: Do validation on the test set
model.eval()

accuracy = 0
test_loss = 0
for ii, (images, labels) in enumerate(dataloaders['validloader']):
    images, labels = Variable(images, volatile=True), Variable(labels, volatile=True)
    
    if cuda:
        images, labels = images.cuda(), labels.cuda()

    output = model.forward(images)
    test_loss += criterion(output, labels).data[0]
    
    ps = torch.exp(output).data
    equality = (labels.data == ps.max(1)[1])
    accuracy += equality.type_as(torch.FloatTensor()).mean()

    print("Test Loss: {:.3f}.. ".format(test_loss/len(dataloaders['validloader'])),
      "Test Accuracy: {:.3f}".format(accuracy/len(dataloaders['validloader'])))


# ## 保存检查点
# 
# 训练好网络后，保存模型，以便稍后加载它并进行预测。你可能还需要保存其他内容，例如从类别到索引的映射，索引是从某个图像数据集中获取的：`image_datasets['train'].class_to_idx`。你可以将其作为属性附加到模型上，这样稍后推理会更轻松。
# 
# 
# 注意，稍后你需要完全重新构建模型，以便用模型进行推理。确保在检查点中包含你所需的任何信息。如果你想加载模型并继续训练，则需要保存周期数量和优化器状态 `optimizer.state_dict`。你可能需要在下面的下个部分使用训练的模型，因此建议立即保存它。
# 

# In[11]:


# TODO: Save the checkpoint 
checkpoint = {
    'epochs': epochs,
    'epochs': epochs,
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'class_to_idx': image_datasets['train'].class_to_idx
}
# sd['classifier.0.weight'] = sd['classifier.1.weight']
# sd['classifier.0.bias'] = sd['classifier.1.bias']
# del sd['classifier.1.weight']
# del sd['classifier.1.bias']

torch.save(checkpoint, 'checkpoint.pth.tar')
print(checkpoint.keys())


# ## 加载检查点
# 
# 此刻，建议写一个可以加载检查点并重新构建模型的函数。这样的话，你可以回到此项目并继续完善它，而不用重新训练网络。

# In[5]:


# TODO: Write a function that loads a checkpoint and rebuilds the model
from collections import OrderedDict

def load_checkpoint(model, optimizer, filepath):
    if os.path.isfile(filepath):
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.idx_to_class = {v: k for k, v in checkpoint['class_to_idx'].items()}
        print('load success')
        return model, optimizer
    else:
        print('file is not exist')

model_recover = models.vgg19(pretrained=True)
for param in model_recover.parameters():
    param.requires_grad = False
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 4096)),
                          ('relu', nn.ReLU()),
                          ('dropout1', nn.Dropout(0.5)),
#                           ('fc2', nn.Linear(4096, 1024)),
#                           ('relu', nn.ReLU()),
#                           ('dropout2', nn.Dropout(0.5)),
                          ('fc3', nn.Linear(4096, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
model_recover.classifier = classifier
optimizer_recover = optim.Adam(model_recover.classifier.parameters(), lr=0.001)

model_recover, optimizer_recover = load_checkpoint(model_recover, optimizer_recover, 'checkpoint.pth.tar')
model = model_recover
optimizer = optimizer_recover
# print(model_recover, optimizer_recover)
# print(model_recover.idx_to_class)
print(model, optimizer)
print(model.idx_to_class)


# # 类别推理
# 
# 现在，你需要写一个使用训练的网络进行推理的函数。即你将向网络中传入一个图像，并预测图像中的花卉类别。写一个叫做 `predict` 的函数，该函数会接受图像和模型，然后返回概率在前 $K$ 的类别及其概率。应该如下所示：

# In[ ]:


probs, classes = predict(image_path, model)
print(probs)
print(classes)
> [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
> ['70', '3', '45', '62', '55']


# 首先，你需要处理输入图像，使其可以用于你的网络。
# 
# ## 图像处理
# 
# 你需要使用 `PIL` 加载图像（[文档](https://pillow.readthedocs.io/en/latest/reference/Image.html)）。建议写一个函数来处理图像，使图像可以作为模型的输入。该函数应该按照训练的相同方式处理图像。
# 
# 首先，调整图像大小，使最小的边为 256 像素，并保持宽高比。为此，可以使用 [`thumbnail`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) 或 [`resize`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) 方法。然后，你需要从图像的中心裁剪出 224x224 的部分。
# 
# 图像的颜色通道通常编码为整数 0-255，但是该模型要求值为浮点数 0-1。你需要变换值。使用 Numpy 数组最简单，你可以从 PIL 图像中获取，例如 `np_image = np.array(pil_image)`。
# 
# 和之前一样，网络要求图像按照特定的方式标准化。均值应标准化为 `[0.485, 0.456, 0.406]`，标准差应标准化为 `[0.229, 0.224, 0.225]`。你需要用每个颜色通道减去均值，然后除以标准差。
# 
# 最后，PyTorch 要求颜色通道为第一个维度，但是在 PIL 图像和 Numpy 数组中是第三个维度。你可以使用 [`ndarray.transpose`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.transpose.html)对维度重新排序。颜色通道必须是第一个维度，并保持另外两个维度的顺序。

# In[6]:


mean = np.asarray([0.485, 0.456, 0.406])
std = np.asarray([0.229, 0.224, 0.225])

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    if os.path.isfile(image) != True:
        print('file is not exist')
        return
    
    im = Image.open(image)
    im = im.resize((256, 256))
    im = im.crop((16, 16, 240, 240))
    
    np_image = np.array(im) / 256
    
    for i in range(3):
        np_image[..., i] -= mean[..., i]
        np_image[..., i] /= std[..., i]
    
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image

print(process_image('./flowers/valid/20/image_04903.jpg'))


# 要检查你的项目，可以使用以下函数来转换 PyTorch 张量并将其显示在  notebook 中。如果 `process_image` 函数可行，用该函数运行输出应该会返回原始图像（但是剪裁掉的部分除外）。

# In[7]:


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax
imshow(torch.from_numpy((process_image('./flowers/valid/20/image_04903.jpg'))))


# ## 类别预测
# 
# 可以获得格式正确的图像后 
# 
# 要获得前 $K$ 个值，在张量中使用 [`x.topk(k)`](http://pytorch.org/docs/master/torch.html#torch.topk)。该函数会返回前 `k` 个概率和对应的类别索引。你需要使用  `class_to_idx`（希望你将其添加到了模型中）将这些索引转换为实际类别标签，或者从用来加载数据的 `ImageFolder`（[请参阅此处](#Save-the-checkpoint)）进行转换。确保颠倒字典
# 
# 同样，此方法应该接受图像路径和模型检查点，并返回概率和类别。

# In[ ]:


probs, classes = predict(image_path, model)
print(probs)
print(classes)
> [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
> ['70', '3', '45', '62', '55']


# In[15]:


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    cuda = torch.cuda.is_available()
    
    image = torch.from_numpy(process_image(image_path))
    image.unsqueeze_(0) # https://discuss.pytorch.org/t/expected-stride-to-be-a-single-integer-value-or-a-list/17612/4
    image = Variable(image)
    
    if cuda:
        model.cuda()
        image = image.cuda()
    else:
        model.cpu()
        
    model.eval()
    with torch.no_grad():
        output = model.forward(image.float())
        ps = torch.exp(output).data
    
    probs, classes =  ps.topk(topk)
    
    if cuda:
        probs = probs.cpu().numpy()
        classes = classes.cpu().numpy()
    else:
        probs = probs.numpy()
        classes = classes.numpy()
    
    classes = np.vectorize(model.idx_to_class.get)(classes)
    classes = np.vectorize(cat_to_name.get)(classes)
    
    return probs[0], classes[0]

probs, classes = predict('./flowers/valid/10/image_07094.jpg', model)

np.set_printoptions(suppress=True)
print(probs)
print(classes)


# ## 检查运行状况
# 
# 你已经可以使用训练的模型做出预测，现在检查模型的性能如何。即使测试准确率很高，始终有必要检查是否存在明显的错误。使用 `matplotlib` 将前 5 个类别的概率以及输入图像绘制为条形图，应该如下所示：
# 
# <img src='assets/inference_example.png' width=300px>
# 
# 你可以使用 `cat_to_name.json` 文件（应该之前已经在 notebook 中加载该文件）将类别整数编码转换为实际花卉名称。要将 PyTorch 张量显示为图像，请使用定义如下的 `imshow` 函数。

# In[16]:


# TODO: Display an image along with the top 5 classes
image_path = './flowers/valid/100/image_07895.jpg'

imshow(torch.from_numpy((process_image(image_path))))
probs, classes = predict(image_path, model)

plt.figure()

print(classes_cn, probs)
plt.barh(range(len(classes)), probs, align = 'center', color='steelblue', alpha = 0.8)
plt.yticks(range(len(classes)), classes)

