import torch
from torch.autograd import Variable
import numpy as np
import utils


def main(args):
    inputfile = args.input
    checkpoint = args.checkpoint
    top_k = args.top_k
    category_names = args.category_names
    gpu = args.gpu
    print(args)

    # load checkpoint
    model, optimizer = utils.loadCheckPoint(checkpoint)

    # get tensor image
    image = torch.from_numpy(utils.processImage(inputfile))
    image.unsqueeze_(0) # https://discuss.pytorch.org/t/expected-stride-to-be-a-single-integer-value-or-a-list/17612/4
    image = Variable(image)

    # cuda is avaliable
    cuda = utils.isCudaAvaliable(gpu)
    if cuda:
        model.cuda()
        image = image.cuda()
    else:
        model.cpu()

    # predict image
    model.eval()
    with torch.no_grad():
        output = model.forward(image.float())
        ps = torch.exp(output).data
    
    probs, classes =  ps.topk(top_k)
    
    if cuda:
        probs = probs.cpu().numpy()
        classes = classes.cpu().numpy()
    else:
        probs = probs.numpy()
        classes = classes.numpy()
    
    classes = np.vectorize(model.idx_to_class.get)(classes)
    
    # class name mapping
    if category_names:
        category_names = utils.getJSONFile(category_names)
        classes = np.vectorize(category_names.get)(classes)
    
    classes = classes[0]
    probs = probs[0]
    
    probs_max_index = np.argmax(probs)
    np.set_printoptions(suppress=True)
    print(classes[probs_max_index], probs[probs_max_index])
    return classes[probs_max_index], probs[probs_max_index]


if __name__ == "__main__":
    args = utils.initPredictParse()
    main(args)
