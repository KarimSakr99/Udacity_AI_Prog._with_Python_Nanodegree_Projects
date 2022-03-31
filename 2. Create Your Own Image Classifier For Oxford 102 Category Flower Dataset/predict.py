import numpy as np
import torch
from datafunc import process_image, imshow, get_cat_names
from input_arg import test_input_args
from model_func import load_build_from_checkpoint
import matplotlib.pyplot as plt


def predict(image_path, model, topk, use_gpu):

    device = 'cpu'
    if use_gpu and torch.cuda.is_available():
        device = 'cuda'
    elif use_gpu:
        print('CUDA not available, Will work using CPU!!')

    model.to(device)
    model.eval()

    image = process_image(image_path).view(1, 3, 224, 224).float()
    image = image.to(device)

    with torch.no_grad():
        log_out = model(image)
        out = torch.exp(log_out)
        props, classes = out.topk(topk, dim=1)

    props, classes = props.to('cpu'), classes.to('cpu')
    classes = [list(model.class_to_idx.keys())[clas]
               for clas in classes.numpy()[0]]

    return props.numpy()[0], classes


def predict_show(image_path, model, topk, use_gpu, category_names):
    props, classes = predict(image_path, model, topk, use_gpu)

    print('\n\nPrediction result for the image:\n')
    for clas, prop in zip(classes, props):
        print(
            f'class {category_names[clas] if category_names else clas} with a probability of {prop:0.3f}')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 10))
    ax1.axis('off')
    imshow(process_image(image_path), ax=ax1)

    ax2.invert_yaxis()
    ax2.set(xlim=(0, 1), xticks=np.linspace(0, 1, 11), xlabel='Probability')

    if category_names:
        ax1.set_title(category_names[classes[0]].capitalize())
        ax2.barh([category_names[i] for i in classes], props)
    else:
        ax2.barh(classes, props)
    plt.show()


inputs = test_input_args()

model = load_build_from_checkpoint(inputs.checkpoint)[0]

cat_to_name = get_cat_names(inputs.category_names)

predict_show(inputs.image, model, inputs.top_k,
             inputs.use_gpu, cat_to_name)
