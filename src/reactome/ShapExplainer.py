import torch
import shap
import torch.nn as nn
from BINN import BINN
from DataLoaders import generate_protein_matrix, generate_data, fit_protein_matrix_to_network_input
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

"""

Good documentation on SHAP: https://christophm.github.io/interpretable-ml-book/shap.html 

"""


checkpoint_file_path = 'lightning_logs/version_16/checkpoints/epoch=99-step=200.ckpt'
model = BINN.load_from_checkpoint(checkpoint_file_path)
model.report_layer_structure()


feature_names = model.column_names[0]

# load data
protein_matrix = generate_protein_matrix('data/ms')
protein_matrix = fit_protein_matrix_to_network_input(protein_matrix, RN_proteins = model.RN.ms_proteins)
X,y = generate_data(protein_matrix, 'data/ms', scale = True)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3)
X_test = torch.Tensor(X_test)
y_test = torch.LongTensor(y_test)
X_train = torch.Tensor(X_train)
y_train = torch.Tensor(y_train)

background = X_train
test_data = X_test


def shap_test(model, background, test):
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(test_data)

    shap.summary_plot(shap_values[0], test_data, feature_names = feature_names)
    plt.savefig('plots/shap/test.jpg')


def shap_for_layers(model, background, test_data):
    
    """
    I guess we'd want to take the sum of chap score on the output layer.
    """
    i = 0
    for layer in model.layers:
        if isinstance(layer, nn.Linear):
            feature_names = model.column_names[i]
            try:
                output_names = model.column_names[i+1]
            except:
                output_names = ["0","1"]
            explainer = shap.DeepExplainer(layer, background[0:20])
            shap_values = explainer.shap_values(test_data[0:5])
            shap.summary_plot(shap_values, test_data[0:5], feature_names = feature_names, class_names = output_names)
            plt.savefig(f'plots/shap/test_layer_{i}.jpg')
            plt.clf()
            i += 1
            background = layer(background)
            test_data = layer(test_data)
            print(f"Shap for layer {i}")
        elif isinstance(layer, nn.Tanh) or isinstance(layer, nn.ReLU):
            background = layer(background)
            test_data = layer(test_data)

shap_for_layers(model, background, test_data)
"""

    # load the model
    model = models.vgg16(pretrained=True).eval()

    X,y = shap.datasets.imagenet50()

    X /= 255

    to_explain = X[[39, 41]]

    # load the ImageNet class names
    url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
    fname = shap.datasets.cache(url)
    with open(fname) as f:
        class_names = json.load(f)

                                                    seventh layer
    e = shap.GradientExplainer((model, model.features[7]), normalize(X))
    shap_testues,indexes = e.shap_testues(normalize(to_explain), ranked_outputs=2, nsamples=200)

    # get the names for the classes
    index_names = np.vectorize(lambda x: class_names[str(x)][1])(indexes)

    # plot the explanations
    shap_testues = [np.swapaxes(np.swapaxes(s, 2, 3), 1, -1) for s in shap_testues]

    shap.image_plot(shap_testues, to_explain, index_names)

"""