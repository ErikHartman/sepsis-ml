import torch
import shap
import numpy as np
from BINN import BINN
from DataLoaders import generate_protein_matrix, generate_data, fit_protein_matrix_to_network_input


checkpoint_file_path = 'lightning_logs/version_13/checkpoints/epoch=99-step=200.ckpt'
model = BINN.load_from_checkpoint(checkpoint_file_path)
model.report_layer_structure()

# load data
protein_matrix = generate_protein_matrix('data/ms')
protein_matrix = fit_protein_matrix_to_network_input(protein_matrix, RN_proteins = model.RN.ms_proteins)
X,y = generate_data(protein_matrix, 'data/ms', scale = True)



layer = 2
to_explain = X[1:3]
e = shap.GradientExplainer(model, X)
shap_values,indexes = e.shap_values(to_explain, ranked_outputs=2, nsamples=200)


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
    shap_values,indexes = e.shap_values(normalize(to_explain), ranked_outputs=2, nsamples=200)

    # get the names for the classes
    index_names = np.vectorize(lambda x: class_names[str(x)][1])(indexes)

    # plot the explanations
    shap_values = [np.swapaxes(np.swapaxes(s, 2, 3), 1, -1) for s in shap_values]

    shap.image_plot(shap_values, to_explain, index_names)

"""