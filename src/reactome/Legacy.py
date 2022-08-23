   
def ensemble_learning(k_folds = 3,  epochs=100, n_layers = 4, save=False):
    ms_proteins = pd.read_csv('data/ms/QuantMatrix.csv')['Protein']
    model = BINN(sparse=True,
                 n_layers = n_layers,
                 learning_rate = 0.001, 
                 ms_proteins=ms_proteins,
                 activation='tanh', 
                 scheduler='plateau')
    protein_matrix = pd.read_csv('data/ms/QuantMatrix.csv')
    protein_matrix = fit_protein_matrix_to_network_input(protein_matrix, RN_proteins=model.RN.ms_proteins)
    X,y = generate_data(protein_matrix, 'data/ms', scale =True)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, stratify = y)
    test_dataset = torch.utils.data.TensorDataset(torch.Tensor(X_train), torch.LongTensor(y_train))
    test_dataloader = DataLoader(test_dataset, num_workers = 12, batch_size=8)
    trained_models = k_fold(model, k_folds=k_folds, scale=True, save=save, log_name = 'ensemble', save_prefix='ensemble', epochs=epochs, X=X_train, y=y_train)
    averaged_model = average_models(trained_models, ms_proteins, n_layers, save=save, save_str = 'ensemble_learning_averaged')
    simple_run(averaged_model, dataloader=test_dataloader, fit=False, validate=True, log_name= 'ensemble_averaged', )
    return averaged_model





 
def average_models(trained_models = [], ms_proteins = [], n_layers=4, save=False, save_str=''):
    averaged_model = BINN(sparse=True,
                 n_layers = n_layers,
                 learning_rate = 0.001, 
                 ms_proteins=ms_proteins,
                 activation='tanh', 
                 scheduler='plateau')
    weights = []
    for m in trained_models:
        curr_weights = []
        for l in m.layers:
            if isinstance(l, nn.Linear):
                curr_weights.append(l.weight.detach().numpy())
        weights.append(curr_weights)
    weights = np.asarray(weights, dtype=object)
    weights = weights.mean(axis=0)
    averaged_model.layers.weight = weights
    averaged_model.report_layer_structure()
    if save:
        torch.save(averaged_model, f'models/{save_str}.pth')
    return averaged_model



def plot_performance_of_ensemble(log_dir, ensemble_log_dir):
    k_metrics = get_metrics_for_dir(log_dir)
    ensemble_accuracies = pd.read_csv(ensemble_log_dir)['accuracy'].values
    final_epoch = max(k_metrics['epoch'])
    k_accuracies = k_metrics[k_metrics['epoch'] == final_epoch]['val_acc'].values
   
    fig = plt.figure(figsize=(3,3))
    plt.bar(x=[1,2], height=[np.mean(k_accuracies), np.mean(ensemble_accuracies)], yerr=[np.std(k_accuracies), np.std(ensemble_accuracies)], color=['red','blue'], alpha=0.5, capsize=5)
    plt.ylim([0.5,1.1])
    sns.despine()
    plt.ylabel('Accuracy')
    plt.xticks([1,2], labels=['Individual', 'Ensemble voting'])
    plt.tight_layout()
    plt.savefig('plots/BINN/Accuracies.jpg', dpi=300)
    