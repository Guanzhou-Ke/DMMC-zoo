import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from data import NoisyMnistDataset
from linear_cca import LinearCCA
from models import DCCA, DCCAE
from objectives import CCALoss
from utils import load_noisy_mnist, svm_classify


def run(data_dir, 
        batch_size, 
        checkpoint_dir, 
        early_stop, 
        gpu):
    
    # Load dataset
    train_set, valid_set, test_set = load_noisy_mnist(data_dir)
    
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(valid_set, batch_size=batch_size, num_workers=4)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, num_workers=4)
    
    if gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    epoches = 100
    seed = 42
    outdim = 10
    lam = 1
    reg_par = 1e-5
    learning_rate = 1e-3
    
    use_all_singular_values = False
    view1_layers = [784, 1024, 1024, 1024, outdim]
    view2_layers = [784, 1024, 1024, 1024, outdim]
    
    encoder1_layers = [784, 1024, 1024, 1024, outdim]
    encoder2_layers = [784, 1024, 1024, 1024, outdim]

    decoder1_layers = [outdim, 1024, 1024, 1024, 784]
    decoder2_layers = [outdim, 1024, 1024, 1024, 784]

    
    
    models_list = [
        'Linear_CCA',
        'DCCA',
        'DCCAE'
    ]
    
    for model_type in models_list:
        pl.seed_everything(seed)
        print(f"Training {model_type}...")
        classification_data = []
        if model_type == 'Linear_CCA':
            linear_cca = LinearCCA(outdim)
            linear_cca.fit(train_set.view1.numpy(), train_set.view2.numpy())
            classification_data.append(linear_cca.test(train_set.view1.numpy(), train_set.view2.numpy())[0])
            classification_data.append(linear_cca.test(valid_set.view1.numpy(), valid_set.view2.numpy())[0])
            classification_data.append(linear_cca.test(test_set.view1.numpy(), test_set.view2.numpy())[0])
        else:
            if model_type == 'DCCA':
                model = DCCA(view1_layers, view2_layers, 
                             outdim=outdim, use_all_singular_values=use_all_singular_values,
                             learning_rate=learning_rate,
                             device=device)
            else:
                model = DCCAE(encoder1_layers=encoder1_layers, encoder2_layers=encoder2_layers,
                              decoder1_layers=decoder1_layers, decoder2_layers=decoder2_layers,
                              outdim=outdim, lam=lam, reg_par=reg_par, use_all_singular_values=use_all_singular_values,
                              learning_rate=learning_rate,
                              device=device)
        
            logger = TensorBoardLogger('../learning_logs')
            callbacks = []
            checkpoint_callback = ModelCheckpoint(monitor='valid_loss',
                                                  dirpath=f'{checkpoint_dir}{model_type}',
                                                  filename='{epoch:02d}-{valid_loss:.2f}',
                                                  save_top_k=1,
                                                  mode='min')
            callbacks.append(checkpoint_callback)
            if early_stop:
                early_stop_callback = EarlyStopping(monitor='valid_loss', patience=10)
                callbacks.append(early_stop_callback)
            else:
                pass
            trainer = pl.Trainer(gpus=(1 if gpu else 0), 
                                 max_epochs=epoches, 
                                 callbacks=callbacks, 
                                 logger=logger)
            trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)
            
            model.eval()
            model.cpu()
            
            # Start test.
            linear_cca = LinearCCA(outdim)
            pred_out = model(train_set.view1, train_set.view2)
            linear_cca.fit(pred_out[0].detach().numpy(), pred_out[1].detach().numpy())
            for ds in (train_set, valid_set, test_set):
                pred_out = model(ds.view1, ds.view2)
                fea_lc = linear_cca.test(pred_out[0].detach().numpy(), pred_out[1].detach().numpy())
                classification_data.append( fea_lc[0] )
        
        print(f"Training {model_type} done! Start test process with linear SVM.")
        svm_classify([
                (classification_data[0], train_set.target.numpy()),
                (classification_data[1], valid_set.target.numpy()),
                (classification_data[2], test_set.target.numpy())
                ], C=0.01)
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='DCCAE experiments arguments')
    parser.add_argument('--data_dir', default='../data')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size for training')
    parser.add_argument('--checkpoint_dir', default='../models_store/', help='Checkpoints path.')
    parser.add_argument('--early_stop', action='store_true', help='Enable early stopping schedule.')
    parser.add_argument('--gpu', action='store_true', help='use GPU for training')
    opt = parser.parse_args()

    run(opt.data_dir, opt.batch_size, opt.checkpoint_dir, opt.early_stop, opt.gpu)