import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from mmd import mix_rbf_wmmd2, mix_rbf_mmd2

import params



def train_ae(encoder, decoder, data_loader_s, data_loader_t):

    num_iter = max(len(data_loader_s), len(data_loader_t))
    
    encoder.train()
    decoder.train()
    
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=params.LEARNING_RATE_I,
    )
    loss_mse_func = nn.MSELoss()
    
    for epoch in range(params.NUM_EPOCH_I):
        for iteration in range(num_iter):

            if (iteration % len(data_loader_s) == 0):
                iter_dataloader_s = iter(data_loader_s)
            
            if (iteration % len(data_loader_t) == 0):
                iter_dataloader_t = iter(data_loader_t)
                
            x_s, _ = iter_dataloader_s.next()
            x_t, = iter_dataloader_t.next()

            if x_s.is_sparse:
                x_s = x_s.to_dense()
            if x_t.is_sparse:
                x_t = x_t.to_dense()
            
            optimizer.zero_grad()
            
            feature_s = encoder(x_s)
            reconstruct_s = decoder(feature_s)

            feature_t = encoder(x_t)
            reconstruct_t = decoder(feature_t)
            
            loss = loss_mse_func(reconstruct_s, x_s) + loss_mse_func(reconstruct_t, x_t)
            
            loss.backward()
            optimizer.step()
            
            # if ((step+1) % params.INTERVAL_STEP_I == 0):
            #     print("Training autoencoder Epoch [{}/{}] Iteration [{}/{}]: loss={}"
            #         .format(epoch + 1,
            #                 params.NUM_EPOCH_I,
            #                 iteration,
            #                 num_iter,
            #                 loss.detach().item()))
            
    return encoder, decoder



def train_s(encoder, decoder, classifier, data_loader_s, data_loader_t):

    encoder.train()
    decoder.train()
    classifier.train()

    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()) + list(classifier.parameters()),
        lr = params.LEARNING_RATE_S,
    )
    loss_crossentropy_func = nn.CrossEntropyLoss()
    loss_mse_func = nn.MSELoss()
    
    for epoch in range(params.NUM_EPOCH_S):
        for step, (x_s, y_s) in enumerate(data_loader_s):

            if (step % len(data_loader_t) == 0):
                iter_dataloader_t = iter(data_loader_t)

            x_t, = iter_dataloader_t.next()

            if x_s.is_sparse:
                x_s = x_s.to_dense()
            if x_t.is_sparse:
                x_t = x_t.to_dense()

            optimizer.zero_grad()

            feature_s = encoder(x_s)
            reconstruct_s = decoder(feature_s)
            prediction_s = classifier(feature_s)

            feature_t = encoder(x_t)
            reconstruct_t = decoder(feature_t)

            loss_cls = loss_crossentropy_func(prediction_s, y_s)
            loss_rec = loss_mse_func(reconstruct_s, x_s) + loss_mse_func(reconstruct_t, x_t)

            loss = params.LAMBDA_S_1 * loss_cls + params.LAMBDA_S_2 * loss_rec
                        
            loss.backward()
            optimizer.step()

            label_pred = prediction_s.detach().max(1)[1]
            acc = (label_pred == y_s).float().mean().item()
            

            # if ((step+1) % params.INTERVAL_STEP_S == 0):
            #     print("Training_S Epoch [{}/{}] Step [{}/{}]: loss={} loss_cls={} loss_rec={} acc={}"
            #         .format(epoch + 1,
            #                 params.NUM_EPOCH_S,
            #                 step,
            #                 len(data_loader),
            #                 loss.detach().item(),
            #                 loss_cls.detach().item(),
            #                 loss_rec.detach().item(),
            #                 acc))
        
    return encoder, decoder, classifier




def train_t(encoder_t, data_loader_t, data_loader_s, weight_t):
    
    encoder_t.train()
    optimizer = optim.Adam(
        encoder_t.parameters(),
        lr=params.LEARNING_RATE_T,
    )
    weight_t = torch.FloatTensor(weight_t)
    for iteration in range(params.NUM_ITERATION_T):
        
        optimizer.zero_grad()
        
        if (iteration % len(data_loader_s) == 0):
            iter_dataloader_s = iter(data_loader_s)
        
        if (iteration % len(data_loader_t) == 0):
            iter_dataloader_t = iter(data_loader_t)
            
        feature_s, y_s = iter_dataloader_s.next()
        x_t, pred_t = iter_dataloader_t.next()
        
        feature_t = encoder_t(x_t)
        
        loss_mmd_class = torch.FloatTensor([0])
        loss_transfer = torch.FloatTensor([0])
        loss_intra = torch.FloatTensor([0])
        if params.CUDA:
            loss_mmd_class = loss_mmd_class.cuda()
            loss_intra = loss_intra.cuda()
            loss_transfer = loss_transfer.cuda()
        
        for i in torch.unique(pred_t).cpu().numpy().tolist():
            t_ix = torch.where(pred_t == i)[0]
            if (weight_t[t_ix] != 0).any():
                feature_t_i = feature_t[t_ix]
                feature_t_i_mean = torch.mean(feature_t_i, dim=0, keepdim=True)
                loss_intra -= torch.mm(F.normalize(feature_t_i), F.normalize(feature_t_i_mean).t()).mean()

                s_ix = torch.where(y_s == i)[0]
                if (len(s_ix) > 0):
                    w_s = torch.ones(len(s_ix))
                    if params.CUDA:
                        w_s = w_s.cuda()
                    loss_transfer += mix_rbf_wmmd2(feature_s[s_ix], feature_t_i, w_s, weight_t[t_ix], params.SIGMA_LIST)
            
        
        loss_intra = loss_intra / len(torch.unique(pred_t))
        loss_mmd_class = loss_transfer / len(torch.unique(pred_t))
        
        loss_mmd_general = mix_rbf_mmd2(feature_t, feature_s, params.SIGMA_LIST)
        
        loss = params.LAMBDA_T_1 * loss_mmd_class + params.LAMBDA_T_2 * loss_mmd_general + params.LAMBDA_T_3 * loss_intra
        
        loss.backward()
        optimizer.step()
        
        if ((iteration+1) % params.INTERVAL_ITERATION_T == 0):
            print("Training_T Iteration [{}/{}]: loss={} loss_mmd_class={} loss_mmd_general={} loss_intra={}"
                .format(iteration + 1,
                        params.NUM_ITERATION_T,
                        loss.detach().item(),
                        loss_mmd_class.detach().item(),
                        loss_mmd_general.detach().item(),
                        loss_intra.detach().item()))
        
    return encoder_t