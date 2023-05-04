#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import argparse

from collections import Counter


class UADLoss(nn.Module):
    def __init__(self, num_clients):
        super(UADLoss, self).__init__()
        self.num_clients = num_clients

    def forward(self, t):
        eps = 1e-7
        batch_loss = torch.div(-1, self.num_clients) * torch.sum(torch.log(t + eps), dim=1)
        return torch.div(batch_loss.sum(), batch_loss.shape[0])


class DiscriminatorLoss(nn.Module):
    def __init__(self, num_group_clients):
        super(DiscriminatorLoss, self).__init__()
        self.num_group_clients = num_group_clients

    def forward(self, d_self, d_other):
        eps = 1e-7
        loss = -torch.log(d_self + eps) - torch.div(1, self.num_group_clients - 1) * \
               torch.sum(torch.log(d_other + eps), dim=1).view(-1, 1)
        return torch.div(loss.sum(), loss.shape[0])


def get_clients_p(num_classes_dicts, num_classes=10):
    num_classes = num_classes
    total_count = Counter()
    num = len(clients)
    for i in range(num):
        total_count += num_classes_dicts[i]
    result = []
    for i in range(num):
        class_frac = [0.0 for i in range(num_classes)]
        # label is type of tensor
        for label, count in num_classes_dicts[i].items():
            class_frac[label] = count / total_count[label]
        result.append(class_frac)
    return np.array(result)


def get_clients_batch_predict(X, clients, args, num_classes=10):
    num = len(clients)
    batch_size = args.local_bs
    num_classes = num_classes
    batch_predict = np.zeros((batch_size, num, num_classes))
    for client_idx in range(num):
        client_batch_predict = clients[client_idx](X).cpu().detach().numpy()
        for idx, pred in enumerate(client_batch_predict):
            batch_predict[idx, client_idx, :] = pred
    return batch_predict


def get_mixed_predict(X, clients, num_classes_dicts, args, num_classes=10):
    clients_batch_predict = get_clients_batch_predict(X, clients, args, num_classes)
    clients_p = get_clients_p(num_classes_dicts, num_classes)
    result = []
    for clients_predict in clients_batch_predict:
        result.append(F.softmax(torch.tensor(np.sum(clients_predict * clients_p, axis=0)), dim=0).detach().tolist())
    return torch.tensor(result)


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        self.criterion = nn.CrossEntropyLoss().to(self.device)

        self.uad_criterion = UADLoss(num_clients=args.num_users).to(self.device)
        
        self.discriminator_optimizer = None
        self.discriminator_loss = DiscriminatorLoss(num_group_clients=args.num_group_clients).to(self.device)
        self.cgr_loss = torch.nn.KLDivLoss(reduction="mean").to(self.device)

        self.lambda_cgl = 2

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val)/10), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test)/10), shuffle=False)
        return trainloader, validloader, testloader

    def update_weights(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.9, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model, model.state_dict(), sum(epoch_loss) / len(epoch_loss)



    def update_weights_align(self, model, idx, all_models, num_classes_dicts, global_model, disc, global_round, writer):
        for local_model in all_models:
            local_model.eval()
        disc.eval()
        model.train()
        global_model.eval()
        epoch_loss = []

        if not self.discriminator_optimizer:
            self.discriminator_optimizer = torch.optim.SGD(disc.parameters(), lr=self.args.lr,
                                                       momentum=0.5, weight_decay=self.args.weight_decay)

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        cross_entropy = nn.CrossEntropyLoss().to(self.device)
        
        
        lambda_val = 0.1 # to control the impact of the uniformity loss
        batch_loss = []
        for batch_idx, (images, labels) in enumerate(self.trainloader):
            data, target = images.to(self.device), labels.to(self.device)
            model.zero_grad()


            y_hat = model(data).to(self.device)
            cls = cross_entropy(y_hat, target)  
            f_self = model.get_features(images).to(self.device)
            d_self = disc(f_self).to(self.device)
            extractor_loss = cls + self.uad_loss(d_self)
            extractor_loss.backward()
            optimizer.step()
            batch_loss.append(extractor_loss.item())

            
            #f_i = global_model.get_features(images)
            #d_hat = disc(f_i)
            #l_uniform = -torch.mean(torch.log(d_hat))
#
            # log_probs = model(images)
            #loss = self.criterion(log_probs, labels)
            #loss_comb = loss + l_uniform
            #loss_comb.backward()
            #loss = self.criterion(log_probs, labels) + l_uniform
            #loss = self.criterion(log_probs, labels) + lambda_val * l_uniform
            # loss_.backward()
            
            # Print gradients
            for name, param in model.named_parameters():
                if param.grad is not None:
                    # print(f'Gradients of {name}: {param.grad}')
                    writer.add_histogram(f'{name}.grad', param.grad, global_round)

            #optimizer.step()


            if self.args.verbose and (batch_idx % 10 == 0):
                print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    global_round, iter, batch_idx * len(images),
                    len(self.trainloader.dataset),
                    100. * batch_idx / len(self.trainloader), loss.item()))
            self.logger.add_scalar('loss', loss.item())
            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss)/len(batch_loss))

        disc.train()
        model.eval()
        disc_loss = 0
        batch_loss = []
        for batch_idx, (images, labels) in enumerate(self.trainloader):
            data, target = images.to(self.device), labels.to(self.device)
            
            disc.zero_grad()

            d_j_batch_list = []
            for j, local_model in enumerate(all_models):
                if j == idx:
                    continue
                other_model = all_models[j]
                f_j_batch = other_model.get_features(data).to(self.device)
                d_j_batch = disc(f_j_batch)[:, j].to(self.device)
                d_j_batch_list.append(d_j_batch.view(1, -1))

            y_hat = self.poster_model(data).to(self.device)
            f_self = self.poster_model.feature.to(self.device)

            self.discriminator_optimizer.zero_grad()
            d_self = disc(f_self.detach())[:, idx].to(self.device)
            d_j_batch = torch.cat(d_j_batch_list).T.detach().to(self.device)
            discriminator_loss = self.discriminator_loss(d_self, d_j_batch)
            discriminator_loss.backward()
            batch_loss.append(discriminator_loss.item())
            self.discriminator_optimizer.step()

            
            #f_i = global_model.get_features(images)
            #d_hat = disc(f_i) 

            #d_tilda_loss = torch.zeros_like(d_hat[:,0])
            #for i, local_model in enumerate(all_models):
            #    d_tilda_loss += torch.log(disc(local_model.get_features(images))[:,i])

            #d_tilda_loss = d_tilda_loss / (len(all_models)-1)
            # print(d_tilda_loss.shape, d_hat[idx].shape)
            #total_loss = -torch.log(d_hat[:,idx]) - d_tilda_loss
            #disc.zero_grad()
            #total_loss.mean().backward()
            #optimizer.step()
            # scalar_total_loss = total_loss.mean()  # or total_loss.sum()
            # scalar_total_loss.backward()
            # total_loss.backward()
            # disc_loss += scalar_total_loss.item()
            #disc_loss += total_loss.mean().item()

        
        kl_loss = nn.KLDivLoss(reduction="batchmean")
        kl_loss_log = nn.KLDivLoss(reduction="batchmean", log_target=True)

         # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5,  weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        disc.eval()
        model.train()
        batch_loss = []
        for iter in range(self.args.local_ep):
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                data, target = images.to(self.device), labels.to(self.device)
                model.zero_grad()


                y_wave = F.softmax(get_mixed_predict(data, all_models, num_classes_dicts, self.args), dim=1).to(self.device)
                y_hat = model(data)
                gl_pred = global_model(data)
                # logging.info("y_wave:"+str(y_wave))
                # logging.info("y_hat:"+str(y_hat))
                cgr_loss = self.cgr_loss(y_wave.log(), y_hat)
                loss_cgl = self.cgr_loss(gl_pred.log(), y_hat)

                y_hat = model(data).to(self.device)
                cls = cross_entropy(y_hat, target)  
                f_self = model.get_features(images).to(self.device)
                d_self = disc(f_self).to(self.device)
                extractor_loss = cls + self.uad_loss(d_self)

                total_loss = cgr_loss + extractor_loss + self.lambda_cgl * loss_cgl
                batch_loss.append(total_loss.item())
                total_loss.backward()
                optimizer.step()


                #y_cgr = torch.zeros(10,10)
                #for i, local_model in enumerate(all_models):
                #    if i == idx:
                #        continue
                #    y_cgr += local_model(images)

                #y_cgr_softmax = nn.functional.softmax(y_cgr, dim=1) 

                
                #f_i = global_model.get_features(images)
                #gl_pred = global_model(images)
                #d_hat = disc(f_i)
                #l_uniform = -torch.mean(torch.log(d_hat))

                #log_probs = model(images)
                #loss = self.criterion(log_probs, labels)
                #loss_cgr = kl_loss(log_probs, y_cgr_softmax)
                #loss_cgl = kl_loss_log(log_probs, gl_pred)
                #loss_comb = loss + l_uniform + loss_cgr + self.lambda_cgl * loss_cgl
                print(f"total loss: {total_loss.item()}")
                #loss_comb.backward()
                # loss.backward()
                #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.args.clip_value)
                #optimizer.step()

        return model.state_dict(), sum(batch_loss) / len(batch_loss), disc_loss / len(self.trainloader)

    def update_weights_disc(self, model, idx, all_models, global_model, disc, global_round, writer):
        for local_model in all_models:
            local_model.eval()
        disc.train()
        model.eval()
        epoch_loss = []

        optimizer = torch.optim.Adam(disc.parameters(), lr=self.args.lr, weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                disc.zero_grad()

                f_i = global_model.get_features(images)
                d_hat = disc(f_i)

                d_tilda_loss = torch.zeros_like(d_hat[:,0])
                for i, local_model in enumerate(all_models):
                    if i == idx:
                        continue
                    d_tilda_loss += torch.log(disc(local_model.get_features(images))[:,i])

                d_tilda_loss = d_tilda_loss / (len(all_models)-1)
                total_loss = -torch.log(d_hat[:,idx]) - d_tilda_loss
                disc.zero_grad()
                total_loss.mean().backward()
                optimizer.step()

                disc_loss = total_loss.mean().item()
                batch_loss.append(disc_loss)
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return sum(epoch_loss) / len(epoch_loss)

    
    def update_weights_align_KL(self, model, idx, all_models, global_model, disc, global_round, writer):
        for local_model in all_models:
            local_model.eval()
        disc.eval()
        model.train()
        global_model.eval()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                        weight_decay=1e-4)

        kl_loss = nn.KLDivLoss(reduction="batchmean")

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()

                y_cgr = torch.zeros(10, 10)
                for i, local_model in enumerate(all_models):
                    if i == idx:
                        continue
                    y_cgr += local_model(images)

                y_cgr_softmax = nn.functional.softmax(y_cgr, dim=1)

                log_probs = model(images)
                loss = kl_loss(log_probs, y_cgr_softmax)
                loss.backward()

                # Print gradients
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        # print(f'Gradients of {name}: {param.grad}')
                        writer.add_histogram(f'{name}.grad', param.grad, global_round)

                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), None


    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss


def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss
