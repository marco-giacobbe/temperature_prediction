import torch

import src.data_utils as data_utils
from matplotlib import pyplot
import math
import os

def save_checkpoint(model, optim, sched, epoch, path):
    checkpoint = {
        'model': model.state_dict()
        #'optim': optim.state_dict(),
        #'sched': sched.state_dict(),
        #'epoch': epoch,
    }
    torch.save(checkpoint, path)

def load_checkpoint(model, path): #, optim=None, sched=None):
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=torch.device("cpu"), weights_only=True)
        model.load_state_dict(checkpoint['model'])
        #if optim:
        #    optim.load_state_dict(checkpoint['optim'])
        #if sched:
        #  sched.load_state_dict(checkpoint['sched'])
        #epoch = checkpoint['epoch']
    return model#, optim, sched, epoch

def train1epoch(model, optim, criterion, dataset, bsz, out_w):
    model.train()
    total_loss = 0.
    for batch, i in enumerate(range(0, len(dataset) - 1, bsz)):
        src, tgt = data_utils.get_batch(dataset, i, bsz, out_w)
        optim.zero_grad()
        out = model(src)
        out, tgt = out[-out_w:], tgt[-out_w:]
        loss = criterion(out, tgt)
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optim.step()
        optim.zero_grad()
    return total_loss/dataset.shape[0]

def eval(model, criterion, dataset, out_w, plot):
    model.eval()
    total_loss = 0.
    test_result, truth = torch.Tensor(0), torch.Tensor(0)
    with torch.no_grad():
        for i in range(len(dataset) - 1):
            src, tgt = data_utils.get_batch(dataset, i,1, out_w)
            out = model(src)
            out, tgt = out[-out_w:], tgt[-out_w:]
            total_loss += criterion(out, tgt).item()
            #out, tgt = torch.round(out*40+2), tgt*40+2
            if not i%out_w:
                test_result = torch.cat((test_result, out.view(-1).cpu()), 0)
                truth = torch.cat((truth, tgt.view(-1).cpu()), 0)

    err = abs(test_result-truth)
    mean_loss = total_loss/dataset.shape[0]
    mean_err = torch.mean(err).item()
    var_err = torch.var(err).item()
    print("validation loss: {:.5f} | errore medio: {:.2f} | varianza: {:.2f} | deviazione standard: {:.2f}".format(mean_loss, mean_err, var_err, math.sqrt(var_err)))
    if plot:
        fig, (ax1, ax2) = pyplot.subplots(nrows=2, ncols=1, sharex=True)
        ax1.plot(truth)
        ax1.plot(test_result,color="red", alpha=0.35)
        ax1.legend(["target", "prediction"], loc='upper right')
        ax1.grid(True, which='both')
        ax1.axhline(y=0, color='k')
        ax1.set_xlabel('step')
        ax1.set_ylabel('value')

        ax2.plot(err)
        ax2.legend(["|prediction-target|"], loc='upper right')
        ax2.grid(True, which='both')
        ax2.axhline(y=0, color='k')
        ax2.set_xlabel('step')
        ax2.set_ylabel('value')
        pyplot.show()
        pyplot.close()

    return mean_loss, (truth, test_result)

def train(model, optim, criterion, scheduler, datasets, bsz, out_w, patience):
    train_dataset, eval_dataset = datasets
    train_losses, eval_losses = [], []
    loss_min = math.inf
    counter = 0
    epoch = 0
    while True:
        print("epoch: {} | ".format(epoch))
        current_loss = eval(model, criterion, eval_dataset, out_w, False)[0]
        eval_losses.append(current_loss)
        epoch += 1
        if current_loss < loss_min:
            counter = 0
            loss_min = current_loss
            save_checkpoint(model, optim, scheduler, epoch, "./tmp/model_ckpnt.pth")
        else:
            counter += 1
        if counter == patience:
            model = load_checkpoint(model, "./tmp/model_ckpnt.pth")
            current_loss = eval(model, criterion, eval_dataset, out_w, True)[0]
            return (train_losses[:-patience], eval_losses[1:-patience]) 
        train_losses.append(train1epoch(model, optim, criterion, train_dataset, bsz, out_w))
        scheduler.step()