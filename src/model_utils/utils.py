import torch
import numpy as np

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

def eval(model, criterion, dataset, out_w, plot, verbose=True):
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

    err = test_result-truth
    abserr = abs(err)
    mean_loss = total_loss/dataset.shape[0]
    mean_err = torch.mean(abserr).item()
    var_err = torch.var(abserr).item()
    if verbose:
        print("validation loss: {:.5f} | errore medio: {:.2f} | varianza: {:.2f} | deviazione standard: {:.2f}".format(mean_loss, mean_err, var_err, math.sqrt(var_err)))
    if plot:
        fig, (ax1, ax2) = pyplot.subplots(nrows=2, ncols=1, sharex=True)
        ax1.plot(truth, color="k", marker="o")
        ax1.plot(test_result, color="#009E73", marker="v")
        ax1.legend(["target", "prediction"], loc='upper right')
        ax1.grid(True, which='both')
        ax1.axhline(y=0, color='k', alpha=0.6)
        ax1.set_xlabel('step')
        ax1.set_ylabel('value')

        ax2.plot(err, color="#D55E00")
        ax2.legend(["|prediction-target|"], loc='upper right')
        ax2.grid(True, which='both')
        ax2.axhline(y=0, color='k', alpha=0.6)
        ax2.set_xlabel('step')
        ax2.set_ylabel('value')
        pyplot.show()
        pyplot.close()

    return mean_loss, (truth, test_result)

def evaltf(interpreter, dataset, out_w):

    def invoke_tflite_interpreter(interpreter, input):
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], input)

        interpreter.invoke()
        # return output
        return np.reshape(interpreter.get_tensor(output_details[0]['index']),-1)

    test_result, truth = torch.Tensor(0), torch.Tensor(0)
    with torch.no_grad():
        for i in range(len(dataset) - 1):
            src, tgt = data_utils.get_batch(dataset, i,1, out_w)
            out = invoke_tflite_interpreter(interpreter, src.numpy())
            out, tgt = out[-out_w:], tgt[-out_w:]
            if not i%out_w:
                test_result = torch.cat((test_result, out.view(-1).cpu()), 0)
                truth = torch.cat((truth, tgt.view(-1).cpu()), 0)
    return truth, test_result

def train(model, optim, criterion, scheduler, datasets, bsz, out_w, patience, verbose):
    train_dataset, eval_dataset = datasets
    train_losses, eval_losses = [], []
    loss_min = math.inf
    counter = 0
    epoch = 0
    while True:
        if verbose:
            print("epoch: {} | ".format(epoch))
        current_loss = eval(model, criterion, eval_dataset, out_w, False, verbose=verbose)[0]
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
            current_loss = eval(model, criterion, eval_dataset, out_w, True, verbose)[0]
            return (train_losses[:-patience], eval_losses[1:-patience]) 
        train_losses.append(train1epoch(model, optim, criterion, train_dataset, bsz, out_w))
        scheduler.step()