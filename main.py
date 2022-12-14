import torch
import torch.nn as nn
import time
import os
from tqdm.auto import tqdm

def accuracy_fn(model, data_loader, device):
    """ validate model """
   
    model.eval()
    with torch.no_grad():
        correct_pred = 0
        num_examples = 0
        for bacth_idx, batch in enumerate(data_loader):
            ### prepare data
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            ### get logits
            logits = model(input_ids, attention_mask=attention_mask)

            ### get max values of the pred class labels, skip indeces
            _, predicted_labels = torch.max(logits, dim=1)

            ### total number of examples
            num_examples += labels.size(0)

            ### number of correct predictions
            correct_pred += (predicted_labels == labels).sum()

    ### accuracy as propotion of the correctly pred labels
    return correct_pred.float()/num_examples * 100

def train_accuracy_fn(model, data_loader, device):
    """ train accuracy """

    with torch.set_grad_enabled(False):
        print(f'Training accuracy: '
                f'{accuracy_fn(model, data_loader, device):.2f}%')


def valid_accuracy_fn(model, data_loader, device):
    """ validation accuracy """

    with torch.set_grad_enabled(False):
        print(f'Validation accuracy: '
                f'{accuracy_fn(model, data_loader, device):.2f}%')


def test_accuracy_fn(model, data_loader, device):
    """ test accuracy """

    print(f'\nTest accuracy: {accuracy_fn(model, data_loader, device):.2f}%')


def train_fn(model, data_loader, device, accumulation_steps, 
        optimizer, scheduler, num_epochs, epoch):
    """ train model """

    start_time = time.time()

    loss_fn = nn.CrossEntropyLoss().to(device)
    
    model.train()
    for batch_idx, batch in enumerate(data_loader):
        ### prepare data
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        ### clear gradients
        model.zero_grad()
    
        ### foward pass
        logits = model(input_ids, attention_mask)
        loss = loss_fn(logits, labels)
    
        ### accumulate gradients to save memory
        ### loss normalization to the mean of the accumulated batch size
        loss = loss_fn(logits, labels) / accumulation_steps
    
        ### bakward pass
        optimizer.zero_grad()
        loss.backward()
        
        ### wait for several steps and update model params
        ### based on cummulative gradeint after specific batch no.
        if (batch_idx +1) % accumulation_steps == 0:
            ### reset gradients, for the next accumulated batches
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
    
        ### logging
        if batch_idx % 500 == 0:
            print(f'Epoch [{epoch+1:04d}/{num_epochs:04d}], '
                    f'Step [{batch_idx+1:04d}/{len(data_loader):04d}], '
                    f'Loss: {loss:.4f}')
    print(f'\nTime elapsed: {(time.time() - start_time)/60:.2f} min')


def save_model(model):
    """ save model and best training parameters """

    print("\nSaving the model")
    modelFolderPath = 'model/'
    path = os.path.join(modelFolderPath, 'model.pth')
    if not os.path.exists(modelFolderPath):
        os.makedirs(modelFolderPath)

    torch.save(model.state_dict(), path)
    print(f"Model saved in {path}")



