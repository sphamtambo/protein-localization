import transformers
import torch
import torch.nn as nn
import time

from transformers import get_linear_schedule_with_warmup
from transformers import AdamW
from tqdm.auto import tqdm

from datamodule import *
from config import * 
from models import ProteinClassifier 
from main import accuracy_fn
from main import train_accuracy_fn
from main import valid_accuracy_fn
from main import test_accuracy_fn
from main import train_fn
from main import save_model


def run():
    """ final run """

    print(f"Training on {DEVICE} device")   
    seed_everything(RANDOM_SEED)

    model = ProteinClassifier()
    model.to(DEVICE)
    
    ### Prepare optimizer and schedule (linear warmup and decay)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    
    ### filter params with and without bias, gamma and beta
    ### bert only has bias, no gamma or beta
    optimizer_grouped_parameters = [
            ### filter params without bias, gamma, and beta
            {'params': [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.001},
            ### filter params with bias, gamma and beta
            {'params': [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0}
            ]
    
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, 
            lr=LEARNING_RATE,
            eps=ADAM_EPSILON)
    
    num_train_steps = len(train_loader) // GRADIENT_ACCUMULATION_STEPS * NUM_EPOCHS
    
    scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=0, 
            num_training_steps=num_train_steps)
    
    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        train_fn(model, train_loader, DEVICE, GRADIENT_ACCUMULATION_STEPS,
                    optimizer, scheduler, NUM_EPOCHS, epoch)
        train_accuracy_fn(model, train_loader, DEVICE)
        valid_accuracy_fn(model, valid_loader, DEVICE)
    
    print(f'\nTotal Training Time: {(time.time() - start_time)/60:.2f} min')
        
    test_accuracy_fn(model, test_loader, DEVICE)
    
    ### saving the model
    save_model(model) 

if __name__ == "__main__":
    run()	
