import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import os
from ultralytics.nn.tasks import DetectionModel
from tqdm import tqdm
from types import SimpleNamespace
from dataset import YoloMapillaryDataset
from ultralytics.data.dataset import YOLODataset
from loss import DetectionLoss

# Adding customer helper function and summary writer
import helperFunctions as helper
from torch.utils.tensorboard import SummaryWriter


# MG - We define the training parameters here
TRAIN_IAMGES_LOCATION = "./dataset_light/images/train"
TRAIN_LABELS_LOCATION = "./dataset_light/labels/train"

VAL_IAMGES_LOCATION = './dataset_light/images/val'
VAL_LABELS_LOCATION = './dataset_light/labels/val'

LOG_FILE_LOCATION = "./logs/training_log_0.txt"

YOLO_CHECKPOINT_PATH = "./yolo11l.pt"

CHECKPOINT_LOCATION = "./checkpoint/"
RESUME_CHECKPOINT_LOCATION = "./checkpoint/checkpoint_best.pth"
DEVICE = "cuda:0"
USE_MULTI_GPU = True
BATCH_SIZE = 60
NUM_WORKERS = 4
LEARNING_RATE = 0.0001
NUM_EPOCHS = 50
START_EPOCH = 0
PRINT_INTERVAL = 20
T_max = 50

os.makedirs("./logs",exist_ok=True)
os.makedirs("./checkpoint",exist_ok=True)
os.makedirs("./output",exist_ok=True)

# Dataloader

train_dataset = YoloMapillaryDataset(
    images_path=TRAIN_IAMGES_LOCATION,
    labels_path=TRAIN_LABELS_LOCATION,
    img_size=640,
)

train_sz = int(len(train_dataset) * 0.8)
test_sz = len(train_dataset) - train_sz
train_dataset, test_dataset = random_split(train_dataset, [train_sz, test_sz])

val_dataset = YoloMapillaryDataset(
    images_path=VAL_IAMGES_LOCATION,
    labels_path=VAL_LABELS_LOCATION,
    img_size=640,
)

batch_size = BATCH_SIZE
num_workers = NUM_WORKERS

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=True,
    collate_fn=YOLODataset.collate_fn,
)


val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=True,
    collate_fn=YOLODataset.collate_fn,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=True,
    collate_fn=YOLODataset.collate_fn,
)

# Load model

model_checkpoint = torch.load(YOLO_CHECKPOINT_PATH)
model_config = model_checkpoint['model'].yaml

model = DetectionModel(cfg=model_config, nc=243, verbose=True)
model.load(model_checkpoint)

# MG - Commenting this one, as we are going to define the model with Data Parallel Later, if we train on multi-gpu
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#model = model.to(device)

for param in model.parameters():
    param.requires_grad = False

for name, param in model.model[-1].named_parameters():
    if name == 'dfl.conv.weight':
        param.requires_grad = False

    else:
        param.requires_grad = True

for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Trainable: {name}")

#MG - If we use multi-gpu, use dataparallel else, skip
if USE_MULTI_GPU:
    model = nn.DataParallel(model)

model = model.to(DEVICE)


# training loop
epochs = NUM_EPOCHS

# Loss weights
loss_hyp = SimpleNamespace(
    box=0.05,
    cls=0.5,
    dfl=1.5
)

criterion = DetectionLoss(model, hyp=loss_hyp, use_multi_gpu=USE_MULTI_GPU)

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# MG - Define LR Scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=0.0000001)

train_box_loss = []
train_cls_loss = []
train_dfl_loss = []

val_box_loss = []
val_cls_loss = []
val_dfl_loss = []

#======================================================
# MG - We keep a log of iteration and losses for logging
#======================================================
iteration = []
training_loss_list = [np.inf]
training_log = []
val_loss_list = [np.inf]
prev_loss = np.inf
iteration = 0

writer = SummaryWriter() # MG - Start summary writer before training

print("Total Training Batches: ",len(train_loader))
for epoch in range(epochs):
    #training_loop = tqdm(train_loader, desc=f"Training epoch: {epoch+1}/{epochs}")
    # MG - Print current epoch and learning rate
    print('Epoch:', (epoch+1), 'LR:', scheduler.get_lr())
    box_loss = []
    cls_loss = []
    dfl_loss = []
    
    model.train()
    for batch in train_loader: # MG - Iterate over the train loader, as we do not want to use tqdm for printing (for now)
        iteration += 1
        batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        images = batch['img']
        
        # Forward
        output = model.forward(images)
        
        loss, loss_items = criterion(output, batch)
        loss = loss.sum()
        
        # Backward
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        box_loss.append(loss_items[0].item())
        cls_loss.append(loss_items[1].item())
        dfl_loss.append(loss_items[2].item())

        #================== Print Information for Loss ======================#
        if iteration % 2 == 0:
            print('Epoch: {}\tIteration: {},\tBox_Loss: {:.7f},\tCLS_Loss: {:.7f},\tDFL_Loss: {:.7f},\tMix_Loss: {:.7f}'.format((epoch+1),
                                                        iteration,
                                                        loss_items[0].item(),
                                                        loss_items[1].item(),
                                                        loss_items[2].item(),
                                                        loss.item()),end="\r")
            '''
            writer.add_scalars("Loss", {'Box Loss': loss_items[0].item(),
                                        'CLS Loss': loss_items[1].item(),
                                        'DFL Loss': loss_items[2].item(),
                                        'Mix Loss': loss.item()}, iteration)
            '''
            writer.add_scalar("Training/Box Loss", loss_items[0].item(), iteration)
            writer.add_scalar("Training/CLS Loss", loss_items[1].item(), iteration)
            writer.add_scalar("Training/DFL Loss", loss_items[2].item(), iteration)
            writer.add_scalar("Training/Mix Loss", loss.item(), iteration)


        if iteration % PRINT_INTERVAL == 0:
            print_string = 'Epoch: {}\tIteration: {},\tBox_Loss: {:.7f},\tCLS_Loss: {:.7f},\tDFL_Loss: {:.7f},\tMix_Loss: {:.7f}'.format((epoch+1),
                                                        iteration,
                                                        loss_items[0].item(),
                                                        loss_items[1].item(),
                                                        loss_items[2].item(),
                                                        loss.item())
            print(print_string,end="\r")

            training_loss_list.append(loss.item())
            training_log.append(print_string)
            helper.writeLog(training_log, LOG_FILE_LOCATION)
    print("\n=============================================\n")

    # MG - Call learning rate scheduler
    scheduler.step()

    train_box_loss.append(np.mean(box_loss))
    train_cls_loss.append(np.mean(cls_loss))
    train_dfl_loss.append(np.mean(dfl_loss))
    print(f"Epoch {epoch} training loss:\nbox: {np.mean(box_loss):.4f}, cls: {np.mean(cls_loss):.4f}, dfl: {np.mean(dfl_loss):.4f}")

    # Validation
    validation_loop = tqdm(val_loader, desc=f"Training epoch: {epoch+1}/{epochs}")
    
    box_loss = []
    cls_loss = []
    dfl_loss = []
    
    
    model.eval()
    with torch.no_grad():
        for batch in validation_loop:
            batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            images = batch['img']
            output = model.forward(images)
            
            loss, loss_items = criterion(output, batch)
            
            box_loss.append(loss_items[0].item())
            cls_loss.append(loss_items[1].item())
            dfl_loss.append(loss_items[2].item())
            
    
    val_box_loss.append(np.mean(box_loss))
    val_cls_loss.append(np.mean(cls_loss))
    val_dfl_loss.append(np.mean(dfl_loss))
    print(f"Epoch {epoch} validation loss:\nbox: {np.mean(box_loss):.4f}, cls: {np.mean(cls_loss):.4f}, dfl: {np.mean(dfl_loss):.4f}")
    writer.add_scalar("Validation/Mix_Loss", val_box_loss[-1]+val_cls_loss[-1]+val_dfl_loss[-1], (epoch+1))
    writer.add_scalar("Validation/Box_Loss", val_box_loss[-1], (epoch+1))
    writer.add_scalar("Validation/Cls_Loss", val_cls_loss[-1], (epoch+1))
    writer.add_scalar("Validation/DFL_Loss", val_dfl_loss[-1], (epoch+1))


    # MG - Saving only when the validation loss decreases.
    val_loss_list.append(val_box_loss[-1]+val_cls_loss[-1]+val_dfl_loss[-1])
    if prev_loss > val_loss_list[-1]:
        prev_loss = val_loss_list[-1]
        # MG - If model is tarined on multi-gpu using data parallel, then saving is a bit different
        if USE_MULTI_GPU:
            helper.save_checkpoint(model, CHECKPOINT_LOCATION+"checkpoint_best.pth",save_parallel = True)
            helper.save_checkpoint(model, CHECKPOINT_LOCATION + "checkpoint_" + str(epoch + 1) + '.pth',save_parallel = True)
        else:
            helper.save_checkpoint(model, CHECKPOINT_LOCATION+"checkpoint_best.pth")
            helper.save_checkpoint(model, CHECKPOINT_LOCATION + "checkpoint_" + str(epoch + 1) + '.pth')


# MG - Close summary writer
writer.close()