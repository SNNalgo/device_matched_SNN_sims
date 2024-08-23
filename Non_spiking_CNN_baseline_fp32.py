import torch, torch.nn as nn
import snntorch as snn
import time
import numpy as np

from snntorch import surrogate
import snntorch.functional as SF

from scipy.io import savemat

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class All_CNN(nn.Module):
    def __init__(self, in_ch, out_sz):
        super().__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.bn1 = nn.BatchNorm2d(96, affine=False)
        self.bn2 = nn.BatchNorm2d(96, affine=False)
        self.bn3 = nn.BatchNorm2d(96, affine=False)
        self.bn4 = nn.BatchNorm2d(192, affine=False)
        self.bn5 = nn.BatchNorm2d(192, affine=False)
        self.bn6 = nn.BatchNorm2d(192, affine=False)
        self.bn7 = nn.BatchNorm2d(192, affine=False)
        self.bn8 = nn.BatchNorm2d(192, affine=False)
        self.bn9 = nn.BatchNorm2d(out_sz, affine=False)
        self.cnn1_1 = nn.Conv2d(in_ch, 96, (3,3), padding='same', bias=False)
        self.cnn1_2 = nn.Conv2d(96, 96, (3,3), padding='same', bias=False)
        self.cnn1_3 = nn.Conv2d(96, 96, (3,3), padding='same', bias=False)
        self.mp1 = nn.MaxPool2d(2, 2)
        self.cnn2_1 = nn.Conv2d(96, 192, (3,3), padding='same', bias=False)
        self.cnn2_2 = nn.Conv2d(192, 192, (3,3), padding='same', bias=False)
        self.cnn2_3 = nn.Conv2d(192, 192, (3,3), padding='same', bias=False)
        self.mp2 = nn.MaxPool2d(2, 2)
        self.cnn3_1 = nn.Conv2d(192, 192, (3,3), padding='same', bias=False)
        self.cnn3_2 = nn.Conv2d(192, 192, (1,1), padding='same', bias=False)
        self.cnn3_3 = nn.Conv2d(192, out_sz, (1,1), padding='same', bias=False)
        self.fl = nn.Flatten()
        self.fc_out = nn.Linear(640, out_sz, bias=False)
    
    def forward(self, x):
        cnn1 = self.cnn1_1(x)
        cnn1 = self.bn1(cnn1)
        cnn1 = self.relu(cnn1)
        
        cnn2 = self.cnn1_2(cnn1)
        cnn2 = self.bn2(cnn2)
        cnn2 = self.relu(cnn2)
        
        cnn3 = self.cnn1_3(cnn2)
        cnn3 = self.bn3(cnn3)
        cnn3 = self.relu(cnn3)
        
        mpool1 = self.mp1(cnn3)
        
        cnn4 = self.cnn2_1(mpool1)
        cnn4 = self.bn4(cnn4)
        cnn4 = self.relu(cnn4)
        
        cnn5 = self.cnn2_2(cnn4)
        cnn5 = self.bn5(cnn5)
        cnn5 = self.relu(cnn5)
        
        cnn6 = self.cnn2_3(cnn5)
        cnn6 = self.bn6(cnn6)
        cnn6 = self.relu(cnn6)
        
        mpool2 = self.mp2(cnn6)
        
        cnn7 = self.cnn3_1(mpool2)
        cnn7 = self.bn7(cnn7)
        cnn7 = self.relu(cnn7)
        
        cnn8 = self.cnn3_2(cnn7)
        cnn8 = self.bn8(cnn8)
        cnn8 = self.relu(cnn8)
        
        cnn9 = self.cnn3_3(cnn8)
        cnn9 = self.bn9(cnn9)
        cnn9 = self.relu(cnn9)
        fl_vec = self.fl(cnn9)
        y = (self.fc_out(fl_vec))
        return y

def test_accuracy(data_loader, net):
    with torch.no_grad():
        total = 0
        acc = 0
        net.eval()
        data_loader = iter(data_loader)
        for data, targets in data_loader:
            data = data.to(device)
            targets = targets.to(device)
            spk_rec = net(data)
            _, predicted = torch.max(spk_rec.data, 1)
            acc += (predicted == targets).sum().item()
            total += spk_rec.size(0)
    return acc/total

if __name__ == '__main__':

    batch_size = 256
    # Quantization scheme. "Wt_bits" refers to no. of bits of quantization
    wt_limit = 0.05
    Wt_bits = 8
    Wt_vals = 2**Wt_bits-1
    zero_pt = 2**(Wt_bits-1)
    quant_step = np.float32((2*wt_limit)/Wt_vals)

    #device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: ", device)

    # Define a transform
    train_transform = transforms.Compose([
            # add some data augmentation
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

    test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

    data_path='../data/CIFAR_10'
    cifar_train = datasets.CIFAR10(data_path, train=True, download=True, transform=train_transform)
    cifar_test = datasets.CIFAR10(data_path, train=False, download=True, transform=test_transform)

    # Create DataLoaders
    train_loader = DataLoader(cifar_train, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(cifar_test, batch_size=batch_size, shuffle=True, num_workers=4)

    in_ch = 3
    out_sz = 10 #for CIFAR-10
    
    num_epochs = 100
    lr = 1e-3
    
    model = All_CNN(in_ch, out_sz).to(device)
    model_test = All_CNN(in_ch, out_sz).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    loss_fn = nn.CrossEntropyLoss()
    
    run_start = time.time()
    accs_wo_var = []
    accs_w_var = []
    best_acc = 0
    for epoch in range(num_epochs):
        model.train()
        for i, (data, targets) in enumerate(iter(train_loader)):
            data = data.to(device)
            targets = targets.to(device)
            
            spk_rec = model(data) # forward-pass
            loss_val = loss_fn(spk_rec, targets) # loss calculation
            
            optimizer.zero_grad() # null gradients
            loss_val.backward() # calculate gradients
            optimizer.step() # update weights
        model.eval()
        model_test.eval()
        print("Epoch: ", epoch)
        acc = test_accuracy(test_loader, model)
        print("test accuracy(%) model: ", acc * 100)

        with torch.no_grad():
            model_test.load_state_dict(model.state_dict())
            acc2 = test_accuracy(test_loader, model_test)
            print("test accuracy(%) model_test: ", acc2 * 100)
        
        if acc2>best_acc:
            best_acc = acc2
        
        accs_wo_var.append(acc)
        accs_w_var.append(acc2)
    
    run_end = time.time()
    print("total runtime: ", run_end - run_start, "seconds")
    print("best test accuracy(%) with variations: ", 100*best_acc)


