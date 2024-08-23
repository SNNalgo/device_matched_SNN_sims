import torch, torch.nn as nn
import snntorch as snn
import time
import numpy as np

from snntorch import surrogate
import snntorch.functional as SF

from scipy.io import savemat

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse

parser = argparse.ArgumentParser(description='Quant Aware Training')
parser.add_argument('--correct_spiking_rate', default=0.8, type=float, help='spiking rate for correct label for loss compute')
parser.add_argument('--base_curr', default=3e-9, type=float, help='base current for Iscale')
parser.add_argument('--vth', default=0.7, type=float, help='spiking threshold voltage')
parser.add_argument('--faulty_vth', default=0.7, type=float, help='faulty spiking threshold voltage in fabricated devices')
parser.add_argument('--Ith', default=4.5e-10, type=float, help='threshold current')
parser.add_argument('--C', default=35e-15, type=float, help='Capacitance of equivalent RC LIF model')
parser.add_argument('--faulty_C', default=35e-15, type=float, help='faulty Capacitance of equivalent RC LIF model')
parser.add_argument('--vth_var', default=5, type=float, help='spiking threshold voltage variability(%)')
parser.add_argument('--tau_var', default=5, type=float, help='time constant variability(%)')

args = parser.parse_args()

class All_CNN(nn.Module):
    def __init__(self, in_ch, out_sz):
        super().__init__()
        self.cnn1_1 = nn.Conv2d(in_ch, 96, (3,3), padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(96, affine=False)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.cnn1_2 = nn.Conv2d(96, 96, (3,3), padding='same', bias=False)
        self.bn2 = nn.BatchNorm2d(96, affine=False)
        self.cnn1_3 = nn.Conv2d(96, 96, (3,3), padding='same', bias=False)
        self.bn3 = nn.BatchNorm2d(96, affine=False)
        self.mp1 = nn.MaxPool2d(2, 2)
        self.cnn2_1 = nn.Conv2d(96, 192, (3,3), padding='same', bias=False)
        self.bn4 = nn.BatchNorm2d(192, affine=False)
        self.cnn2_2 = nn.Conv2d(192, 192, (3,3), padding='same', bias=False)
        self.bn5 = nn.BatchNorm2d(192, affine=False)
        self.cnn2_3 = nn.Conv2d(192, 192, (3,3), padding='same', bias=False)
        self.bn6 = nn.BatchNorm2d(192, affine=False)
        self.mp2 = nn.MaxPool2d(2, 2)
        self.cnn3_1 = nn.Conv2d(192, 192, (3,3), padding='same', bias=False)
        self.bn7 = nn.BatchNorm2d(192, affine=False)
        self.cnn3_2 = nn.Conv2d(192, 192, (1,1), padding='same', bias=False)
        self.bn8 = nn.BatchNorm2d(192, affine=False)
        self.cnn3_3 = nn.Conv2d(192, out_sz, (1,1), padding='same', bias=False)
        self.bn9 = nn.BatchNorm2d(out_sz, affine=False)
        self.fl = nn.Flatten()
        self.fc_out = nn.Linear(640, out_sz, bias=False)
    
    def forward(self, x):
        cnn1 = self.relu(self.cnn1_1(x))
        cnn1 = self.bn1(cnn1)
        cnn2 = self.relu(self.cnn1_2(cnn1))
        cnn2 = self.bn2(cnn2)
        cnn3 = self.relu(self.cnn1_3(cnn2))
        cnn3 = self.bn3(cnn3)
        mpool1 = self.mp1(cnn3)
        cnn4 = self.relu(self.cnn2_1(mpool1))
        cnn4 = self.bn4(cnn4)
        cnn5 = self.relu(self.cnn2_2(cnn4))
        cnn5 = self.bn5(cnn5)
        cnn6 = self.relu(self.cnn2_3(cnn5))
        cnn6 = self.bn6(cnn6)
        mpool2 = self.mp2(cnn6)
        cnn7 = self.relu(self.cnn3_1(mpool2))
        cnn7 = self.bn7(cnn7)
        cnn8 = self.relu(self.cnn3_2(cnn7))
        cnn8 = self.bn8(cnn8)
        cnn9 = self.relu(self.cnn3_3(cnn8))
        cnn9 = self.bn9(cnn9)
        fl_vec = self.fl(cnn9)
        y = self.relu(self.fc_out(fl_vec))
        return y

    def forward_act(self, x):
        cnn1 = self.relu(self.cnn1_1(x))
        cnn1 = self.bn1(cnn1)
        #cnn1 = self.dropout(cnn1)
        cnn2 = self.relu(self.cnn1_2(cnn1))
        cnn2 = self.bn2(cnn2)
        #cnn2 = self.dropout(cnn2)
        cnn3 = self.relu(self.cnn1_3(cnn2))
        cnn3 = self.bn3(cnn3)
        mpool1 = self.mp1(cnn3)
        cnn4 = self.relu(self.cnn2_1(mpool1))
        cnn4 = self.bn4(cnn4)
        #cnn4 = self.dropout(cnn4)
        cnn5 = self.relu(self.cnn2_2(cnn4))
        cnn5 = self.bn5(cnn5)
        #cnn5 = self.dropout(cnn5)
        cnn6 = self.relu(self.cnn2_3(cnn5))
        cnn6 = self.bn6(cnn6)
        mpool2 = self.mp2(cnn6)
        cnn7 = self.relu(self.cnn3_1(mpool2))
        cnn7 = self.bn7(cnn7)
        #cnn7 = self.dropout(cnn7)
        cnn8 = self.relu(self.cnn3_2(cnn7))
        cnn8 = self.bn8(cnn8)
        #cnn8 = self.dropout(cnn8)
        cnn9 = self.relu(self.cnn3_3(cnn8))
        cnn9 = self.bn9(cnn9)
        fl_vec = self.fl(cnn9)
        y = self.relu(self.fc_out(fl_vec))
        return [cnn1, cnn2, cnn3, cnn4, cnn5, cnn6, cnn7, cnn8, cnn9, y]

class All_CNN_spiking(nn.Module):
    def __init__(self, in_ch, out_sz, Iscale, time_step, taus, vths):
        super().__init__()
        self.Iscale = Iscale # Scaling factor for hardware equivalent model
        self.dropout = nn.Dropout()
        
        self.cnn1_1 = nn.Conv2d(in_ch, 96, (3,3), padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(96, affine=False)
        self.cnn1_2 = nn.Conv2d(96, 96, (3,3), padding='same', bias=False)
        self.bn2 = nn.BatchNorm2d(96, affine=False)
        self.cnn1_3 = nn.Conv2d(96, 96, (3,3), padding='same', bias=False)
        self.bn3 = nn.BatchNorm2d(96, affine=False)
        self.mp1 = nn.MaxPool2d(2, 2)
        #self.mp1 = nn.AvgPool2d(2, 2)
        self.cnn2_1 = nn.Conv2d(96, 192, (3,3), padding='same', bias=False)
        self.bn4 = nn.BatchNorm2d(192, affine=False)
        self.cnn2_2 = nn.Conv2d(192, 192, (3,3), padding='same', bias=False)
        self.bn5 = nn.BatchNorm2d(192, affine=False)
        self.cnn2_3 = nn.Conv2d(192, 192, (3,3), padding='same', bias=False)
        self.bn6 = nn.BatchNorm2d(192, affine=False)
        self.mp2 = nn.MaxPool2d(2, 2)
        #self.mp2 = nn.AvgPool2d(2, 2)
        self.cnn3_1 = nn.Conv2d(192, 192, (3,3), padding='same', bias=False)
        self.bn7 = nn.BatchNorm2d(192, affine=False)
        self.cnn3_2 = nn.Conv2d(192, 192, (1,1), padding='same', bias=False)
        self.bn8 = nn.BatchNorm2d(192, affine=False)
        self.cnn3_3 = nn.Conv2d(192, out_sz, (1,1), padding='same', bias=False)
        self.bn9 = nn.BatchNorm2d(out_sz, affine=False)
        self.fl = nn.Flatten()
        self.fc_out = nn.Linear(640, out_sz, bias=False)
        self.lifs = []
        for i in range(len(taus)):
            lif_i = snn.Leaky(beta=1-time_step/taus[i], threshold=vths[i], learn_beta=False, learn_threshold=False) #spiking neurons
            self.lifs.append(lif_i)
    
    def forward(self, x, num_steps):
        #Initialize the LIF neurons
        mems = []
        spk_rec = []
        for i in range(len(self.lifs)):
            mems.append(self.lifs[i].init_leaky())
        
        for step in range(num_steps):
            scaled_x = torch.mul(x, self.Iscale[0])
            cnn1 = (self.cnn1_1(scaled_x))
            cnn1 = self.bn1(cnn1)
            spk1, mems[0] = self.lifs[0](cnn1, mems[0])
            scaled_spk1 = torch.mul(spk1, self.Iscale[1]) # Scaling the hidden layer spikes in terms of current
            
            cnn2 = (self.cnn1_2(scaled_spk1))
            cnn2 = self.bn2(cnn2)
            spk2, mems[1] = self.lifs[1](cnn2, mems[1])
            scaled_spk2 = torch.mul(spk2, self.Iscale[2])
            
            cnn3 = (self.cnn1_3(scaled_spk2))
            cnn3 = self.bn3(cnn3)
            spk3, mems[2] = self.lifs[2](cnn3, mems[2])
            scaled_spk3 = torch.mul(spk3, 4*self.Iscale[3])
            #scaled_spk3 = torch.mul(spk3, self.Iscale)
            
            mpool1 = self.mp1(scaled_spk3)
            
            cnn4 = (self.cnn2_1(mpool1))
            cnn4 = self.bn4(cnn4)
            spk4, mems[3] = self.lifs[3](cnn4, mems[3])
            scaled_spk4 = torch.mul(spk4, self.Iscale[4]) # Scaling the hidden layer spikes in terms of current
            
            cnn5 = (self.cnn2_2(scaled_spk4))
            cnn5 = self.bn5(cnn5)
            spk5, mems[4] = self.lifs[4](cnn5, mems[4])
            scaled_spk5 = torch.mul(spk5, self.Iscale[5])
            
            cnn6 = (self.cnn2_3(scaled_spk5))
            cnn6 = self.bn6(cnn6)
            spk6, mems[5] = self.lifs[5](cnn6, mems[5])
            scaled_spk6 = torch.mul(spk6, 4*self.Iscale[6])
            #scaled_spk6 = torch.mul(spk6, self.Iscale)

            mpool2 = self.mp2(scaled_spk6)
            
            cnn7 = (self.cnn3_1(mpool2))
            cnn7 = self.bn7(cnn7)
            spk7, mems[6] = self.lifs[6](cnn7, mems[6])
            scaled_spk7 = torch.mul(spk7, self.Iscale[7]) # Scaling the hidden layer spikes in terms of current
            
            cnn8 = (self.cnn3_2(scaled_spk7))
            cnn8 = self.bn8(cnn8)
            spk8, mems[7] = self.lifs[7](cnn8, mems[7])
            scaled_spk8 = torch.mul(spk8, self.Iscale[8])
            
            cnn9 = (self.cnn3_3(scaled_spk8))
            cnn9 = self.bn9(cnn9)
            spk9, mems[8] = self.lifs[8](cnn9, mems[8])
            scaled_spk9 = torch.mul(spk9, self.Iscale[9])
            
            fl_vec = self.fl(scaled_spk9)
            curr_out = (self.fc_out(fl_vec))
            spk_out, mems[9] = self.lifs[9](curr_out, mems[9])
            
            spk_rec.append(spk_out)
        spk_rec_out = torch.stack(spk_rec)
        return spk_rec_out

def test_accuracy(data_loader, net, num_steps):
    with torch.no_grad():
        total = 0
        acc = 0
        net.eval()
        data_loader = iter(data_loader)
        for data, targets in data_loader:
            data = data.to(device)
            targets = targets.to(device)
            spk_rec = net(data, num_steps)
            acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
            total += spk_rec.size(1)
    return acc/total

if __name__ == '__main__':

    batch_size = 256
    # Quantization scheme. "Wt_bits" refers to no. of bits of quantization
    Wt_bits = 8

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
    
    # Hardware equivalent spiking neuron model
    vth = args.vth # Taken from device description
    faulty_vth = args.faulty_vth
    vrst = 0
    Ith = args.Ith # Taken from device description
    R = (vth - vrst)/Ith
    C = args.C # Estimated by matching "f vs I" of simulated LIF neurons against device data (Vg=-1.1V)
    tau = R*C # time constant
    faulty_C = args.faulty_C
    faulty_tau = R*faulty_C
    print('tau: ', tau)
    time_step = tau/5 # time-step of the SNN simulation
    Iscale = []# Use different Iscale based on kernel size/num filters
    
    num_steps = 10
    #num_steps = 5
    num_epochs = 200
    lr = 1e-3
    
    info_model = All_CNN(in_ch, out_sz).to(device)
    
    taus = []
    vths = []
    
    taus_tr = []
    vths_tr = []
    
    vth_var = args.vth_var
    tau_var = args.tau_var

    print("vth variation(%): ", vth_var)
    print("tau variation(%): ", tau_var)
    
    with torch.no_grad():
        data, targets = next(iter(train_loader))
        print(data.shape)
        data = data.to(device)
        activations = info_model.forward_act(data)
        for act in activations:
            act_np = act.data.cpu().numpy()
            act_np_shape = act_np.shape
            layer_shape = act_np_shape[1:]
            rn_var1 = np.float32((vth_var/100)*np.random.normal(size=layer_shape))
            rn_var2 = np.float32((tau_var/100)*np.random.normal(size=layer_shape))
            vth_w_var = torch.from_numpy(faulty_vth*(1+rn_var1)).to(device)
            tau_w_var = torch.from_numpy(faulty_tau*(1+rn_var2)).to(device)
            vths.append(vth_w_var)
            taus.append(tau_w_var)
            
            vth_tr = torch.from_numpy(vth*np.float32(np.ones(layer_shape))).to(device)
            tau_tr = torch.from_numpy(tau*np.float32(np.ones(layer_shape))).to(device)
            vths_tr.append(vth_tr)
            taus_tr.append(tau_tr)
        
        for (names, params) in info_model.named_parameters():
            if params.requires_grad and ('bn' not in names) and ('downsample' not in names) and ('bias' not in names):
                if len(params.shape) == 4:
                    incoming_sz = params.shape[1]*params.shape[2]*params.shape[3]
                    base_curr = args.base_curr
                    base_incoming_sz = 800
                    scaled_curr = base_curr*np.sqrt(base_incoming_sz)/np.sqrt(incoming_sz) #equalizing I*N/sqrt(N) [denominator appears because weights are initialized around 1/sqrt(N)]
                    if scaled_curr > 5e-9:
                        scaled_curr = 5e-9
                    print('scaled_curr: ', scaled_curr)
                    effective_Iscale = scaled_curr*time_step/C
                    Iscale.append(effective_Iscale)
                if len(params.shape) == 2:
                    incoming_sz = params.shape[1]
                    base_curr = args.base_curr
                    base_incoming_sz = 800
                    if scaled_curr > 5e-9:
                        scaled_curr = 5e-9
                    scaled_curr = base_curr*np.sqrt(base_incoming_sz)/np.sqrt(incoming_sz)
                    print('scaled_curr: ', scaled_curr)
                    effective_Iscale = scaled_curr*time_step/C
                    Iscale.append(effective_Iscale)
    
    model = All_CNN_spiking(in_ch, out_sz, Iscale, time_step, taus_tr, vths_tr).to(device)
    model_test = All_CNN_spiking(in_ch, out_sz, Iscale, time_step, taus, vths).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)
    #loss_fn = SF.mse_count_loss(correct_rate=args.correct_spiking_rate, incorrect_rate=1.0-args.correct_spiking_rate)
    
    run_start = time.time()
    accs_wo_var = []
    accs_w_var = []
    best_acc = 0
    second_best_acc = 0
    for epoch in range(num_epochs):
        model.train()
        for i, (data, targets) in enumerate(iter(train_loader)):
            data = data.to(device)
            targets = targets.to(device)
            
            spk_rec = model(data, num_steps) # forward-pass
            loss_val = loss_fn(spk_rec, targets) # loss calculation
            
            optimizer.zero_grad() # null gradients
            loss_val.backward() # calculate gradients
            optimizer.step() # update weights
        model.eval()
        model_test.eval()
        print("Epoch: ", epoch)
        acc = test_accuracy(test_loader, model, num_steps)
        print("test accuracy(%) without variations: ", acc * 100)
        
        with torch.no_grad():
            model_test.load_state_dict(model.state_dict())
            for (names, params) in model_test.named_parameters():
                if params.requires_grad and ('bn' not in names) and ('downsample' not in names) and ('bias' not in names):
                    param_np = params.data.cpu().numpy()

                    max_Wt_val = 2**Wt_bits - 1
                    scale = np.float32(2*np.mean(np.abs(param_np))/np.sqrt(max_Wt_val/2))

                    param_np = np.int32(np.round(param_np/scale))
                    param_np[param_np>2**(Wt_bits-1) - 1] = 2**(Wt_bits-1) - 1
                    param_np[param_np<-2**(Wt_bits-1)] = -2**(Wt_bits-1)
                    print('max val: ', np.max(param_np))
                    print('min val: ', np.min(param_np))
                    param_q = np.float32(param_np)*scale
                    param_q = torch.from_numpy(param_q).to(device)
                    params.data = param_q
            
            acc2 = test_accuracy(test_loader, model_test, num_steps)
            print("test accuracy(%) with variations: ", acc2 * 100)
        
        if acc2>best_acc or acc2>second_best_acc:
            if acc2>best_acc:
                second_best_acc = best_acc
                best_acc = acc2
            else:
                second_best_acc = acc2
        print("best test accuracy(%) with quantization and variations: ", 100*best_acc)
        print("2nd best test accuracy(%) with quantization and variations: ", 100*second_best_acc)
        accs_wo_var.append(acc)
        accs_w_var.append(acc2)
    
    run_end = time.time()
    print("total runtime: ", run_end - run_start, "seconds")
    print("vth variation(%): ", vth_var)
    print("tau variation(%): ", tau_var)
    print("final best test accuracy(%) with quantization and variations: ", 100*best_acc)
    print("final 2nd best test accuracy(%) with quantization and variations: ", 100*second_best_acc)
    print('all accuracies with quantization and variations: ', accs_w_var)


