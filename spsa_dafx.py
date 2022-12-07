# Custom gradient example with a batch of vectors (signal) and vectors (of parameter) arguments
# Pytorch lightning version
import librosa
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import Dataset, DataLoader
from wrappers.generic_dafx_wrapper import GenericDAFXWrapper
from pedalboard.pedalboard import load_plugin

# ==== Hyperparameters ====
SEED = 0
NUM_SAMPLES = 10_000
SAMPLE_RATE = 20
EPOCHS = 10
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = 'cuda'
TRAINER_DEVICE = 'gpu' if DEVICE == 'cuda' else 'cpu'

# ==== Set seeds ====
pl.seed_everything(SEED)
np.random.seed(SEED)

# ==== VST Setup ====
DAFX = "mda Combo"
DAFX_FILE = "./dafx/mda.vst3"
PARAMS = ['hpf_freq', 'hpf_reso', 'output_db']

dafx_obj = load_plugin(path_to_plugin_file=DAFX_FILE, plugin_name=DAFX)
VST = GenericDAFXWrapper(dafx_obj, sample_rate=SAMPLE_RATE, param_names=PARAMS)
TARGET_PARAMS = torch.sigmoid(torch.Tensor(np.random.randn(VST.get_num_params())))

# ==== Dummy DAFX ====
def dafx_gain(signal, params):
    """
    Dummy DAFX
    Applies signal processing operator on signal as function of parameters
    """

    signal = signal.cpu().detach().numpy()
    params = params.cpu().detach().numpy()

    VST.apply_normalised_parameter_vector(params)
    effected = VST.process_mono_as_stereo(signal)
    return torch.Tensor(effected)


# ==== Utility function for SPSA ====
def rademacher(size):
    m = torch.distributions.binomial.Binomial(1, 0.5)
    x = m.sample(size)
    x[x == 0] = -1
    return x


# ==== Normalised to dB utility ====
def db(x):
    return 20 * np.log10(np.sqrt(np.mean(np.square(x))))


# ==== Dataset synthesis ====
def get_data(num_samples, sample_rate):
    # Synthesize random audio signals with random gains, predict normalized signals
    signals = np.random.randn(num_samples, sample_rate)
    x_train = signals.copy()
    y_train = 10 * signals.copy()

    VST.apply_normalised_parameter_vector(TARGET_PARAMS)

    for i in range(signals.shape[0]):
        stereo_out = VST.process_mono_as_stereo(signals[i, :])
        y_train[i, :] = librosa.to_mono(stereo_out)

    return torch.Tensor(x_train).float(), torch.Tensor(y_train).float()


# ==== Print utility ====
def stringify(a, i):
    return str(round(db(a[i, :]), 6)).ljust(20)


# ==== Custom SPSA numerical gradient ====
class SPSABatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        z = []
        for i in range(x.size()[0]):
            z.append(dafx_gain(x[i], y[i]).to(DEVICE))

        return torch.stack(z)

    @staticmethod
    def backward(ctx, upstream_grad):
        epsilon = 0.01
        x, y = ctx.saved_tensors

        def _grad(dye, xe, ye):
            print("-"*30)
            print("BACKWARDS!")
            print("-"*30)

            # Grad w.r.t x
            delta_Kx = rademacher(xe.shape)

            signal_plus = xe + (epsilon * delta_Kx)
            Jx_plus = dafx_gain(signal_plus, ye)

            signal_minus = xe - (epsilon * delta_Kx)
            Jx_minus = dafx_gain(signal_minus, ye)
            d_dx = (Jx_plus - Jx_minus) / (2.0 * epsilon)

            # Grad w.r.t y
            delta_Ky = rademacher(ye.shape)

            params_plus = ye + (epsilon * delta_Ky)
            Jy_plus = dafx_gain(xe, params_plus)

            params_minus = ye - (epsilon * delta_Ky)
            Jy_minus = dafx_gain(xe, params_minus)

            grad_param = Jy_plus - Jy_minus

            downstream_dy = torch.zeros_like(ye)

            # perturb one parameter at a time and measure output
            for i in range(ye.size()[0]):
                d_dy = grad_param / (2.0 * epsilon * delta_Ky[i])
                # add entry to output jacobian
                downstream_dy[i] = torch.dot(dye, d_dy)

            downstream_dx = d_dx * dye

            return downstream_dx.to(DEVICE), downstream_dy.to(DEVICE)

        dy1 = []
        dy2 = []

        for i in range(upstream_grad.size()[0]):
            vecJxe, vecJye = _grad(upstream_grad[i].cpu().detach(),
                                   x[i].cpu().detach(),
                                   y[i].cpu().detach())
            dy1.append(vecJxe)
            dy2.append(vecJye)

        return torch.stack(dy1), torch.stack(dy2)


# ==== SPSA with DAFX ====
class DAFXLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.spsa = SPSABatch()

    def forward(self, inputs):
        signal = inputs[0]
        params = inputs[1]
        return self.spsa.apply(signal, params)


# ==== Dataset class ====
class TrainDataset(Dataset):
    def __init__(self, train, targets):
        self.train = train
        self.targets = targets

    def __len__(self):
        return len(self.train)

    def __getitem__(self, idx):
        return self.train[idx], self.targets[idx]


# ==== Very simple feed-forward NN to predict parameter setting and output effected audio ====
class SimpleNN(pl.LightningModule):
    def __init__(self, time_samples, num_params):
        super(SimpleNN, self).__init__()

        self.dense1 = nn.Sequential(
            nn.Linear(time_samples, 32),
            nn.ReLU()
        )

        self.dense2 = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU()
        )

        self.dense3 = nn.Sequential(
            nn.Linear(16, num_params),
        )
        # x = tf.math.reduce_mean(x, axis=1)

        self.dafx = DAFXLayer()

    def forward(self, x):
        audio = x.clone()
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = torch.sigmoid(x)

        dafx_output = self.dafx([audio, x])

        return dafx_output

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.mse_loss(y_pred, y)

        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def train_dataloader(self):
        x_train, y_train = get_data(NUM_SAMPLES, SAMPLE_RATE)
        train_data = TrainDataset(x_train, y_train)
        train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=16)
        return train_loader


if __name__ == "__main__":
    # Logger
    # wandblogger = WandbLogger(name='Adam-64-0.001', project='SPSALightning')
    early_stopping = EarlyStopping(monitor='train_loss')

    # Train
    trainer = Trainer(max_epochs=EPOCHS,
                      fast_dev_run=False,
                      accelerator=TRAINER_DEVICE,
                      devices=1,
                      logger=None,
                      callbacks=[early_stopping]
                      )
    model = SimpleNN(SAMPLE_RATE, VST.get_num_params())

    print(model)

    trainer.fit(model)

    # Test
    x_train, y_train = get_data(25, 20)
    y_pred = model(x_train)

    a = x_train.detach().numpy()
    b = y_pred.detach().cpu().numpy()
    c = y_train.numpy()

    print('x_train'.ljust(20), 'y_pred'.ljust(20), 'y_true'.ljust(20))
    print('-' * 50)
    for i in range(len(x_train)):
        print(stringify(a, i), stringify(b, i), stringify(c, i))

    print("Final VST params", list(VST.get_current_normalised_param_settings().values()))
    print("Target VST params", [round(i, 3) for i in TARGET_PARAMS.detach().tolist()])
