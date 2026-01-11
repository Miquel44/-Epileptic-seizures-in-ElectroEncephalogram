import torch
import torch.nn as nn
import torch.nn.functional as F

from config import NUM_CHANNELS, WINDOW_SIZE, NUM_CLASSES


class ChannelFusionCNN(nn.Module):
    """
    CNN con fusión de canales para detección de convulsiones.
    Basado en el enfoque del paper de Channel Fusion.
    """

    def __init__(self, num_channels: int = NUM_CHANNELS,
                 window_size: int = WINDOW_SIZE,
                 num_classes: int = NUM_CLASSES):
        super(ChannelFusionCNN, self).__init__()

        # Bloque 1: Convolución sobre canales y tiempo
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 5), padding=(1, 2))
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        # Bloque 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 5), padding=(1, 2))
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        # Bloque 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))

        # Calcular tamaño después de convoluciones
        self._calculate_fc_input_size(num_channels, window_size)

        # Capas fully connected
        self.fc1 = nn.Linear(self.fc_input_size, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def _calculate_fc_input_size(self, num_channels: int, window_size: int):
        """Calcula el tamaño de entrada para la capa FC."""
        h, w = num_channels, window_size
        for _ in range(3):  # 3 capas de pooling
            h = h // 2
            w = w // 2
        self.fc_input_size = 128 * max(1, h) * max(1, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Bloque 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))

        # Bloque 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))

        # Bloque 3
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        # Flatten y clasificación
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class ChannelFusionLSTM(nn.Module):
    """Modelo híbrido CNN-LSTM para fusión de canales."""

    def __init__(self, num_channels: int = NUM_CHANNELS,
                 num_classes: int = NUM_CLASSES):
        super(ChannelFusionLSTM, self).__init__()

        # CNN para extracción de características espaciales
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(num_channels, 5), padding=(0, 2))
        self.bn1 = nn.BatchNorm2d(32)

        # LSTM para características temporales
        self.lstm = nn.LSTM(input_size=32, hidden_size=64,
                            num_layers=2, batch_first=True,
                            dropout=0.3, bidirectional=True)

        # Clasificador
        self.fc = nn.Linear(128, num_classes)  # 128 = 64*2 (bidirectional)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN: fusión de canales
        x = F.relu(self.bn1(self.conv1(x)))  # (batch, 32, 1, time)
        x = x.squeeze(2)  # (batch, 32, time)
        x = x.permute(0, 2, 1)  # (batch, time, 32)

        # LSTM
        x, _ = self.lstm(x)  # (batch, time, 128)
        x = x[:, -1, :]  # Último timestep

        # Clasificación
        x = self.fc(x)
        return x
