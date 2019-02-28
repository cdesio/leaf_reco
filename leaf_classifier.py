#import numpy as np
#import os
from network_models_km3 import TZ_updown_classification


model = TZ_updown_classification(num_classes=4, kernel_size=3, pooling_size=3)

model.summary()