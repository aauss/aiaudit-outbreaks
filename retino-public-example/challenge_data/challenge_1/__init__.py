install("pandas")
install("numpy")
install("python-csv")
install("sklearn")

import pandas as pd
import numpy as np
import csv
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from .main import evaluate
