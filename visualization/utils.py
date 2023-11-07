import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

def plot_matrix(matrix):
    df_cm = pd.DataFrame(matrix, range(len(matrix)), range(len(matrix)))
    # plt.figure(figsize=(10,7))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size
    plt.show()
