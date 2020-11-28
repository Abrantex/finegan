import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Plot losses from csv file')
    parser.add_argument('--i',required=True, dest='input_file', type=str, help='input csv file')
    parser.add_argument('--o',required=True, dest='output_folder', type=str, 
                        help='output folder')
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parse_args()

    DIR = args.output_folder + "/" + "finegan_losses"  
    if not os.path.exists( DIR ):
        os.mkdir( DIR )

    df = pd.read_csv(args.input_file)

    df.plot(kind='line',x='epoch',y=['d0_loss', 'd2_loss'])
    plt.savefig( DIR + '/' + 'd_losses.png')

    df.plot(kind='line',x='epoch',y=['errD_total'])
    plt.savefig( DIR + '/' + 'errD_total.png')

    df.plot(kind='line',x='epoch',y=['errG_total'])
    plt.savefig( DIR + '/' + 'errG_total.png')









'''
DIR = "images/fish_e_106to206"

if not os.path.exists( DIR ):
    os.mkdir( DIR )

df = pd.read_csv('/home/matheus/Downloads/models/fish_e_106to206/losses.csv')
#df2 = pd.read_csv('/home/matheus/Downloads/losses_birds.csv')

# a scatter plot comparing num_children and num_pets

df.plot(kind='line',x='epoch',y=['z_pred_loss','b_pred_loss','p_pred_loss','c_pred_loss'])
plt.savefig( DIR + '/' + 'z_pred_loss.png')

df.plot(kind='line',x='epoch',y='bg_rf_loss',color='red')
plt.savefig( DIR + '/' + 'bg_rf_loss.png')

df.plot(kind='line',x='epoch',y='bg_class_loss',color='red')
plt.savefig( DIR + '/' + 'bg_class_loss.png')

df.plot(kind='line',x='epoch',y='child_rf_loss',color='red')
plt.savefig( DIR + '/' + 'child_rf_loss.png')

df.plot(kind='line',x='epoch',y='fool_BD_loss',color='red')
plt.savefig( DIR + '/' + 'fool_BD_loss.png')

df.plot(kind='line',x='epoch',y='EG_loss',color='red')
plt.savefig( DIR + '/' + 'EG_loss.png')

df.plot(kind='line',x='epoch',y='d0_loss',color='red')
plt.savefig( DIR + '/' + 'd0_loss.png')

df.plot(kind='line',x='epoch',y='d2_loss',color='red')
plt.savefig( DIR + '/' + 'd2_loss.png')

df.plot(kind='line',x='epoch',y='bd_loss',color='red')
plt.savefig( DIR + '/' + 'bd_loss.png')

'''