import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

def evaluated_prediction(pred,test_lab,lab ):
    """
    Function which evaluate quality of prdiction of linear svm, calculating TP,FP,FN,TN

    Input pred = prediction
    Input test_lab = labels
    Input num_classes(15) = number of classes
    Input lab = name of classes

    Output  res = dataframe with all these values
    """
    num_classes = len(lab)
    tp = []
    fp = []
    fn = []
    tn = []
    for i in range(num_classes):
        tp_temp = 0
        fp_temp = 0
        fn_temp = 0
        tn_temp = 0
        for j in range(len(pred)):
            if(pred[j]==i and test_lab[j]==i):
                tp_temp = tp_temp + 1
            if(pred[j]==i and test_lab[j]!=i):
                fp_temp = fp_temp + 1
            if(pred[j]!=i and test_lab[j]==i):
                fn_temp = fn_temp + 1
            if(pred[j]!=i and test_lab[j]!=i):
                tn_temp = tn_temp +1
        tp.append(tp_temp)
        fp.append(fp_temp)
        fn.append(fn_temp)
        tn.append(tn_temp)
    data = {'labels':lab , 'True positive':tp,'True negative':tn,'False positive':fp,'False negative':fn}
    res = pd.DataFrame(data, columns = ['labels','True positive','True negative','False positive','False negative'])
    return res



def build_confusion_matrix(df,pred,test_labels,lab):
    """
    Function tu construct confusion matrix
    """
    num_classes = len(lab)
    cm = np.zeros((num_classes,num_classes))
    # insert true positive on the diagonal
    for i in range(num_classes):
        cm[i,i] = df.loc[i]['True positive']
    for i in range(num_classes): # lavoro sulle classes true
        for j in range(num_classes): #lavoro su classes predicted
            temp = 0
            for k in range(len(test_labels)):
                if(test_labels[k]==i and pred[k]==j):
                    temp = temp +1
            cm[i,j]=temp
    return cm

def plot_confusion_matrix(cm, classes,string,directory,normalize=False,title='Confusion matrix'):
    """
    Function which plots confusion matrix

    Input cm = confusion matrix
    Input classes = array with class labels
    Input string = string to give name of the saved image
    Input directory = string to give directory to save the image
    Input normalize (False) = If true function will give accuracy instead of pure number
    Input Title (Confusion matrix) = title of the image


    Output : None
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize = (15,15))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if(i==j or cm[i,j] > 0.05):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            continue

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(directory +'confusion_matrix'+string+'.jpg')
