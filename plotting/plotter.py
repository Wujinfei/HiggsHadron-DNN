import numpy as np
import pandas
import os
import sklearn
import subprocess
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

class plotter(object):

    def __init__(self):
        self.separations_categories = []
        self.output_directory = ''
        #self.bin_edges_low_high = np.array([0.,0.0625,0.125,0.1875,0.25,0.3125,0.375,0.4375,0.5,0.5625,0.6125,0.6875,0.75,0.8125,0.875,0.9375,1.0])
        self.nbins = np.linspace(0.0,1.0,num=50)
        w, h = 5, 5
        self.yscores_train_categorised = [[0 for x in range(w)] for y in range(h)]
        self.yscores_test_categorised = [[0 for x in range(w)] for y in range(h)]
        self.yscores_train_non_categorised = [[0 for x in range(w)] for y in range(h)]
        self.yscores_test_non_categorised = [[0 for x in range(w)] for y in range(h)]
        self.plots_directory = ''
        pass

    def save_plots(self, dir='plots/', filename=''):
        self.check_dir(dir)
        filepath = os.path.join(dir,filename)
        self.fig.savefig(filepath)
        return self.fig

    def check_dir(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

    def plot_training_progress_acc(self, histories, labels):
        self.fig, self.ax1 = plt.subplots(ncols=1, figsize=(10,10))
        for i in range(len(histories)):
            history1 = histories[i]
            label_name_train = '%s train' % (labels[i])
            label_name_test = '%s test' % (labels[i])
            plt.plot(history1.history['loss'], label=label_name_train)
            plt.plot(history1.history['val_loss'], label=label_name_test)

        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')
        acc_title = 'plots/DNN_accuracy_wrt_epoch.pdf'
        plt.tight_layout()
        return

    def correlation_matrix(self, data, **kwds):

        #iloc[<row selection>,<column selection>]
        self.data = data.iloc[:, :-4]
        self.labels = self.data.corr(**kwds).columns.values
        self.fig, self.ax1 = plt.subplots(ncols=1, figsize=(20,20))
        opts = {"annot" : True, "ax" : self.ax1, "vmin" : 0, "vmax" : 1*100, "annot_kws" : {"size":8}, "cmap" : plt.get_cmap("Blues",20), 'fmt' : '0.2f',}
        self.ax1.set_title("Correlations")
        sns.heatmap(self.data.corr(method='spearman')*100, **opts)
        for ax in (self.ax1,):
            # Shift tick location to bin centre
            ax.set_xticks(np.arange(len(self.labels))+0.5, minor=False)
            ax.set_yticks(np.arange(len(self.labels))+0.5, minor=False)
            ax.set_xticklabels(self.labels, minor=False, ha='right', rotation=45)
            #ax.set_yticklabels(np.flipud(self.labels), minor=False, rotation=45)
            ax.set_yticklabels(self.labels, minor=False, rotation=45)

        plt.tight_layout()

        return

    def conf_matrix(self, y_true, y_predicted, EventWeights_, norm=' '):
        y_true = pandas.Series(y_true, name='truth')
        y_predicted = pandas.Series(y_predicted, name='prediction')
        EventWeights_ = pandas.Series(EventWeights_, name='eventweights')
        if norm == 'index':
            self.matrix = pandas.crosstab(y_true,y_predicted,EventWeights_,aggfunc=sum,normalize='index')
            vmax = 1
        elif norm == 'columns':
            self.matrix = pandas.crosstab(y_true,y_predicted,EventWeights_,aggfunc=sum,normalize='columns')
            vmax = 1
        else:
            self.matrix = pandas.crosstab(y_true,y_predicted,EventWeights_,aggfunc=sum)
            vmax = 150

        self.labelsx = self.matrix.columns
        self.labelsy = self.matrix.index
        self.fig, self.ax1 = plt.subplots(ncols=1, figsize=(10,10))
        plt.rcParams.update({'font.size': 22})
        self.ax1.set_title("Confusion Matrix", fontsize=18)
        opts = {"annot" : True, "ax" : self.ax1, "vmin" : 0, "vmax" : vmax, "annot_kws" : {"size":18}, "cmap" : plt.get_cmap("Reds",20), 'fmt' : '0.2f',}
        sns.set(font_scale=2.4)
        sns.heatmap(self.matrix, **opts)
        label_dict = {
            0 : 'ttH',
            1 : 'Other',
            2 : 'ttW',
            3 : 'tHQ',
            4 : 'newphysics'
        }
        for ax in (self.ax1,):
            #Shift tick location to bin centre
            ax.set_xticks(np.arange(len(self.labelsx))+0.5, minor=False)
            ax.set_yticks(np.arange(len(self.labelsy))+0.5, minor=False)
            new_xlabel = []
            new_ylabel = []
            for xlabel in self.labelsx:
                new_xlabel.append(label_dict.get(xlabel))
            for ylabel in self.labelsy:
                new_ylabel.append(label_dict.get(ylabel))
            ax.set_xticklabels(new_xlabel, minor=False, ha='right', rotation=45, fontsize=18)
            ax.set_yticklabels(new_ylabel, minor=False, rotation=45, fontsize=18)
        plt.tight_layout()
        return


    def ROC_sklearn(self, original_encoded_train_Y, result_probs_train, original_encoded_test_Y, result_probs_test, encoded_signal, pltname=''):

        #SKLearn ROC and AUC
        self.fig, self.ax1 = plt.subplots(ncols=1, figsize=(10,10))

        # Set value in list to 1 for signal and 0 for any background.
        SorB_class_train = []
        SorB_class_test = []
        output_probs_train = []
        output_probs_test = []
        # Loop over all training events
        for i in range(0,len(original_encoded_train_Y)):
            # If training events truth value is target for the node assigned as signal by the variable encoded_signal append a 1
            # else assign as background and append a 0.
            if original_encoded_train_Y[i] == encoded_signal:
                SorB_class_train.append(1)
            else:
                SorB_class_train.append(0)
            # For ith event, get the probability that this event is from the signal process
            output_probs_train.append(result_probs_train[i][encoded_signal])
        # Loop over all testing events
        for i in range(0,len(original_encoded_test_Y)):
            if original_encoded_test_Y[i] == encoded_signal:
                SorB_class_test.append(1)
            else:
                SorB_class_test.append(0)
            output_probs_test.append(result_probs_test[i][encoded_signal])

        if len(original_encoded_test_Y) == 0:
            labels = ['SR applied']
        else:
            labels = ['TR train','TR test']

        if len(SorB_class_train) > 0:
            # Create ROC curve - scan across the node distribution and calculate the true and false positive rate for given thresholds.
            fpr, tpr, thresholds = roc_curve(SorB_class_train, output_probs_train)
            auc_train_node_score = roc_auc_score(SorB_class_train, output_probs_train)
            # Plot the roc curve for the model
            # Interpolate between points of fpr and tpr on graph to get curve
            plt.plot(fpr, tpr, marker='.', markersize=8, label='%s (area = %0.2f)' % (labels[0],auc_train_node_score))

        if len(SorB_class_test) > 0:
            fpr, tpr, thresholds = roc_curve(SorB_class_test, output_probs_test)
            auc_test_node_score = roc_auc_score(SorB_class_test, output_probs_test)
            # Plot the roc curve for the model
            plt.plot(fpr, tpr, marker='.', markersize=8, label='%s (area = %0.2f)' % (labels[1],auc_test_node_score))

        plt.plot([0, 1], [0, 1], linestyle='--', markersize=8,)
        plt.rcParams.update({'font.size': 22})
        self.ax1.set_title(pltname, fontsize=18)
        plt.legend(loc="lower right")
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.tight_layout()
        # save the plot
        save_name = pltname
        self.save_plots(dir=self.plots_directory, filename=save_name)
        return

    def GetSeparation(self, hist_sig, hist_bckg):

        # compute "separation" defined as
        # <s2> = (1/2) Int_-oo..+oo { (S(x) - B(x))^2/(S(x) + B(x)) dx }
        separation = 0;
        # sanity check: signal and background histograms must have same number of bins and same limits
        if len(hist_sig) != len(hist_bckg):
            print('Number of bins different for sig. and bckg')

        nBins = len(hist_sig)
        #dX = (hist_sig.GetXaxis().GetXmax() - hist_sig.GetXaxis().GetXmin()) / len(hist_sig)
        nS = np.sum(hist_sig)#*dX
        nB = np.sum(hist_bckg)#*dX

        if nS == 0:
            print('WARNING: no signal')
        if nB == 0:
            print('WARNING: no bckg')

        for i in range(1,nBins):
            sig_bin_norm = hist_sig[i]/nS
            bckg_bin_norm = hist_bckg[i]/nB
            # Separation:
            if(sig_bin_norm+bckg_bin_norm > 0):
                separation += 0.5 * ((sig_bin_norm - bckg_bin_norm) * (sig_bin_norm - bckg_bin_norm)) / (sig_bin_norm + bckg_bin_norm)
        #separation *= dX
        return separation


    def draw_category_overfitting_plot(self, y_scores_train, y_scores_test, plot_info):
        labels = plot_info[0]
        colours = plot_info[1]
        data_type = plot_info[2]
        plots_dir = plot_info[3]
        node_name = plot_info[4]
        plot_title = plot_info[5]
        name = list(filter(str.isalnum, str(data_type).split(".")[-1]))
        self.fig, self.ax = plt.subplots(figsize=(8,6))
        self.ax.grid(which='major', linestyle='-', linewidth='0.2', color='gray')
        self.ax.set_facecolor('white')

        bin_edges_low_high = np.array([0.,0.0625,0.125,0.1875,0.25,0.3125,0.375,0.4375,0.5,0.5625,0.6125,0.675,0.7375,0.8,0.8625,0.9375,1.0])

        for index in range(len(y_scores_train)):
            y_train = y_scores_train[index]
            y_test = y_scores_test[index]
            label = labels[index]
            colour = colours[index]

            trainlabel = label + ' train'
            width = np.diff(bin_edges_low_high)
            histo_train_, bin_edges = np.histogram(y_train, bins=bin_edges_low_high)
            bincenters = 0.5*(bin_edges[1:]+bin_edges[:-1])
            dx_scale_train =(bin_edges[len(bin_edges)-1] - bin_edges[0]) / (len(bin_edges)-1)
            histo_train_ = histo_train_ / np.sum(histo_train_, dtype=np.float32) / dx_scale_train
            plt.bar(bincenters, histo_train_, width=width, color=colour, edgecolor=colour, alpha=0.5, label=trainlabel)

            if index == 0:
                histo_train_ttH = histo_train_ / np.sum(histo_train_, dtype=np.float32)
            if index == 1:
                histo_train_Other = histo_train_ / np.sum(histo_train_, dtype=np.float32)
            if index == 2:
                histo_train_ttW = histo_train_ / np.sum(histo_train_, dtype=np.float32)
            if index == 3:
                histo_train_tHQ = histo_train_ / np.sum(histo_train_, dtype=np.float32)
            if index == 4:
                histo_train_newphysics = histo_train_ / np.sum(histo_train_, dtype=np.float32)
            testlabel = label + ' test'
            histo_test_, bin_edges = np.histogram(y_test, bins=bin_edges_low_high)
            dx_scale_test =(bin_edges[len(bin_edges)-1] - bin_edges[0]) / (len(bin_edges)-1)
            bincenters = 0.5*(bin_edges[1:]+bin_edges[:-1])

            if np.sum(histo_test_, dtype=np.float32) <= 0 :
                histo_test_ = histo_test_
                err = 0
                plt.errorbar(bincenters, histo_test_, yerr=err, fmt='o', c=colour, label=testlabel)
                if index == 0:
                    histo_test_ttH = histo_test_
                if index == 1:
                    histo_test_Other = histo_test_
                if index == 2:
                    histo_test_ttW = histo_test_
                if index == 3:
                    histo_test_tHQ = histo_test_
                if index == 4:
                    histo_test_newphysics = histo_test_

            else:
                err = np.sqrt(histo_test_/np.sum(histo_test_, dtype=np.float32))
                histo_test_ = histo_test_ / np.sum(histo_test_, dtype=np.float32) / dx_scale_test
                plt.errorbar(bincenters, histo_test_, yerr=err, fmt='o', c=colour, label=testlabel)
                if index == 0:
                    histo_test_ttH = histo_test_ / np.sum(histo_test_, dtype=np.float32)
                if index == 1:
                    histo_test_Other = histo_test_ / np.sum(histo_test_, dtype=np.float32)
                if index == 2:
                    histo_test_ttW = histo_test_ / np.sum(histo_test_, dtype=np.float32)
                if index == 3:
                    histo_test_tHQ = histo_test_ / np.sum(histo_test_, dtype=np.float32)
                if index == 4:
                    histo_test_newphysics = histo_test_ / np.sum(histo_test_, dtype=np.float32)
        if 'ttH' in node_name:
            train_ttHvOtherSep = "{0:.5g}".format(self.GetSeparation(histo_train_ttH,histo_train_Other))
            train_ttHvttWSep = "{0:.5g}".format(self.GetSeparation(histo_train_ttH,histo_train_ttW))
            train_ttHvtHQSep = "{0:.5g}".format(self.GetSeparation(histo_train_ttH,histo_train_tHQ))
            train_ttHvnewphysicsSep = "{0:.5g}".format(self.GetSeparation(histo_train_ttH,histo_train_newphysics))

            test_ttHvOtherSep = "{0:.5g}".format(self.GetSeparation(histo_test_ttH,histo_test_Other))
            test_ttHvttWSep = "{0:.5g}".format(self.GetSeparation(histo_test_ttH,histo_test_ttW))
            test_ttHvtHQSep = "{0:.5g}".format(self.GetSeparation(histo_test_ttH,histo_test_tHQ))
            test_ttHvnewphysicsSep = "{0:.5g}".format(self.GetSeparation(histo_test_ttH,histo_test_newphysics))

            ttH_v_Other_train_sep = 'ttH vs Other train Sep.: %s' % ( train_ttHvOtherSep )
            self.ax.annotate(ttH_v_Other_train_sep,  xy=(0.7, 2.5), xytext=(0.7, 2.5), fontsize=9)
            ttH_v_ttW_train_sep = 'ttH vs ttW train Sep.: %s' % ( train_ttHvttWSep )
            self.ax.annotate(ttH_v_ttW_train_sep,  xy=(0.7, 2.25), xytext=(0.7, 2.25), fontsize=9)
            ttH_v_tHQ_train_sep = 'ttH vs tHQ train Sep.: %s' % ( train_ttHvtHQSep )
            self.ax.annotate(ttH_v_tHQ_train_sep,  xy=(0.7, 1.75), xytext=(0.7, 1.75), fontsize=9)
            ttH_v_newphysics_train_sep = 'ttH vs newphysics train Sep.: %s' % ( train_ttHvnewphysicsSep )
            self.ax.annotate(ttH_v_newphysics_train_sep,  xy=(0.7, 1.75), xytext=(0.7, 1.75), fontsize=9)


            ttH_v_Other_test_sep = 'ttH vs Other test Sep.: %s' % ( test_ttHvOtherSep )
            self.ax.annotate(ttH_v_Other_test_sep,  xy=(0.7, 1.75), xytext=(0.7, 1.5), fontsize=9)
            ttH_v_ttW_test_sep = 'ttH vs ttW test Sep.: %s' % ( test_ttHvttWSep )
            self.ax.annotate(ttH_v_ttW_test_sep,  xy=(0.7, 1.5), xytext=(0.7, 1.25), fontsize=9)
            ttH_v_tHQ_test_sep = 'ttH vs tHQ test Sep.: %s' % ( test_ttHvtHQSep )
            self.ax.annotate(ttH_v_tHQ_test_sep,  xy=(0.7, 1.), xytext=(0.7, .75), fontsize=9)
            ttH_v_newphysics_test_sep = 'ttH vs newphysics test Sep.: %s' % ( test_ttHvnewphysicsSep )
            self.ax.annotate(ttH_v_newphysics_test_sep,  xy=(0.7, 1.), xytext=(0.7, .75), fontsize=9)
            separations_forTable = r'''\textbackslash & %s & %s & %s & %s''' % (test_ttHvOtherSep, test_ttHvttWSep, test_ttHvtHQSep, test_ttHvnewphysicsSep)
        if 'Other' in node_name:
            train_OthervttH = "{0:.5g}".format(self.GetSeparation(histo_train_Other,histo_train_ttH))
            train_OthervttW = "{0:.5g}".format(self.GetSeparation(histo_train_Other,histo_train_ttW))
            train_OthervtHQ = "{0:.5g}".format(self.GetSeparation(histo_train_Other,histo_train_tHQ))
            train_Othervnewphysics = "{0:.5g}".format(self.GetSeparation(histo_train_Other,histo_train_newphysics))

            test_OthervttH = "{0:.5g}".format(self.GetSeparation(histo_test_Other,histo_test_ttH))
            test_OthervttW = "{0:.5g}".format(self.GetSeparation(histo_test_Other,histo_test_ttW))
            test_OthervtHQ = "{0:.5g}".format(self.GetSeparation(histo_test_Other,histo_test_tHQ))
            test_Othervnewphysics = "{0:.5g}".format(self.GetSeparation(histo_test_Other,histo_test_newphysics))

            Other_v_ttH_train_sep = 'Other vs ttH train Sep.: %s' % ( train_OthervttH )
            self.ax.annotate(Other_v_ttH_train_sep,  xy=(0.7, 2.5), xytext=(0.7, 2.5), fontsize=9)
            Other_v_ttW_train_sep = 'Other vs ttW train Sep.: %s' % ( train_OthervttW )
            self.ax.annotate(Other_v_ttW_train_sep,  xy=(0.7, 2.25), xytext=(0.7, 2.25), fontsize=9)
            Other_v_tHQ_train_sep = 'Other vs tHQ train Sep.: %s' % ( train_OthervtHQ )
            self.ax.annotate(Other_v_tHQ_train_sep,  xy=(0.7, 1.75), xytext=(0.7, 1.75), fontsize=9)
            Other_v_newphysics_train_sep = 'Other vs newphysics train Sep.: %s' % ( train_Othervnewphysics )
            self.ax.annotate(Other_v_newphysics_train_sep,  xy=(0.7, 1.75), xytext=(0.7, 1.75), fontsize=9)


            Other_v_ttH_test_sep = 'Other vs ttH test Sep.: %s' % ( test_OthervttH )
            self.ax.annotate(Other_v_ttH_test_sep,  xy=(0.7, 1.75), xytext=(0.7, 1.75), fontsize=9)
            Other_v_ttW_test_sep = 'Other vs ttW test Sep.: %s' % ( test_OthervttW )
            self.ax.annotate(Other_v_ttW_test_sep,  xy=(0.7, 1.5), xytext=(0.7, 1.5), fontsize=9)
            Other_v_tHQ_test_sep = 'Other vs tHQ test Sep.: %s' % ( test_OthervtHQ )
            self.ax.annotate(Other_v_tHQ_test_sep,  xy=(0.7, 1.), xytext=(0.7, 1.), fontsize=9)
            Other_v_newphysics_test_sep = 'Other vs newphysics test Sep.: %s' % ( test_Othervnewphysics )
            self.ax.annotate(Other_v_newphysics_test_sep,  xy=(0.7, 1.), xytext=(0.7, 1.), fontsize=9)


            separations_forTable = r'''%s & \textbackslash & %s & %s & %s''' % (test_OthervttH,test_OthervttW,Other_v_tHQ_test_sep, Other_v_newphysics_test_sep)

        if 'ttW' in node_name:
            train_ttWvttH = "{0:.5g}".format(self.GetSeparation(histo_train_ttW,histo_train_ttH))
            train_ttWvOther = "{0:.5g}".format(self.GetSeparation(histo_train_ttW,histo_train_Other))
            train_ttWvtHQ = "{0:.5g}".format(self.GetSeparation(histo_train_ttW,histo_train_tHQ))
            train_ttWvnewphysics = "{0:.5g}".format(self.GetSeparation(histo_train_ttW,histo_train_newphysics))

            test_ttWvttH = "{0:.5g}".format(self.GetSeparation(histo_test_ttW,histo_test_ttH))
            test_ttWvOther = "{0:.5g}".format(self.GetSeparation(histo_test_ttW,histo_test_Other))
            test_ttWvtHQ = "{0:.5g}".format(self.GetSeparation(histo_test_ttW,histo_test_tHQ))
            test_ttWvnewphysics = "{0:.5g}".format(self.GetSeparation(histo_test_ttW,histo_test_newphysics))

            ttW_v_ttH_train_sep = 'ttW vs ttH train Sep.: %s' % ( train_ttWvttH )
            self.ax.annotate(ttW_v_ttH_train_sep,  xy=(0.7, 2.5), xytext=(0.7, 2.5), fontsize=9)
            ttW_v_Other_train_sep = 'ttW vs Other train Sep.: %s' % ( train_ttWvOther )
            self.ax.annotate(ttW_v_Other_train_sep,  xy=(0.7, 2.25), xytext=(0.7, 2.25), fontsize=9)
            ttW_v_tHQ_train_sep = 'ttW vs tHQ train Sep.: %s' % ( train_ttWvtHQ )
            self.ax.annotate(ttW_v_tHQ_train_sep,  xy=(0.7, 1.75), xytext=(0.7, 1.75), fontsize=9)
            ttW_v_newphysics_train_sep = 'ttW vs newphysics train Sep.: %s' % ( train_ttWvnewphysics )
            self.ax.annotate(ttW_v_newphysics_train_sep,  xy=(0.7, 1.75), xytext=(0.7, 1.75), fontsize=9)

            ttW_v_ttH_test_sep = 'ttW vs ttH test Sep.: %s' % ( test_ttWvttH )
            self.ax.annotate(ttW_v_ttH_test_sep,  xy=(0.7, 1.75), xytext=(0.7, 1.75), fontsize=9)
            ttW_v_Other_test_sep = 'ttW vs Other test Sep.: %s' % ( test_ttWvOther )
            self.ax.annotate(ttW_v_Other_test_sep,  xy=(0.7, 1.5), xytext=(0.7, 1.5), fontsize=9)
            ttW_v_tHQ_test_sep = 'ttW vs tHQ test Sep.: %s' % ( test_ttWvtHQ )
            self.ax.annotate(ttW_v_tHQ_test_sep,  xy=(0.7, 1.), xytext=(0.7, 1.), fontsize=9)
            ttW_v_newphysics_test_sep = 'ttW vs newphysics test Sep.: %s' % ( test_ttWvnewphysics )
            self.ax.annotate(ttW_v_newphysics_test_sep,  xy=(0.7, 1.), xytext=(0.7, 1.), fontsize=9)

            separations_forTable = r'''%s & %s & \textbackslash & %s & %s''' % (test_ttWvttH,test_ttWvOther,ttW_v_tHQ_test_sep, ttW_v_newphysics_test_sep)

        if 'tHQ' in node_name:
            train_tHQvttH = "{0:.5g}".format(self.GetSeparation(histo_train_tHQ,histo_train_ttH))
            train_tHQvOther = "{0:.5g}".format(self.GetSeparation(histo_train_tHQ,histo_train_Other))
            train_tHQvttW = "{0:.5g}".format(self.GetSeparation(histo_train_tHQ,histo_train_ttW))
            train_tHQvnewphysics = "{0:.5g}".format(self.GetSeparation(histo_train_tHQ,histo_train_newphysics))

            test_tHQvttH = "{0:.5g}".format(self.GetSeparation(histo_test_tHQ,histo_test_ttH))
            test_tHQvOther = "{0:.5g}".format(self.GetSeparation(histo_test_tHQ,histo_test_Other))
            test_tHQvttW = "{0:.5g}".format(self.GetSeparation(histo_test_tHQ,histo_test_ttW))
            test_tHQvnewphysics = "{0:.5g}".format(self.GetSeparation(histo_test_tHQ,histo_test_newphysics))

            tHQ_v_ttH_train_sep = 'tHQ vs ttH train Sep.: %s' % ( train_tHQvttH )
            self.ax.annotate(tHQ_v_ttH_train_sep,  xy=(0.7, 2.5), xytext=(0.7, 2.5), fontsize=9)
            tHQ_v_Other_train_sep = 'tHQ vs Other train Sep.: %s' % ( train_tHQvOther )
            self.ax.annotate(tHQ_v_Other_train_sep,  xy=(0.7, 2.25), xytext=(0.7, 2.25), fontsize=9)
            tHQ_v_ttW_train_sep = 'tHQ vs ttW train Sep.: %s' % ( train_tHQvttW )
            self.ax.annotate(tHQ_v_ttW_train_sep,  xy=(0.7, 2.), xytext=(0.7, 2.), fontsize=9)
            tHQ_v_newphysics_train_sep = 'tHQ vs newphysics train Sep.: %s' % ( train_tHQvnewphysics )
            self.ax.annotate(tHQ_v_newphysics_train_sep,  xy=(0.7, 2.), xytext=(0.7, 2.), fontsize=9)

            tHQ_v_ttH_test_sep = 'tHQ vs ttH test Sep.: %s' % ( test_tHQvttH )
            self.ax.annotate(tHQ_v_ttH_test_sep,  xy=(0.7, 1.75), xytext=(0.7, 1.75), fontsize=9)
            tHQ_v_Other_test_sep = 'tHQ vs Other test Sep.: %s' % ( test_tHQvOther )
            self.ax.annotate(tHQ_v_Other_test_sep,  xy=(0.7, 1.5), xytext=(0.7, 1.5), fontsize=9)
            tHQ_v_ttW_test_sep = 'tHQ vs ttW test Sep.: %s' % ( test_tHQvttW )
            self.ax.annotate(tHQ_v_ttW_test_sep,  xy=(0.7, 1.25), xytext=(0.7, 1.25), fontsize=9)
            tHQ_v_newphysics_test_sep = 'tHQ vs newphysics test Sep.: %s' % ( test_tHQvnewphysics )
            self.ax.annotate(tHQ_v_newphysics_test_sep,  xy=(0.7, 1.25), xytext=(0.7, 1.25), fontsize=9)

            separations_forTable = r'''%s & %s & %s & %s \textbackslash ''' % (test_tHQvttH,test_tHQvOther,tHQ_v_ttW_test_sep, tHQ_v_newphysics_test_sep)
        



        if 'newphysics' in node_name:
            train_newphysicsvttH = "{0:.5g}".format(self.GetSeparation(histo_train_newphysics,histo_train_ttH))
            train_newphysicsvOther = "{0:.5g}".format(self.GetSeparation(histo_train_newphysics,histo_train_Other))
            train_newphysicsvttW = "{0:.5g}".format(self.GetSeparation(histo_train_newphysics,histo_train_ttW))
            train_newphysicsvtHQ = "{0:.5g}".format(self.GetSeparation(histo_train_newphysics,histo_train_tHQ))

            test_newphysicsvttH = "{0:.5g}".format(self.GetSeparation(histo_test_newphysics,histo_test_ttH))
            test_newphysicsvOther = "{0:.5g}".format(self.GetSeparation(histo_test_newphysics,histo_test_Other))
            test_newphysicsvttW = "{0:.5g}".format(self.GetSeparation(histo_test_newphysics,histo_test_ttW))
            test_newphysicsvtHQ = "{0:.5g}".format(self.GetSeparation(histo_test_newphysics,histo_test_tHQ))

            newphysics_v_ttH_train_sep = 'newphysics vs ttH train Sep.: %s' % ( train_newphysicsvttH )
            self.ax.annotate(newphysics_v_ttH_train_sep,  xy=(0.7, 2.5), xytext=(0.7, 2.5), fontsize=9)
            newphysics_v_Other_train_sep = 'newphysics vs Other train Sep.: %s' % ( train_newphysicsvOther )
            self.ax.annotate(newphysics_v_Other_train_sep,  xy=(0.7, 2.25), xytext=(0.7, 2.25), fontsize=9)
            newphysics_v_ttW_train_sep = 'newphysics vs ttW train Sep.: %s' % ( train_newphysicsvttW )
            self.ax.annotate(newphysics_v_ttW_train_sep,  xy=(0.7, 2.), xytext=(0.7, 2.), fontsize=9)
            newphysics_v_tHQ_train_sep = 'newphysics vs tHQ train Sep.: %s' % ( train_newphysicsvtHQ )
            self.ax.annotate(newphysics_v_tHQ_train_sep,  xy=(0.7, 2.), xytext=(0.7, 2.), fontsize=9)

            newphysics_v_ttH_test_sep = 'newphysics vs ttH test Sep.: %s' % ( test_newphysicsvttH )
            self.ax.annotate(newphysics_v_ttH_test_sep,  xy=(0.7, 1.75), xytext=(0.7, 1.75), fontsize=9)
            newphysics_v_Other_test_sep = 'newphysics vs Other test Sep.: %s' % ( test_newphysicsvOther )
            self.ax.annotate(newphysics_v_Other_test_sep,  xy=(0.7, 1.5), xytext=(0.7, 1.5), fontsize=9)
            newphysics_v_ttW_test_sep = 'newphysics vs ttW test Sep.: %s' % ( test_newphysicsvttW )
            self.ax.annotate(newphysics_v_ttW_test_sep,  xy=(0.7, 1.25), xytext=(0.7, 1.25), fontsize=9)
            newphysics_v_tHQ_test_sep = 'newphysics vs tHQ test Sep.: %s' % ( test_newphysicsvtHQ )
            self.ax.annotate(newphysics_v_tHQ_test_sep,  xy=(0.7, 1.25), xytext=(0.7, 1.25), fontsize=9)


            separations_forTable = r'''%s & %s & %s & %s \textbackslash ''' % (test_newphysicsvttH,test_newphysicsvOther,newphysics_v_ttW_test_sep, newphysics_v_tHQ_test_sep)
    

        title_ = '%s %s node' % (plot_title,node_name)
        plt.title(title_)
        label_name = 'DNN Output Score'
        plt.xlabel(label_name)
        plt.ylabel('(1/N)dN/dX')

        leg = plt.legend(loc='best', frameon=False, fancybox=False, fontsize=9)
        leg.get_frame().set_edgecolor('w')
        frame = leg.get_frame()
        frame.set_facecolor('White')

        overfitting_plot_file_name = 'overfitting_plot_%s_%s.pdf' % (node_name,plot_title)
        print('Saving : %s%s' % (plots_dir, overfitting_plot_file_name))
        self.save_plots(dir=plots_dir, filename=overfitting_plot_file_name)

        return separations_forTable


    def separation_table(self , outputdir):
        content = r'''\documentclass{article}
\begin{document}
\begin{center}
\begin{table}
\begin{tabular}{| c | c | c | c | c |} \hline
Node \textbackslash Background & ttH & Other & ttW & tHQ \\ \hline
ttH & %s \\
Other & %s \\
ttW & %s \\
tHQ & %s \\ \hline
newphysics & %s \\ \hline
\end{tabular}
\caption{Separation power on each output node. The separation is given with respect to the `signal' process the node is trained to separate (one node per row) and the background processes for that node (one background per column).}
\end{table}
\end{center}
\end{document}
'''
        table_path = os.path.join(outputdir,'separation_table')
        table_tex = table_path+'.tex'
        print('table_tex: ', table_tex)
        with open(table_tex,'w') as f:
            f.write( content % (self.separations_categories[0], self.separations_categories[1], self.separations_categories[2], self.separations_categories[3], self.separations_categories[4] ) )
        return

    def overfitting(self, estimator, Y_train, Y_test, result_probs, result_probs_test, plots_dir, train_weights, test_weights, nbins=50):

        model = estimator
        data_type = type(model)

        #Arrays to store all ttH values
        y_scores_train_ttH_sample_ttHnode = []
        y_scores_train_Other_sample_ttHnode = []
        y_scores_train_ttW_sample_ttHnode = []
        y_scores_train_tHQ_sample_ttHnode = []
        y_scores_train_newphysics_sample_ttHnode = []
        # Arrays to store ttH categorised event values
        y_scores_train_ttH_sample_ttH_categorised = []
        y_scores_train_Other_sample_ttH_categorised = []
        y_scores_train_ttW_sample_ttH_categorised = []
        y_scores_train_tHQ_sample_ttH_categorised = []
        y_scores_train_newphysics_sample_ttH_categorised = []
        # Arrays to store all Other node values
        y_scores_train_ttH_sample_Othernode = []
        y_scores_train_Other_sample_Othernode = []
        y_scores_train_ttW_sample_Othernode = []
        y_scores_train_tHQ_sample_Othernode = []
        y_scores_train_newphysics_sample_Othernode = []
        # Arrays to store Other categorised event values
        y_scores_train_ttH_sample_Other_categorised = []
        y_scores_train_Other_sample_Other_categorised = []
        y_scores_train_ttW_sample_Other_categorised = []
        y_scores_train_tHQ_sample_Other_categorised = []
        y_scores_train_newphysics_sample_Other_categorised = []
        # Arrays to store all ttW node values
        y_scores_train_ttH_sample_ttWnode = []
        y_scores_train_Other_sample_ttWnode = []
        y_scores_train_ttW_sample_ttWnode = []
        y_scores_train_tHQ_sample_ttWnode = []
        y_scores_train_newphysics_sample_ttWnode = []
        # Arrays to store ttW categorised events
        y_scores_train_ttH_sample_ttW_categorised = []
        y_scores_train_Other_sample_ttW_categorised = []
        y_scores_train_ttW_sample_ttW_categorised = []
        y_scores_train_tHQ_sample_ttW_categorised = []
        y_scores_train_newphysics_sample_ttW_categorised = []
        # Arrays to store all tHQ node values
        y_scores_train_ttH_sample_tHQnode = []
        y_scores_train_Other_sample_tHQnode = []
        y_scores_train_ttW_sample_tHQnode = []
        y_scores_train_tHQ_sample_tHQnode = []
        y_scores_train_newphysics_sample_tHQnode = []
        # Arrays to store tHQ categorised events
        y_scores_train_ttH_sample_tHQ_categorised = []
        y_scores_train_Other_sample_tHQ_categorised = []
        y_scores_train_ttW_sample_tHQ_categorised = []
        y_scores_train_tHQ_sample_tHQ_categorised = []
        y_scores_train_newphysics_sample_tHQ_categorised = []

        # Arrays to store all newphysics node values
        y_scores_train_ttH_sample_newphysicsnode = []
        y_scores_train_Other_sample_newphysicsnode = []
        y_scores_train_ttW_sample_newphysicsnode = []
        y_scores_train_tHQ_sample_newphysicsnode = []
        y_scores_train_newphysics_sample_newphysicsnode = []
        # Arrays to store newphysics categorised events
        y_scores_train_ttH_sample_newphysics_categorised = []
        y_scores_train_Other_sample_newphysics_categorised = []
        y_scores_train_ttW_sample_newphysics_categorised = []
        y_scores_train_tHQ_sample_newphysics_categorised = []
        y_scores_train_newphysics_sample_newphysics_categorised = []




        for i in range(len(result_probs)):
            train_event_weight = train_weights[i]
            if Y_train[i][0] == 1:
                y_scores_train_ttH_sample_ttHnode.append(result_probs[i][0])
                y_scores_train_ttH_sample_Othernode.append(result_probs[i][1])
                y_scores_train_ttH_sample_ttWnode.append(result_probs[i][2])
                y_scores_train_ttH_sample_tHQnode.append(result_probs[i][3])
                y_scores_train_ttH_sample_newphysicsnode.append(result_probs[i][4])
                # Get index of maximum argument.
                if np.argmax(result_probs[i]) == 0:
                    y_scores_train_ttH_sample_ttH_categorised.append(result_probs[i][0])
                if np.argmax(result_probs[i]) == 1:
                    y_scores_train_ttH_sample_Other_categorised.append(result_probs[i][1])
                if np.argmax(result_probs[i]) == 2:
                    y_scores_train_ttH_sample_ttW_categorised.append(result_probs[i][2])
                if np.argmax(result_probs[i]) == 3:
                    y_scores_train_ttH_sample_tHQ_categorised.append(result_probs[i][3])
                if np.argmax(result_probs[i]) == 4:
                    y_scores_train_ttH_sample_newphysics_categorised.append(result_probs[i][4])

            if Y_train[i][1] == 1:
                y_scores_train_Other_sample_ttHnode.append(result_probs[i][0])
                y_scores_train_Other_sample_Othernode.append(result_probs[i][1])
                y_scores_train_Other_sample_ttWnode.append(result_probs[i][2])
                y_scores_train_Other_sample_tHQnode.append(result_probs[i][3])
                y_scores_train_Other_sample_newphysicsnode.append(result_probs[i][4])
                if np.argmax(result_probs[i]) == 0:
                    y_scores_train_Other_sample_ttH_categorised.append(result_probs[i][0])
                if np.argmax(result_probs[i]) == 1:
                    y_scores_train_Other_sample_Other_categorised.append(result_probs[i][1])
                if np.argmax(result_probs[i]) == 2:
                    y_scores_train_Other_sample_ttW_categorised.append(result_probs[i][2])
                if np.argmax(result_probs[i]) == 3:
                    y_scores_train_Other_sample_tHQ_categorised.append(result_probs[i][3])
                if np.argmax(result_probs[i]) == 4:
                    y_scores_train_Other_sample_newphysics_categorised.append(result_probs[i][4])

            if Y_train[i][2] == 1:
                y_scores_train_ttW_sample_ttHnode.append(result_probs[i][0])
                y_scores_train_ttW_sample_Othernode.append(result_probs[i][1])
                y_scores_train_ttW_sample_ttWnode.append(result_probs[i][2])
                y_scores_train_ttW_sample_tHQnode.append(result_probs[i][3])
                y_scores_train_ttW_sample_newphysicsnode.append(result_probs[i][4])
                if np.argmax(result_probs[i]) == 0:
                    y_scores_train_ttW_sample_ttH_categorised.append(result_probs[i][0])
                if np.argmax(result_probs[i]) == 1:
                    y_scores_train_ttW_sample_Other_categorised.append(result_probs[i][1])
                if np.argmax(result_probs[i]) == 2:
                    y_scores_train_ttW_sample_ttW_categorised.append(result_probs[i][2])
                if np.argmax(result_probs[i]) == 3:
                    y_scores_train_ttW_sample_tHQ_categorised.append(result_probs[i][3])
                if np.argmax(result_probs[i]) == 4:
                    y_scores_train_ttW_sample_newphysics_categorised.append(result_probs[i][4])
            if Y_train[i][3] == 1:
                y_scores_train_tHQ_sample_ttHnode.append(result_probs[i][0])
                y_scores_train_tHQ_sample_Othernode.append(result_probs[i][1])
                y_scores_train_tHQ_sample_ttWnode.append(result_probs[i][2])
                y_scores_train_tHQ_sample_tHQnode.append(result_probs[i][3])
                y_scores_train_tHQ_sample_newphysicsnode.append(result_probs[i][4])
                if np.argmax(result_probs[i]) == 0:
                    y_scores_train_tHQ_sample_ttH_categorised.append(result_probs[i][0])
                if np.argmax(result_probs[i]) == 1:
                    y_scores_train_tHQ_sample_Other_categorised.append(result_probs[i][1])
                if np.argmax(result_probs[i]) == 2:
                    y_scores_train_tHQ_sample_ttW_categorised.append(result_probs[i][2])
                if np.argmax(result_probs[i]) == 3:
                    y_scores_train_tHQ_sample_tHQ_categorised.append(result_probs[i][3])
                if np.argmax(result_probs[i]) == 4:
                    y_scores_train_tHQ_sample_newphysics_categorised.append(result_probs[i][4])
            
            if Y_train[i][4] == 1:
                y_scores_train_newphysics_sample_ttHnode.append(result_probs[i][0])
                y_scores_train_newphysics_sample_Othernode.append(result_probs[i][1])
                y_scores_train_newphysics_sample_ttWnode.append(result_probs[i][2])
                y_scores_train_newphysics_sample_tHQnode.append(result_probs[i][3])
                y_scores_train_newphysics_sample_newphysicsnode.append(result_probs[i][4])
                if np.argmax(result_probs[i]) == 0:
                    y_scores_train_newphysics_sample_ttH_categorised.append(result_probs[i][0])
                if np.argmax(result_probs[i]) == 1:
                    y_scores_train_newphysics_sample_Other_categorised.append(result_probs[i][1])
                if np.argmax(result_probs[i]) == 2:
                    y_scores_train_newphysics_sample_ttW_categorised.append(result_probs[i][2])
                if np.argmax(result_probs[i]) == 3:
                    y_scores_train_newphysics_sample_tHQ_categorised.append(result_probs[i][3])
                if np.argmax(result_probs[i]) == 4:
                    y_scores_train_newphysics_sample_newphysics_categorised.append(result_probs[i][4])

           
        #Arrays to store all ttH values
        y_scores_test_ttH_sample_ttHnode = []
        y_scores_test_Other_sample_ttHnode = []
        y_scores_test_ttW_sample_ttHnode = []
        y_scores_test_tHQ_sample_ttHnode = []
        y_scores_test_newphysics_sample_ttHnode = []
        # Arrays to store ttH categorised event values
        y_scores_test_ttH_sample_ttH_categorised = []
        y_scores_test_Other_sample_ttH_categorised = []
        y_scores_test_ttW_sample_ttH_categorised = []
        y_scores_test_tHQ_sample_ttH_categorised = []
        y_scores_test_newphysics_sample_ttH_categorised = []
        # Arrays to store all Other node values
        y_scores_test_ttH_sample_Othernode = []
        y_scores_test_Other_sample_Othernode = []
        y_scores_test_ttW_sample_Othernode = []
        y_scores_test_tHQ_sample_Othernode = []
        y_scores_test_newphysics_sample_Othernode = []
        # Arrays to store Other categorised event values
        y_scores_test_ttH_sample_Other_categorised = []
        y_scores_test_Other_sample_Other_categorised = []
        y_scores_test_ttW_sample_Other_categorised = []
        y_scores_test_tHQ_sample_Other_categorised = []
        y_scores_test_newphysics_sample_Other_categorised = []
        # Arrays to store all ttW node values
        y_scores_test_ttH_sample_ttWnode = []
        y_scores_test_Other_sample_ttWnode = []
        y_scores_test_ttW_sample_ttWnode = []
        y_scores_test_tHQ_sample_ttWnode = []
        y_scores_test_newphysics_sample_ttWnode = []
        # Arrays to store ttW categorised events
        y_scores_test_ttH_sample_ttW_categorised = []
        y_scores_test_Other_sample_ttW_categorised = []
        y_scores_test_ttW_sample_ttW_categorised = []
        y_scores_test_tHQ_sample_ttW_categorised = []
        y_scores_test_newphysics_sample_ttW_categorised = []
        # Arrays to store all tHQ node values
        y_scores_test_ttH_sample_tHQnode = []
        y_scores_test_Other_sample_tHQnode = []
        y_scores_test_ttW_sample_tHQnode = []
        y_scores_test_tHQ_sample_tHQnode = []
        y_scores_test_newphysics_sample_tHQnode = []
        # Arrays to store tHQ categorised events
        y_scores_test_ttH_sample_tHQ_categorised = []
        y_scores_test_Other_sample_tHQ_categorised = []
        y_scores_test_ttW_sample_tHQ_categorised = []
        y_scores_test_tHQ_sample_tHQ_categorised = []
        y_scores_test_newphysics_sample_tHQ_categorised = []

        # Arrays to store all newphysics node values
        y_scores_test_ttH_sample_newphysicsnode = []
        y_scores_test_Other_sample_newphysicsnode = []
        y_scores_test_ttW_sample_newphysicsnode = []
        y_scores_test_tHQ_sample_newphysicsnode = []
        y_scores_test_newphysics_sample_newphysicsnode = []
        # Arrays to store tHQ categorised events
        y_scores_test_ttH_sample_newphysics_categorised = []
        y_scores_test_Other_sample_newphysics_categorised = []
        y_scores_test_ttW_sample_newphysics_categorised = []
        y_scores_test_tHQ_sample_newphysics_categorised = []
        y_scores_test_newphysics_sample_newphysics_categorised = []
 
        
        for i in range(len(result_probs_test)):
            test_event_weight = test_weights[i]
            if Y_test[i][0] == 1:
                y_scores_test_ttH_sample_ttHnode.append(result_probs_test[i][0])
                y_scores_test_ttH_sample_Othernode.append(result_probs_test[i][1])
                y_scores_test_ttH_sample_ttWnode.append(result_probs_test[i][2])
                y_scores_test_ttH_sample_tHQnode.append(result_probs_test[i][3])
                y_scores_test_ttH_sample_newphysicsnode.append(result_probs_test[i][4])
                if np.argmax(result_probs_test[i]) == 0:
                    y_scores_test_ttH_sample_ttH_categorised.append(result_probs_test[i][0])
                if np.argmax(result_probs_test[i]) == 1:
                    y_scores_test_ttH_sample_Other_categorised.append(result_probs_test[i][1])
                if np.argmax(result_probs_test[i]) == 2:
                    y_scores_test_ttH_sample_ttW_categorised.append(result_probs_test[i][2])
                if np.argmax(result_probs_test[i]) == 3:
                    y_scores_test_ttH_sample_tHQ_categorised.append(result_probs_test[i][3])
                if np.argmax(result_probs_test[i]) == 4:
                    y_scores_test_ttH_sample_newphysics_categorised.append(result_probs_test[i][4])

            if Y_test[i][1] == 1:
                y_scores_test_Other_sample_ttHnode.append(result_probs_test[i][0])
                y_scores_test_Other_sample_Othernode.append(result_probs_test[i][1])
                y_scores_test_Other_sample_ttWnode.append(result_probs_test[i][2])
                y_scores_test_Other_sample_tHQnode.append(result_probs_test[i][3])
                y_scores_test_Other_sample_newphysicsnode.append(result_probs_test[i][4])
                if np.argmax(result_probs_test[i]) == 0:
                    y_scores_test_Other_sample_ttH_categorised.append(result_probs_test[i][0])
                if np.argmax(result_probs_test[i]) == 1:
                    y_scores_test_Other_sample_Other_categorised.append(result_probs_test[i][1])
                if np.argmax(result_probs_test[i]) == 2:
                    y_scores_test_Other_sample_ttW_categorised.append(result_probs_test[i][2])
                if np.argmax(result_probs_test[i]) == 3:
                    y_scores_test_Other_sample_tHQ_categorised.append(result_probs_test[i][3])
                if np.argmax(result_probs_test[i]) == 4:
                    y_scores_test_Other_sample_newphysics_categorised.append(result_probs_test[i][4])
            if Y_test[i][2] == 1:
                y_scores_test_ttW_sample_ttHnode.append(result_probs_test[i][0])
                y_scores_test_ttW_sample_Othernode.append(result_probs_test[i][1])
                y_scores_test_ttW_sample_ttWnode.append(result_probs_test[i][2])
                y_scores_test_ttW_sample_tHQnode.append(result_probs_test[i][3])
                y_scores_test_ttW_sample_newphysicsnode.append(result_probs_test[i][4])
                if np.argmax(result_probs_test[i]) == 0:
                    y_scores_test_ttW_sample_ttH_categorised.append(result_probs_test[i][0])
                if np.argmax(result_probs_test[i]) == 1:
                    y_scores_test_ttW_sample_Other_categorised.append(result_probs_test[i][1])
                if np.argmax(result_probs_test[i]) == 2:
                    y_scores_test_ttW_sample_ttW_categorised.append(result_probs_test[i][2])
                if np.argmax(result_probs_test[i]) == 3:
                    y_scores_test_ttW_sample_tHQ_categorised.append(result_probs_test[i][3])
                if np.argmax(result_probs_test[i]) == 4:
                    y_scores_test_ttW_sample_newphysics_categorised.append(result_probs_test[i][4])

            if Y_test[i][3] == 1:
                y_scores_test_tHQ_sample_ttHnode.append(result_probs_test[i][0])
                y_scores_test_tHQ_sample_Othernode.append(result_probs_test[i][1])
                y_scores_test_tHQ_sample_ttWnode.append(result_probs_test[i][2])
                y_scores_test_tHQ_sample_tHQnode.append(result_probs_test[i][3])
                y_scores_test_tHQ_sample_newphysicsnode.append(result_probs_test[i][4])
                if np.argmax(result_probs_test[i]) == 0:
                    y_scores_test_tHQ_sample_ttH_categorised.append(result_probs_test[i][0])
                if np.argmax(result_probs_test[i]) == 1:
                    y_scores_test_tHQ_sample_Other_categorised.append(result_probs_test[i][1])
                if np.argmax(result_probs_test[i]) == 2:
                    y_scores_test_tHQ_sample_ttW_categorised.append(result_probs_test[i][2])
                if np.argmax(result_probs_test[i]) == 3:
                    y_scores_test_tHQ_sample_tHQ_categorised.append(result_probs_test[i][3])
                if np.argmax(result_probs_test[i]) == 4:
                    y_scores_test_tHQ_sample_newphysics_categorised.append(result_probs_test[i][4])

 
            if Y_test[i][4] == 1:
                y_scores_test_newphysics_sample_ttHnode.append(result_probs_test[i][0])
                y_scores_test_newphysics_sample_Othernode.append(result_probs_test[i][1])
                y_scores_test_newphysics_sample_ttWnode.append(result_probs_test[i][2])
                y_scores_test_newphysics_sample_tHQnode.append(result_probs_test[i][3])
                y_scores_test_newphysics_sample_newphysicsnode.append(result_probs_test[i][4])
                if np.argmax(result_probs_test[i]) == 0:
                    y_scores_test_newphysics_sample_ttH_categorised.append(result_probs_test[i][0])
                if np.argmax(result_probs_test[i]) == 1:
                    y_scores_test_newphysics_sample_Other_categorised.append(result_probs_test[i][1])
                if np.argmax(result_probs_test[i]) == 2:
                    y_scores_test_newphysics_sample_ttW_categorised.append(result_probs_test[i][2])
                if np.argmax(result_probs_test[i]) == 3:
                    y_scores_test_newphysics_sample_tHQ_categorised.append(result_probs_test[i][3])
                if np.argmax(result_probs_test[i]) == 4:
                    y_scores_test_newphysics_sample_newphysics_categorised.append(result_probs_test[i][4])



        # Create 2D lists (dimension 4x4) to hold max DNN discriminator values for each sample. One for train data, one for test data.
        #
        #               ttH sample | Other sample | ttW sample | ttZ sample | tHQ sample
        # ttH category
        # Other category
        # ttW category
        # ttZ category
        # tHQ category

        #w, h = 4, 4
        #y_scores_train = [[0 for x in range(w)] for y in range(h)]
        #y_scores_test = [[0 for x in range(w)] for y in range(h)]
        self.yscores_train_categorised[0] = [y_scores_train_ttH_sample_ttH_categorised, y_scores_train_Other_sample_ttH_categorised, y_scores_train_ttW_sample_ttH_categorised, y_scores_train_tHQ_sample_ttH_categorised, y_scores_train_newphysics_sample_ttH_categorised]
        self.yscores_train_categorised[1] = [y_scores_train_ttH_sample_Other_categorised, y_scores_train_Other_sample_Other_categorised, y_scores_train_ttW_sample_Other_categorised, y_scores_train_tHQ_sample_Other_categorised, y_scores_train_newphysics_sample_Other_categorised]
        self.yscores_train_categorised[2] = [y_scores_train_ttH_sample_ttW_categorised, y_scores_train_Other_sample_ttW_categorised, y_scores_train_ttW_sample_ttW_categorised, y_scores_train_tHQ_sample_ttW_categorised, y_scores_train_newphysics_sample_ttW_categorised]
        self.yscores_train_categorised[3] = [y_scores_train_ttH_sample_tHQ_categorised, y_scores_train_Other_sample_tHQ_categorised, y_scores_train_ttW_sample_tHQ_categorised, y_scores_train_tHQ_sample_tHQ_categorised, y_scores_train_newphysics_sample_tHQ_categorised]
        self.yscores_train_categorised[4] = [y_scores_train_ttH_sample_newphysics_categorised, y_scores_train_Other_sample_newphysics_categorised, y_scores_train_ttW_sample_newphysics_categorised, y_scores_train_tHQ_sample_newphysics_categorised, y_scores_train_newphysics_sample_newphysics_categorised]

        self.yscores_test_categorised[0] = [y_scores_test_ttH_sample_ttH_categorised, y_scores_test_Other_sample_ttH_categorised, y_scores_test_ttW_sample_ttH_categorised, y_scores_test_tHQ_sample_ttH_categorised, y_scores_test_newphysics_sample_ttH_categorised]
        self.yscores_test_categorised[1] = [y_scores_test_ttH_sample_Other_categorised, y_scores_test_Other_sample_Other_categorised, y_scores_test_ttW_sample_Other_categorised, y_scores_test_tHQ_sample_Other_categorised, y_scores_test_newphysics_sample_Other_categorised]
        self.yscores_test_categorised[2] = [y_scores_test_ttH_sample_ttW_categorised, y_scores_test_Other_sample_ttW_categorised, y_scores_test_ttW_sample_ttW_categorised, y_scores_test_tHQ_sample_ttW_categorised, y_scores_test_newphysics_sample_ttW_categorised]
        self.yscores_test_categorised[3] = [y_scores_test_ttH_sample_tHQ_categorised, y_scores_test_Other_sample_tHQ_categorised, y_scores_test_ttW_sample_tHQ_categorised, y_scores_test_tHQ_sample_tHQ_categorised, y_scores_test_newphysics_sample_tHQ_categorised]
        self.yscores_test_categorised[4] = [y_scores_test_ttH_sample_newphysics_categorised, y_scores_test_Other_sample_newphysics_categorised, y_scores_test_ttW_sample_newphysics_categorised, y_scores_test_tHQ_sample_newphysics_categorised, y_scores_test_newphysics_sample_newphysics_categorised]

        #y_scores_train_nonCat = [[0 for x in range(w)] for y in range(h)]
        #y_scores_test_nonCat = [[0 for x in range(w)] for y in range(h)]
        self.yscores_train_non_categorised[0] = [y_scores_train_ttH_sample_ttHnode, y_scores_train_Other_sample_ttHnode, y_scores_train_ttW_sample_ttHnode, y_scores_train_tHQ_sample_ttHnode, y_scores_train_newphysics_sample_ttHnode]
        self.yscores_train_non_categorised[1] = [y_scores_train_ttH_sample_Othernode, y_scores_train_Other_sample_Othernode, y_scores_train_ttW_sample_Othernode, y_scores_train_tHQ_sample_Othernode, y_scores_train_newphysics_sample_Othernode]
        self.yscores_train_non_categorised[2] = [y_scores_train_ttH_sample_ttWnode, y_scores_train_Other_sample_ttWnode, y_scores_train_ttW_sample_ttWnode, y_scores_train_tHQ_sample_ttWnode, y_scores_train_newphysics_sample_ttWnode]
        self.yscores_train_non_categorised[3] = [y_scores_train_ttH_sample_tHQnode, y_scores_train_Other_sample_tHQnode, y_scores_train_ttW_sample_tHQnode, y_scores_train_tHQ_sample_tHQnode, y_scores_train_newphysics_sample_tHQnode]
        self.yscores_train_non_categorised[4] = [y_scores_train_ttH_sample_newphysicsnode, y_scores_train_Other_sample_newphysicsnode, y_scores_train_ttW_sample_newphysicsnode, y_scores_train_tHQ_sample_newphysicsnode, y_scores_train_newphysics_sample_newphysicsnode]
         
        self.yscores_test_non_categorised[0] = [y_scores_test_ttH_sample_ttHnode, y_scores_test_Other_sample_ttHnode, y_scores_test_ttW_sample_ttHnode, y_scores_test_tHQ_sample_ttHnode, y_scores_test_newphysics_sample_ttHnode]
        self.yscores_test_non_categorised[1] = [y_scores_test_ttH_sample_Othernode, y_scores_test_Other_sample_Othernode, y_scores_test_ttW_sample_Othernode, y_scores_test_tHQ_sample_Othernode, y_scores_test_newphysics_sample_Othernode]
        self.yscores_test_non_categorised[2] = [y_scores_test_ttH_sample_ttWnode, y_scores_test_Other_sample_ttWnode, y_scores_test_ttW_sample_ttWnode, y_scores_test_tHQ_sample_ttWnode, y_scores_test_newphysics_sample_ttWnode]
        self.yscores_test_non_categorised[3] = [y_scores_test_ttH_sample_tHQnode, y_scores_test_Other_sample_tHQnode, y_scores_test_ttW_sample_tHQnode, y_scores_test_tHQ_sample_tHQnode, y_scores_test_newphysics_sample_tHQnode]
        self.yscores_test_non_categorised[4] = [y_scores_test_ttH_sample_newphysicsnode, y_scores_test_Other_sample_newphysicsnode, y_scores_test_ttW_sample_newphysicsnode, y_scores_test_tHQ_sample_newphysicsnode, y_scores_test_newphysics_sample_newphysicsnode]

        node_name = ['ttH','Other','ttW','tHQ','newphysics']
        counter = 0
        for y_scorestrain,y_scorestest in zip(self.yscores_train_categorised,self.yscores_test_categorised):
            colours = ['r','steelblue','g','Fuchsia','darkgoldenrod']
            node_title = node_name[counter]
            plot_title = 'Categorised'
            plot_info = [node_name,colours,data_type,plots_dir,node_title,plot_title]
            self.separations_categories.append(self.draw_category_overfitting_plot(y_scorestrain,y_scorestest,plot_info))
            counter = counter +1

        counter =0
        separations_all = []
        for y_scores_train_nonCat,y_scores_test_nonCat in zip(self.yscores_train_non_categorised,self.yscores_test_non_categorised):
            colours = ['r','steelblue','g','Fuchsia','darkgoldenrod']
            node_title = node_name[counter]
            plot_title = 'Non-Categorised'
            plot_info = [node_name,colours,data_type,plots_dir,node_title,plot_title]
            separations_all.append(self.draw_category_overfitting_plot(y_scores_train_nonCat,y_scores_test_nonCat,plot_info))
            counter = counter +1

        return
