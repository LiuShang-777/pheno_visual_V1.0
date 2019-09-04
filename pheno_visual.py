# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 11:06:41 2019
@author: liushang
"""
print('>===starting the pheno_visual===<')
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sns
import numpy as np
from scipy import stats
import sklearn
import os
import argparse
import warnings
warnings.filterwarnings('ignore')
print('the packages have been correctly imported')
parser=argparse.ArgumentParser(description=
                               'present basic visualization of phenotype data and accomplish basic analysis')
parser.add_argument('-df','--dataframe',type=str,help='the path of dataframe file')
parser.add_argument('-rf','--result_file',type=str,help='the path of result file')
parser.add_argument('-f','--features',type=str,help='features you want to analyze',nargs='*')
parser.add_argument('-ff','--features_file',type=str,help='features file')
parser.add_argument('-c','--color_list',type=str,nargs='*',help='the color use in visualization')
parser.add_argument('-cf','--color_list_file',type=str,help='the file contains color name')
parser.add_argument('-cl','--class_list',type=str,help='the file contain information about the sample classification')
parser.add_argument('-n','--component',type=int,help=
                    'the number of component when perform the PCA analysis')
parser.add_argument('-sht','--sheet_number',type=int,help='the index of a certain sheet in excel sheets')
parser.add_argument('-op','--option',choices=['single','pair','multi'],help='options to deal with dataframe')
args=parser.parse_args()
if args.dataframe==None:
    print('Please provide the input file!')
if args.result_file==None:
    print('There is no output result file path!')
def translate_excel(excel,select_sheet):
    df=pd.read_excel(excel,sheet_name=[select_sheet])
    for k,v in df.items():
        df=v
    del v
    return df
if args.dataframe[-4:]=='xlsx':
    df=translate_excel(args.dataframe,args.sheet_number)
elif args.dataframe[-3:]=='csv':
        df=pd.read_csv(args.dataframe,sep=',')
elif args.dataframe[-3:]=='txt':
        df=pd.read_table(args.dataframe,sep='\t') 
else:
    print('the format of input file is not compatible')
features=[]       
if (args.features==None)&(args.features_file==None):
    features=df.columns.tolist()[1:]
elif(args.features!=None)&(args.features_file==None):
    features=args.features
elif (args.features==None)&(args.features_file!=None):    
    with open(args.features_file,'r') as file:
        for line in file:
            line=line.strip()
            features.append(line)
else:
    print('Could only use one of the -f and -ff')
color_list=[]
if (args.color_list==None)&(args.color_list_file==None):
    color_list=None
elif (args.color_list!=None)&(args.color_list_file==None):
    color_list=args.color_list
elif (args.color_list==None)&(args.color_list_file!=None):
    with open(args.color_list_file,'r') as file:
        for line in file:
            line=line.strip()
            color_list.append(line)
else:
    print('Could only use one of the -c and -cf')
result_file=args.result_file
class_list=[]
if args.class_list==None:
    pass
else:   
    with open(args.class_list,'r') as file:
        temp=[]
        for line in file:
            if line=='\n':
                class_list.append(temp)
                temp=[]
            else:
                line=line.strip()
                line=str(line)
                temp.append(line)
        class_list.append(temp)
        del temp
column_names=df.columns.tolist()
list_str=[]
for i in column_names:
    list_str.append(str(i))
column_names=list_str
del list_str
df_index=[]
for i in df[column_names[0]]:
    df_index.append(str(i))
df[column_names[0]]=df_index
def single_plot(df,feature,color,result_file):    
    plt.figure(figsize=(8,6))
    n,bins,patches=plt.hist(df[feature],color=color,edgecolor='k',bins=10,rwidth=0.9)
    sigma,mean,p=df[feature].std(),df[feature].mean(),stats.shapiro(df[feature])                       
    plt.xlabel(feature)
    plt.ylabel('number')
    plt.title('the distibution of %s (mean=%.3f,std=%.3f p=%.3f)'%(feature,mean,sigma,p[1]))
    plt.savefig(result_file)
    plt.clf()      
def density(df,feature,color,result_file):
    plt.figure(figsize=(8,6))
    df[feature].plot(kind='kde',color=color)
    sigma,mean,p=df[feature].std(),df[feature].mean(),stats.shapiro(df[feature])
    plt.xlabel(feature)
    plt.title('the distibution of %s (mean=%.3f,std=%.3f p=%.3f)'%(feature,mean,sigma,p[1]))
    plt.savefig(result_file)
    plt.clf()      
def distribution_2d(df,features,result_file):
    plt.figure(figsize=(8,6))
    plt.hist2d(df[features[0]],df[features[1]],color='black',cmap='Reds')
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.savefig(result_file)
    plt.clf()
def scatter(df,features,list_class,color_list,result_file):
    plt.figure(figsize=(8,6))
    for i in range(len(list_class)):                
        plt.scatter(df.loc[(df[column_names[0]]).isin(list_class[i])][features[0]],
                           df.loc[(df[column_names[0]]).isin(list_class[i])][features[1]],
                           color=color_list[i])
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.title('the distribution of %s & %s'%(features[0],features[1]))
    plt.savefig(result_file)
    plt.clf()
def t_test(df,features):
    varience_equal=stats.levene(df[features[0]],df[features[1]])
    if varience_equal[1]<=0.05:
        sig_judge=stats.ttest_ind(df[features[0]],df[features[1]],equal_var=False)
    else:
        sig_judge=stats.ttest_ind(df[features[0]],df[features[1]],equal_var=True)
    if sig_judge[1]<=0.05:
        print ('significant',sig_judge[1])
    else:
        print('not significant',sig_judge[1])     
def multi_bars(df,features,color_list,result_file):
    bar_list=[]
    var_list=[]
    for i in range(len(features)):
        bar_list.append(df[features[i]].mean())
        var_list.append(df[features[i]].std())
    plt.figure(figsize=(8,6))   
    plt.bar(np.arange(len(features)),bar_list,0.3,color=color_list,
            yerr=var_list,error_kw={'ecolor':'0.2','capsize':6},label=features)
    plt.xticks(np.arange(len(features)),features)
    plt.savefig(result_file)
    plt.clf()
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
def pca(df,features,list_class,color_list,result_file,nun_comp=2):
    def translate_to_index(df):
        index_out=[]
        for i in list_class:
            index_in=[]
            for j in i:
                index_in.append(df_index.index(j))
            index_out.append(index_in)
        return index_out
    pca_single=PCA(n_components=nun_comp)
    index=translate_to_index(df)
    pca_result=pca_single.fit_transform(df[features])
    if nun_comp>3:
        pca_result=pd.DataFrame(pca_result)
        pca_result.to_csv(result_file+'pca_result.csv',index=False)
    elif nun_comp==3:
        plt.figure(figsize=(15,10))
        ax=plt.subplot(projection='3d')
        pca_result_df=pd.DataFrame()
        pca_result_df['pca1']=pca_result[:,0]
        pca_result_df['pca2']=pca_result[:,1]
        pca_result_df['pca3']=pca_result[:,2]
        for i in range(len(list_class)):
            ax.scatter(pca_result_df.iloc[index[i],0],
                       pca_result_df.iloc[index[i],1],
                       pca_result_df.iloc[index[i],2],color=color_list[i])
        ax.set_xlabel('PCA1')
        ax.set_ylabel('PCA2')
        ax.set_zlabel('PCA3')
        plt.title('explain ratio:%.3f, %.3f, %.3f'%(pca_single.explained_variance_ratio_[0],
                                                    pca_single.explained_variance_ratio_[1],
                                                    pca_single.explained_variance_ratio_[2]))
        plt.savefig(result_file+'pca.pdf')
        plt.clf()
    elif nun_comp==2:
        plt.figure(figsize=(10,8))
        for i in range(len(list_class)):            
            plt.scatter(pca_result[index[i],0],
                    pca_result[index[i],1],color=color_list[i])
        plt.xlabel('PCA1')
        plt.ylabel('PCA2')
        plt.title('explain ratio:%.3f, %.3f'%(pca_single.explained_variance_ratio_[0],
                                            pca_single.explained_variance_ratio_[1]))
        plt.savefig(result_file+'pca.pdf')
        plt.clf()
def multi_env_boxplot(df,features,color_list,result_file):
    box_plot_list=[]
    for i in range(len(features)):
        box_plot_list.append(df[features[i]])       
    box_list=plt.boxplot(box_plot_list,labels=features,showmeans=True,sym='.',patch_artist=True,
                         flierprops={'marker':'o','markerfacecolor':'black'})
    for box,color in zip(box_list['boxes'],color_list):
        box.set_facecolor(color)
    plt.savefig(result_file) 
    plt.clf()
from sklearn.cluster import KMeans
print('Kmeans packages has been imported')
def multi_kmeans(df,features,result_file):
    df_kmeans=df[features]
    k_class=[]
    for i in range(2,len(df_kmeans)):
        k_clust=KMeans(n_clusters=i)
        k_labels=k_clust.fit_predict(df_kmeans)
        k_class.append((i,sklearn.metrics.silhouette_score(df_kmeans,k_labels)))
    proper_k=0
    k_score=0
    for i in k_class:
        if i[1]>k_score:
            k_score=i[1]
            proper_k=i[0]
        else:
            continue
    kmeans_cluster=KMeans(n_clusters=proper_k).fit(df_kmeans)
    df_kmeans['clusters']=kmeans_cluster.labels_
    df_kmeans[column_names[0]]=df[column_names[0]]
    df_kmeans.to_csv(result_file,index=False)
    del df_kmeans
from scipy.cluster import hierarchy 
print('The hierarchy package has been imported') 
def multi_hierarchy(df,features,result_file):    
    df_hierarchy=df
    df_hierarchy=df_hierarchy.set_index(column_names[0])
    sns.clustermap(df_hierarchy[features],method ='ward',metric='euclidean',cmap='RdYlBu_r',linewidths=0.5)
    plt.savefig(result_file)
    plt.clf()
def cor_plot(df,features,result_file):
    df_cor=df[features].corr()
    plt.figure(figsize=(8,6),dpi=80)
    sns.heatmap(df_cor,xticklabels=df_cor.columns,yticklabels=df_cor.columns,cmap='RdYlBu_r',linewidths=0.5)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig(result_file)
    plt.clf()
if args.option=='single':
    if len(features)!=1:
        print('You have choose the single traits analyze,please input only one features')
    else:
        single_plot(df,features[0],color_list,result_file+'single_trait_plot.pdf')
        density(df,features,color_list,result_file+'density.pdf')
elif args.option=='pair':
    if len(features)!=2:
        print('You have choose pair traits analyze,please input two features')
    else:
        distribution_2d(df,features,result_file+'2ddistribution.pdf')
        if args.class_list==[]:
            pass
        else:
            scatter(df,features,class_list,color_list,result_file+'scatter.pdf')
        t_test(df,features)
elif args.option==None:
    print('You need to choose a type of analysis')
elif args.option=='multi':
    print('***Starting the multi-mode***')
    print('***Performing the multibars plot***')
    multi_bars(df,features,color_list,result_file+'multi_bars.pdf')
    if (args.component==None) or (args.component==0) or (args.component==1):
        print('***PCA analysis was ignored***')
        pass    
    else:
        print('***Starting the PCA analysis***')
        pca(df,features,class_list,color_list,result_file,nun_comp=args.component)
    print('***Performing the multi-environment boxplot***')
    multi_env_boxplot(df,features,color_list,result_file+'boxplot.pdf')
    print('***Performing the K means analysis***')
    multi_kmeans(df,features,result_file+'kmeans.csv')
    print('***Performing the hierarchy analysis***')
    multi_hierarchy(df,features,result_file+'hierarchy.pdf')
    print('***Performing the corrplot***')
    cor_plot(df,features,result_file+'cor_plot.pdf')
else:
	print('You need to input right choices: single,pair or multi')
print('analyzing finished')
print('>=====END====<')