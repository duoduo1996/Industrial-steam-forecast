import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore")

train_data_file="./zhengqi_train.txt"
test_data_file="./zhengqi_test.txt"
train_data=pd.read_csv(train_data_file,sep='\t',encoding='utf-8')
test_data=pd.read_csv(test_data_file,sep='\t',encoding='utf-8')

#批量画出箱体图
plt.tight_layout(pad=0.9)
plt.figure(figsize=(30,20))
plt.subplots_adjust(wspace = 0.2)
nbr_columns = 8
nbr_graphs = len(train_data.columns)
nbr_rows = int(np.ceil(nbr_graphs/nbr_columns))
columns = list(train_data.columns.values)
with sns.axes_style("whitegrid"):
    for i in range(0,len(columns)-1):
        plt.subplot(nbr_rows,nbr_columns,i+1)
        ax1=sns.boxplot(y= columns[i], data= train_data, orient="h",color=sns.color_palette("Blues")[3])
    plt.show()

#采用模型预测找出异常值
def find_outliers(model,x,y,sigma=3):

    try:
        y_pred=pd.Series(model.predict(x),index=y.index)
    except:
        model.fit(x,y)
        y_pred=pd.Series(model.predict(x),index=y.index)
    resid=y-y_pred
    mean_resid=resid.mean()
    std_resid=resid.std()

    z=(resid-mean_resid)/std_resid
    outliers=z[abs(z)>sigma].index

    print('R2=',model.score(x,y))
    print('mse=',mean_squared_error(y,y_pred))
    print('-'*10)

    print('mean of residuals:',mean_resid)
    print('std of residuals:',std_resid)
    print('-'*10)

    print(len(outliers),'outliers:')
    print(outliers.tolist())

    plt.figure(figsize=(15,5))
    ax_131=plt.subplot(1,3,1)
    plt.plot(y,y_pred,'.')
    plt.plot(y.loc[outliers],y_pred[outliers],'ro')
    plt.legend(['Accepted','Outlier'])
    plt.xlabel('y')
    plt.ylabel('y_pred');

    ax_132=plt.subplot(1,3,2)
    plt.plot(y,y_pred,'.')
    plt.plot(y.loc[outliers],y.loc[outliers]-y_pred.loc[outliers],'ro')
    plt.legend(['Accepted','Outliers'])
    plt.xlabel('y')
    plt.ylabel('y-y_pred');

    ax_133=plt.subplot(1,3,3)
    z.plot.hist(bins=50,ax=ax_133)
    z.loc[outliers].plot.hist(color='r',bins=50,ax=ax_133)
    plt.legend(['Accepted','Outlier'])
    plt.xlabel('z')

    plt.savefig('outliers.png')
    plt.show()

    return outliers
x_train=train_data.iloc[:,0:-1]
y_train=train_data.iloc[:,-1]
outliers=find_outliers(Ridge(),x_train,y_train)

#批量画出直方图和QQ图
train_cols=6
train_rows=len(train_data.columns)
plt.figure(figsize=(60,40))

i=0
with sns.axes_style("whitegrid"):
    for col in train_data.columns:
        i+=1
        ax=plt.subplot(10,8,i)
        sns.distplot(train_data[col],fit=stats.norm)

        i+=1
        ax=plt.subplot(10,8,i)
        res=stats.probplot(train_data[col],plot=plt)
plt.tight_layout()
plt.show()
#KDE分布图
plt.figure(figsize=(50,30))
i=1
with sns.axes_style("whitegrid"):
    for col in test_data.columns:
        ax =plt.subplot(7,6,i)
        ax =sns.kdeplot(train_data[col],color="Red",shade=True)
        ax =sns.kdeplot(test_data[col],color="Blue",shade=True)
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        ax=ax.legend(["train","test"])
        i+=1
plt.show()
#线性回归关系图
plt.figure(figsize=(60,40))
i=0
with sns.axes_style("whitegrid"):
    for col in test_data.columns:
        i+=1
        ax=plt.subplot(10,8,i)
        sns.regplot(x=col,y='target',data=train_data,ax=ax,scatter_kws={'marker':'.','s':3,'alpha':0.3},
                    line_kws={'color':'k'});
        plt.xlabel(col)
        plt.ylabel('target')

        i+=1
        ax=plt.subplot(10,8,i)
        sns.distplot(train_data[col].dropna())
        plt.xlabel(col)
plt.tight_layout()
plt.show()



#删除训练集和测试集中分布不一致的特征变量
pd.set_option('display.max_columns',10)
pd.set_option('display.max_rows',10)
data_train1=train_data.drop(['V5','V9','V11','V17','V22','V28'],axis=1)

train_corr=data_train1.corr()

#根据相关系数筛选特征变量
k=10
cols=train_corr.nlargest(k,'target')['target'].index
cm=np.corrcoef(train_data[cols].values.T)
hm=plt.subplots(figsize=(10,10))
hm=sns.heatmap(train_data[cols].corr(),annot=True,square=True)
plt.show()

threshold=0.5
#相关系数矩阵
corr_matrix=data_train1.corr().abs()
drop_col=corr_matrix[corr_matrix["target"]<threshold].index
#data_train1.drop(drop_col,axis=1,inplace=True)

#BOX-cox变换
drop_columns=['V5','V9','V11','V17','V22','V28']
train_x=train_data.drop(['target'],axis=1)
data_all=pd.concat([train_x,test_data])
data_all.drop(drop_columns,axis=1,inplace=True)
data_all.head()
cols_numeric=list(data_all.columns)

def scale_minmax(col):
    return (col-col.min())/(col.max()-col.min())
data_all[cols_numeric]=data_all[cols_numeric].apply(scale_minmax,axis=0)
print(data_all[cols_numeric].describe())

train_data_process=train_data[cols_numeric]
train_data_process=train_data_process[cols_numeric].apply(scale_minmax,axis=0)
cols_numeric_left=cols_numeric[0:13]
cols_numeric_right=cols_numeric[13:]
train_data_process=pd.concat([train_data_process,train_data['target']],axis=1)
fcols=6
frows=len(cols_numeric_left)
plt.figure(figsize=(4*fcols,4*frows))
i=0
for var in cols_numeric_left:
    dat=train_data_process[[var,'target']].dropna()
    i+=1
    plt.subplot(frows,fcols,i)
    sns.distplot(dat[var],fit=stats.norm)
    plt.title(var+'Original')
    plt.xlabel('')
    i+=1
    plt.subplot(frows, fcols, i)
    _=stats.probplot(dat[var],plot=plt)
    plt.title('skew='+'{:.4f}'.format(stats.skew(dat[var])))
    plt.xlabel('')
    plt.ylabel('')
    i+=1
    plt.subplot(frows,fcols,i)
    plt.plot(dat[var],dat['target'],'.',alpha=0.5)
    plt.title('corr'+'{:.2f}'.format(np.corrcoef(dat[var],dat['target'])[0][1]))
    i+=1
    plt.subplot(frows,fcols,i)
    trains_var,lambda_var=stats.boxcox(dat[var].dropna()+1)
    trains_var=scale_minmax(trains_var)

