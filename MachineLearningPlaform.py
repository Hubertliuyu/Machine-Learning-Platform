import tkinter.filedialog
from tkinter import *
from tkinter.scrolledtext import ScrolledText
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from tkinter import ttk
from pandas import read_csv,set_option,get_dummies,concat,DataFrame
from numpy import zeros
import jieba
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
root = Tk()

root.minsize(850,500)
root.title("机器学习应用平台")


set_option('display.max_columns', 15)
set_option('display.width', 100)
set_option('display.max_colwidth', 100)


def getpath_train():

    global data_train
    filename = tkinter.filedialog.askopenfilename()
    data = read_csv(filename,index_col = 0)
    data_train = data

    contents.insert(INSERT, '----------------------------')
    contents.insert(INSERT, '\n')
    contents.insert(INSERT,'已选择训练数据集文件   ')

    contents.insert(INSERT,filename)
    contents.insert(INSERT, '\n')
    contents.insert(INSERT, '\n')



def getpath_test():

    global data_test
    filename = tkinter.filedialog.askopenfilename()
    data = read_csv(filename,index_col = 0)#index_col 默认值为index_col = None。当作为具体的数字时表示该列为索引列
    data_test = data

    contents.insert(INSERT, '----------------------------')
    contents.insert(INSERT, '\n')
    contents.insert(INSERT,'已选择测试数据集文件   ')

    contents.insert(INSERT,filename)
    contents.insert(INSERT, '\n')
    contents.insert(INSERT, '\n')


def loaddata():
    global data_train,data_test


    contents.insert(INSERT, '----------------------------')
    contents.insert(INSERT, '\n')
    contents.insert(INSERT, '\n')
    contents.insert(INSERT,'数据导入成功')
    contents.insert(INSERT,'\n')
    contents.insert(INSERT,'\n')
    contents.insert(INSERT,'训练集样本总数：  ')
    contents.insert(INSERT, len(data_train))
    contents.insert(INSERT, '\n')
    contents.insert(INSERT, '\n')
    contents.insert(INSERT,'训练集数据前5行预览')
    contents.insert(INSERT, '\n')
    contents.insert(INSERT, '\n')
    contents.insert(INSERT,data_train.head())
    contents.insert(INSERT, '\n')
    contents.insert(INSERT, '\n')
    contents.insert(INSERT,'测试集样本总数：  ')
    contents.insert(INSERT, len(data_test))
    contents.insert(INSERT, '\n')
    contents.insert(INSERT, '\n')
    contents.insert(INSERT,'测试集数据前5行预览')
    contents.insert(INSERT, '\n')
    contents.insert(INSERT, '\n')
    contents.insert(INSERT,data_test.head())
    contents.insert(INSERT, '\n')
    contents.insert(INSERT, '\n')


def enter_click(event):
    global p
    k=float(x_entry00.get())
    p=k
    contents.insert(INSERT, '\n')
    contents.insert(INSERT, '----------------------------')
    contents.insert(INSERT, '\n')
    contents.insert(INSERT, '\n')
    contents.insert(INSERT, '验证集比例   ')
    contents.insert(INSERT, k)
    contents.insert(INSERT, '   加载成功')
    contents.insert(INSERT, '\n')



def data_preprocessing():

    global x_train,y_train,data_train,data_test
    global x_test
    global regressor_name

    y_train_pdSeries = data_train.pop('result')
    y_train = y_train_pdSeries.values


    whole_data = concat((data_train,data_test),axis = 0)

    whole_dummies = get_dummies(whole_data)
    contents.insert(INSERT, '\n')
    contents.insert(INSERT, '\n')
    contents.insert(INSERT, '----------------------------')
    contents.insert(INSERT, '\n')
    contents.insert(INSERT, '\n')
    contents.insert(INSERT, '特征向量化 完成')
    contents.insert(INSERT, '\n')

    # whole_mat = whole_dummies.values

    whole_dummies.isnull().sum().sort_values(ascending=False).head(20)  # 列举出前十个缺省值个数的特征
    meancols = whole_dummies.mean()  # 取每个特征的均值
    whole_dummies = whole_dummies.fillna(meancols)  # 这里我们使用均值填充缺省位置
    if whole_dummies.isnull().sum().sum()== 0:
        contents.insert(INSERT, '\n')
        contents.insert(INSERT, '缺省值填充 完成')
        contents.insert(INSERT,'\n')       # 计数显示是否处理完所有缺省值，0则表示缺省值处理完毕


    contents.insert(INSERT, '\n')
    contents.insert(INSERT, '\n')
    train_dumies = whole_dummies.loc[data_train.index]

    test_dumies = whole_dummies.loc[data_test.index]

    x_train = train_dumies.values
    x_test = test_dumies.values


def regressor(event):

    global regressor_name
    regressor_name = cmb00.get()

    if cmb00.get()=='RandomForest':
        contents.insert(INSERT, '\n')
        contents.insert(INSERT, '----------------------------')
        contents.insert(INSERT,'\n')
        contents.insert(INSERT,'\n')
        contents.insert(INSERT,'选择的训练模型为    ')

        contents.insert(INSERT,'RandomForest regressor')
        contents.insert(INSERT,'\n')
    if cmb00.get()=='GradientBoosting':
        contents.insert(INSERT, '\n')
        contents.insert(INSERT, '----------------------------')
        contents.insert(INSERT,'\n')
        contents.insert(INSERT,'\n')
        contents.insert(INSERT,'选择的训练模型为    ')

        contents.insert(INSERT,'GradientBoosting regressor')
        contents.insert(INSERT,'\n')
    if cmb00.get()=='SVR':
        contents.insert(INSERT, '\n')
        contents.insert(INSERT, '----------------------------')
        contents.insert(INSERT,'\n')
        contents.insert(INSERT,'\n')
        contents.insert(INSERT,'选择的训练模型为    ')
        # contents.insert(INSERT,'\n')
        contents.insert(INSERT,'Support Vector regressor')
        contents.insert(INSERT,'\n')
#
def predict_result():

    global x_train,y_train
    global p
    global x_test
    global regressor_name

    x_train_train, x_train_test, y_train_train, y_train_test = train_test_split(x_train,y_train, test_size=p, random_state=33)

    # 数据标准化

    ss_x = preprocessing.StandardScaler()
    x_train_train = ss_x.fit_transform(x_train_train)
    x_train_test = ss_x.fit_transform(x_train_test)
    x_test = ss_x.transform(x_test)

    # print('*****len(y_train)****2', len(y_train))
    ss_y = preprocessing.StandardScaler()
    y_train_train = ss_y.fit_transform(y_train_train.reshape(-1, 1))
    y_train_test = ss_y.transform(y_train_test.reshape(-1, 1))


    svr = SVR(kernel='rbf')
    svr.fit(x_train_train, y_train_train)
    svr_y_predict = svr.predict(x_test)
    svr_score_vali = svr.score(x_train_test, y_train_test)

    rfr = RandomForestRegressor()
    rfr.fit(x_train_train, y_train_train)
    rfr_y_predict = rfr.predict(x_test)
    rfr_score_vali = rfr.score(x_train_test, y_train_test)

    gbr = GradientBoostingRegressor()
    gbr.fit(x_train_train, y_train_train)
    gbr_y_predict = gbr.predict(x_test)
    gbr_score_vali = gbr.score(x_train_test, y_train_test)


    svr_y_predict = ss_y.inverse_transform(svr_y_predict)
    rfr_y_predict = ss_y.inverse_transform(rfr_y_predict)
    gbr_y_predict = ss_y.inverse_transform(gbr_y_predict)


    Id_test = [i for i in range(len(x_train)+1, len(x_train)+len(x_test)+1, 1)]#<class 'list'>

    Id = zeros((len(x_test), 1)) # <class 'numpy.ndarray'>

    for i in range(len(x_test)):
        Id[i] = Id_test[i]

    Id = Id.ravel()

    if regressor_name == 'SVR':
        contents.insert(INSERT, '\n')
        contents.insert(INSERT, '----------------------------')
        contents.insert(INSERT, '\n')
        contents.insert(INSERT, '\n')

        contents.insert(INSERT, 'SupportVectorRegressor  ')
        contents.insert(INSERT, '预测结果加载成功  svr_predict.csv')
        contents.insert(INSERT, '\n')
        contents.insert(INSERT, '\n')
        contents.insert(INSERT, 'SupportVectorRegressor score：  ')
        contents.insert(INSERT, svr_score_vali)
        contents.insert(INSERT, '\n')
        contents.insert(INSERT, '\n')
        svr_predict = DataFrame({"Id": Id,"predicted value": svr_y_predict})
        svr_predict.to_csv("svr_predict.csv", index=False)

    if regressor_name == 'RandomForest':
        contents.insert(INSERT, '\n')
        contents.insert(INSERT, '----------------------------')
        contents.insert(INSERT, '\n')
        contents.insert(INSERT, '\n')

        contents.insert(INSERT, 'RandomForest regessor  ')
        contents.insert(INSERT, '预测结果加载成功  rfr_predict.csv')
        contents.insert(INSERT, '\n')
        contents.insert(INSERT, '\n')
        contents.insert(INSERT, 'RandomForestRegressor score：  ')
        contents.insert(INSERT, rfr_score_vali)
        contents.insert(INSERT, '\n')
        contents.insert(INSERT, '\n')
        rfr_predict = DataFrame({"Id": Id,"predicted value": rfr_y_predict})
        rfr_predict.to_csv("rfr_predict.csv", index=False)

    if regressor_name == 'GradientBoosting':
        contents.insert(INSERT, '\n')
        contents.insert(INSERT, '----------------------------')
        contents.insert(INSERT, '\n')
        contents.insert(INSERT, '\n')

        contents.insert(INSERT, 'GradientBoosting regressor   ')
        contents.insert(INSERT, '预测结果加载成功  gbr_predict.csv')
        contents.insert(INSERT, '\n')
        contents.insert(INSERT, '\n')
        contents.insert(INSERT, 'GradientBoostingRegressor score：  ')
        contents.insert(INSERT, gbr_score_vali)
        contents.insert(INSERT, '\n')
        contents.insert(INSERT, '\n')
        gbr_predict = DataFrame({"Id": Id,"predicted value": gbr_y_predict})
        gbr_predict.to_csv("gbr_predict.csv", index=False)

#######################################################################################for text processing

def getpath_train_text():

    global data_train_text
    filename = tkinter.filedialog.askopenfilename()
    data = read_csv(filename)
    data_train_text = data

    contents.insert(INSERT, '----------------------------')
    contents.insert(INSERT, '\n')
    contents.insert(INSERT,'已选择训练数据集文件   ')

    contents.insert(INSERT,filename)
    contents.insert(INSERT, '\n')
    contents.insert(INSERT, '\n')



def getpath_test_text():

    global data_test_text
    filename = tkinter.filedialog.askopenfilename()
    data = read_csv(filename,encoding='gbk',)#index_col 默认值为index_col = None。当作为具体的数字时表示该列为索引列
    data_test_text = data

    contents.insert(INSERT, '----------------------------')
    contents.insert(INSERT, '\n')
    contents.insert(INSERT,'已选择测试数据集文件   ')

    contents.insert(INSERT,filename)
    contents.insert(INSERT, '\n')
    contents.insert(INSERT, '\n')

def getpath_stopwords_text():

    global data_stopwords_text
    filename = tkinter.filedialog.askopenfilename()
    data = read_csv(filename, index_col=False,quoting=3,sep="\t",names=['stopword'], encoding='utf-8')

    data_stopwords_text = data

    contents.insert(INSERT, '----------------------------')
    contents.insert(INSERT, '\n')
    contents.insert(INSERT,'已选择停用词文件   ')

    contents.insert(INSERT,filename)
    contents.insert(INSERT, '\n')
    contents.insert(INSERT, '\n')


def loaddata_text():
    global data_train_text,data_test_text,data_stopwords_text


    contents.insert(INSERT, '----------------------------')
    contents.insert(INSERT, '\n')
    contents.insert(INSERT, '\n')
    contents.insert(INSERT,'数据导入成功')
    contents.insert(INSERT,'\n')
    contents.insert(INSERT, '----------------------------')
    contents.insert(INSERT,'\n')
    contents.insert(INSERT,'\n')
    contents.insert(INSERT,'训练集样本总数：  ')
    contents.insert(INSERT, len(data_train_text))
    contents.insert(INSERT, '\n')
    contents.insert(INSERT, '\n')
    contents.insert(INSERT,'训练集数据前5行预览')
    contents.insert(INSERT, '\n')
    contents.insert(INSERT, '\n')
    contents.insert(INSERT,data_train_text.head())
    contents.insert(INSERT, '\n')
    contents.insert(INSERT, '\n')
    contents.insert(INSERT,'\n')
    contents.insert(INSERT, '----------------------------')
    contents.insert(INSERT,'\n')
    contents.insert(INSERT,'测试集样本总数：  ')
    contents.insert(INSERT, len(data_test_text))
    contents.insert(INSERT, '\n')
    contents.insert(INSERT, '\n')
    contents.insert(INSERT,'测试集数据前5行预览')
    contents.insert(INSERT, '\n')
    contents.insert(INSERT, '\n')
    contents.insert(INSERT,data_test_text.head())
    contents.insert(INSERT, '\n')
    contents.insert(INSERT, '\n')
    contents.insert(INSERT,'\n')
    contents.insert(INSERT, '----------------------------')
    contents.insert(INSERT,'\n')
    contents.insert(INSERT,'停用词前5行预览')
    contents.insert(INSERT, '\n')
    contents.insert(INSERT, '\n')
    contents.insert(INSERT,data_stopwords_text.head())
    contents.insert(INSERT, '\n')
    contents.insert(INSERT, '\n')

def enter_click_text(event):
    global p_text
    k=float(x_entry.get())
    p_text=k
    print(p_text)
    contents.insert(INSERT, '\n')
    contents.insert(INSERT, '----------------------------')
    contents.insert(INSERT, '\n')
    contents.insert(INSERT, '\n')
    contents.insert(INSERT, '验证集比例   ')
    contents.insert(INSERT, k)
    contents.insert(INSERT, '   加载成功')
    contents.insert(INSERT, '\n')



def data_preprocessing_text():
    global data_train_text,data_test_text
    global data_stopwords_text
    global sentences,sentences01,labels

    x_train_text = data_train_text['evaluation']
    y_train_text = [[i] for i in data_train_text['label']]

    x_test_text = data_test_text['evaluation']

    stopwords = data_stopwords_text['stopword'].values

    corpus = []
    corpus01=[]

    labels = zeros(len(y_train_text))

    for i in range(len(x_train_text)):
        corpus.append(x_train_text[i])

    for i in range(len(x_test_text)):
        corpus01.append(x_test_text[i])

    for j in range(len(y_train_text)):
        if y_train_text[j] == ['正面']:
            labels[j] = 1
        else:
            labels[j] = 0

    def preprocess_text(content_lines, sentences):
        for line in content_lines:
            try:
                segs = jieba.lcut(line)
                segs = filter(lambda x: len(x) > 1, segs)
                segs = filter(lambda x: x not in stopwords, segs)
                sentences.append((" ".join(segs)))
            except:
                print(line)
                continue


    # 生成训练数据
    sentences = []
    sentences01=[]

    preprocess_text(corpus, sentences)
    preprocess_text(corpus01, sentences01)


    # train_corpus, test_corpus, train_labels, test_labels = train_test_split(sentences, labels, test_size=p, random_state=33)
    # print(train_corpus[0:3])
    # ['电视 第三天 死机 黑屏 重启 第六天 客服 客服 售后 派人来 鉴定 换机 电视 打包 封箱 客服 星期 却说 我出 运费 电视 质量 我来 运费 轻易 差评 太气', '父母 电视 显示 清晰 物流 很快 安装 京东 质量 放心', '送货 速度 很快 昨天早上 单子 喜欢 品牌 家里 两台 微鲸 一年 感觉 不错 认准 微鲸 牌子 搞了个 回来 正好 赶上 京东 618 活动 挂架 挂架 没到 送给']
    contents.insert(INSERT, '\n')
    contents.insert(INSERT, '\n')
    contents.insert(INSERT, '----------------------------')
    contents.insert(INSERT, '\n')
    contents.insert(INSERT, '\n')
    contents.insert(INSERT, '分词 完成')
    contents.insert(INSERT, '\n')
    contents.insert(INSERT, '\n')
    contents.insert(INSERT, '\n')
    contents.insert(INSERT, '----------------------------')
    contents.insert(INSERT, '\n')
    contents.insert(INSERT, '\n')
    contents.insert(INSERT, '去停用词 完成')
    contents.insert(INSERT, '\n')
    contents.insert(INSERT, '\n')
    contents.insert(INSERT, '\n')

    count_vector = CountVectorizer()
    # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    vector_matrix = count_vector.fit_transform(sentences)
    # tfidf度量模型
    corpus_tfidf = TfidfTransformer(use_idf=False).fit_transform(vector_matrix)
    # print('corpus_tfidf[0:3]',corpus_tfidf[0:3])
    test_tfidf = count_vector.transform(sentences01)
    corpus01_tfidf = TfidfTransformer(use_idf=False).fit_transform(test_tfidf)
    # print('corpus01_tfidf[0:3]',corpus01_tfidf[0:3])


    contents.insert(INSERT, '----------------------------')
    contents.insert(INSERT, '\n')
    contents.insert(INSERT, '\n')
    contents.insert(INSERT, '文本向量化 完成')
    contents.insert(INSERT, '\n')
    contents.insert(INSERT, '\n')
    contents.insert(INSERT, '\n')
    #



def regressor_text(event):
    global regressor_text_name
    regressor_text_name = cmb.get()

    if cmb.get()=='MultinomialNB':
        contents.insert(INSERT, '\n')
        contents.insert(INSERT, '----------------------------')
        contents.insert(INSERT,'\n')
        contents.insert(INSERT,'\n')
        contents.insert(INSERT,'选择的训练模型为    ')

        contents.insert(INSERT,'MultinomialNB Classifier')
        contents.insert(INSERT,'\n')
    if cmb.get()=='SGDClassifier':
        contents.insert(INSERT, '\n')
        contents.insert(INSERT, '----------------------------')
        contents.insert(INSERT,'\n')
        contents.insert(INSERT,'\n')
        contents.insert(INSERT,'选择的训练模型为    ')

        contents.insert(INSERT,'SGDClassifier')
        contents.insert(INSERT,'\n')
    if cmb.get()=='LogisticRegression':
        contents.insert(INSERT, '\n')
        contents.insert(INSERT, '----------------------------')
        contents.insert(INSERT,'\n')
        contents.insert(INSERT,'\n')
        contents.insert(INSERT,'选择的训练模型为    ')
        # contents.insert(INSERT,'\n')
        contents.insert(INSERT,'LogisticRegression')
        contents.insert(INSERT,'\n')

def enter_click_text(event):
    global p_text
    k=float(x_entry.get())
    p_text=k
    # print(p_text)
    contents.insert(INSERT, '\n')
    contents.insert(INSERT, '----------------------------')
    contents.insert(INSERT, '\n')
    contents.insert(INSERT, '\n')
    contents.insert(INSERT, '验证集比例   ')
    contents.insert(INSERT, k)
    contents.insert(INSERT, '   加载成功')
    contents.insert(INSERT, '\n')


def predict_result_text():

    global p_text
    global sentences,sentences01,labels

    # print(sentences[0:3])
    # print(sentences01[0:3])

    train_corpus, test_corpus, train_labels, test_labels = train_test_split(sentences, labels, test_size=p_text, random_state=33)

    count_vector = CountVectorizer()
    # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    vector_matrix = count_vector.fit_transform(train_corpus)
    # tfidf度量模型
    train_tfidf = TfidfTransformer(use_idf=False).fit_transform(vector_matrix)
    # print(train_tfidf)
    test_tfidf = count_vector.transform(test_corpus)
    new_tfidf = TfidfTransformer(use_idf=False).fit_transform(test_tfidf)

    corpus01_tfidf = count_vector.transform(sentences01)
    result_tfidf = TfidfTransformer(use_idf=False).fit_transform(corpus01_tfidf)

    mnb = MultinomialNB().fit(train_tfidf, train_labels)
    svm = SGDClassifier(loss='hinge', n_iter=100).fit(train_tfidf, train_labels)
    lr = LogisticRegression().fit(train_tfidf, train_labels)


    mnb_predict_result = mnb.predict(new_tfidf)  # 预测结果
    mnb_score= accuracy_score(test_labels,mnb_predict_result)
    mnb_result = mnb.predict(result_tfidf)
    # print(mnb_result)


    svm_predict_result = svm.predict(new_tfidf)  # 预测结果
    svm_score= accuracy_score(test_labels,svm_predict_result)
    svm_result = svm.predict(result_tfidf)


    lr_predict_result = lr.predict(new_tfidf)  # 预测结果
    lr_score= accuracy_score(test_labels,lr_predict_result)
    lr_result = lr.predict(result_tfidf)


    if regressor_text_name == 'MultinomialNB':
        contents.insert(INSERT, '\n')
        contents.insert(INSERT, '----------------------------')
        contents.insert(INSERT, '\n')
        contents.insert(INSERT, '\n')

        contents.insert(INSERT, 'MultinomialNB  ')
        contents.insert(INSERT, '预测结果加载成功  mnb_predict.csv')
        contents.insert(INSERT, '\n')
        contents.insert(INSERT, '\n')
        contents.insert(INSERT, 'SupportVectorRegressor score：  ')
        contents.insert(INSERT, mnb_score)
        contents.insert(INSERT, '\n')
        contents.insert(INSERT, '\n')
        mnb_predict = DataFrame({"label": mnb_result})
        mnb_predict.to_csv("mnb_predict.csv", index=False)

    if regressor_text_name == 'SGDClassifier':
        contents.insert(INSERT, '\n')
        contents.insert(INSERT, '----------------------------')
        contents.insert(INSERT, '\n')
        contents.insert(INSERT, '\n')

        contents.insert(INSERT, 'SGDClassifier  ')
        contents.insert(INSERT, '预测结果加载成功  svm_predict.csv')
        contents.insert(INSERT, '\n')
        contents.insert(INSERT, '\n')
        contents.insert(INSERT, 'SGDClassifier score：  ')
        contents.insert(INSERT, svm_score)
        contents.insert(INSERT, '\n')
        contents.insert(INSERT, '\n')
        svm_predict = DataFrame({"label": svm_result})
        svm_predict.to_csv("svm_predict.csv", index=False)

    if regressor_text_name == 'LogisticRegression':
        contents.insert(INSERT, '\n')
        contents.insert(INSERT, '----------------------------')
        contents.insert(INSERT, '\n')
        contents.insert(INSERT, '\n')

        contents.insert(INSERT, 'LogisticRegression   ')
        contents.insert(INSERT, '预测结果加载成功  lr_predict.csv')
        contents.insert(INSERT, '\n')
        contents.insert(INSERT, '\n')
        contents.insert(INSERT, 'LogisticRegression score：  ')
        contents.insert(INSERT, lr_score)
        contents.insert(INSERT, '\n')
        contents.insert(INSERT, '\n')
        lr_predict = DataFrame({"label": lr_result})
        lr_predict.to_csv("lr_predict.csv", index=False)


def quit():
    sys.exit()

label00 = Label(root,text="数据预测").place(x = '5',y='5',width = '80',height='30')
Button(text='选择训练集文件',bg='DarkGray',command=getpath_train).place(x = '10',y='40',width = '100',height='30')
Button(text='选择测试集文件',bg='DarkGray',command=getpath_test).place(x = '10',y='80',width = '100',height='30')
label01 = Label(root,text="======>>").place(x = '110',y='60',width = '80',height='30')
Button(text='读取文件数据',bg='DarkGray',command=loaddata).place(x = '190',y='60',width = '100',height='30')
label02 = Label(root,text="======>>").place(x = '290',y='60',width = '80',height='30')


enter_button = Button(root,bg='DarkGray', text="数据预处理",command=data_preprocessing)
enter_button.place(x = '370',y='60',width = '100',height='30')


label03 = Label(root,text="======>>").place(x = '470',y='60',width = '80',height='30')

label2 = Label(root,bg='DarkGray',text="选择训练模型").place(x = '550',y='25',width = '100',height='30')

cmb00 = ttk.Combobox()
cmb00.place(x = '550',y='55',width = '100',height='20')
cmb00['value']=('SVR','RandomForest','GradientBoosting')
cmb00.current(1)
cmb00.bind("<<ComboboxSelected>>",regressor)

x_entry00 = Entry(root)
x_entry00.place(x = '550',y='110',width = '100',height='20')
enter_button00 = Button(root,bg='DarkGray', text="加载验证集比例")
enter_button00.place(x = '550',y='80',width = '100',height='30')
enter_button00.bind("<Button-1>", enter_click)
enter_button00.bind("<Return>", enter_click)

label04 = Label(root,text="======>>").place(x = '650',y='60',width = '80',height='30')

Button(text='导出预测结果',bg='DarkGray',command=predict_result).place(x = '730',y='60',width = '100',height='30')

label05 = Label(root,text="*****************************************************************************************************************************************************************************").place(x = '0',y='150',width = '850',height='10')


label06 = Label(root,text="文本分类").place(x = '5',y='155',width = '80',height='30')
Button(text='选择训练集文件',bg='DarkGray',command=getpath_train_text).place(x = '10',y='190',width = '100',height='30')
Button(text='选择测试集文件',bg='DarkGray',command=getpath_test_text).place(x = '10',y='230',width = '100',height='30')
Button(text='选择停用词文件',bg='DarkGray',command=getpath_stopwords_text).place(x = '10',y='270',width = '100',height='30')
label07 = Label(root,text="======>>").place(x = '110',y='230',width = '80',height='30')
Button(text='读取文件数据',bg='DarkGray',command=loaddata_text).place(x = '190',y='230',width = '100',height='30')
label08 = Label(root,text="======>>").place(x = '290',y='230',width = '80',height='30')

enter_button = Button(root,bg='DarkGray', text="数据预处理",command=data_preprocessing_text)
enter_button.place(x = '370',y='230',width = '100',height='30')


label09 = Label(root,text="======>>").place(x = '470',y='230',width = '80',height='30')

label010 = Label(root,bg='DarkGray',text="选择训练模型").place(x = '550',y='185',width = '100',height='30')

cmb = ttk.Combobox()
cmb.place(x = '550',y='215',width = '100',height='20')
cmb['value']=('MultinomialNB','SGDClassifier','LogisticRegression')
cmb.current(1)
cmb.bind("<<ComboboxSelected>>",regressor_text)

x_entry = Entry(root)
x_entry.place(x = '550',y='280',width = '100',height='20')
enter_button = Button(root,bg='DarkGray', text="加载验证集比例")
enter_button.place(x = '550',y='250',width = '100',height='30')
enter_button.bind("<Button-1>", enter_click_text)
enter_button.bind("<Return>", enter_click_text)



label011 = Label(root,text="======>>").place(x = '650',y='230',width = '80',height='30')

Button(text='导出预测结果',bg='DarkGray',command=predict_result_text).place(x = '730',y='230',width = '100',height='30')



contents = ScrolledText()
contents.place(x = '5',y='310',width = '830',height='140')

Button(text='退出系统',bg='DarkGray', command=quit).place(x = '385',y='460',width = '80',height='30')

mainloop()