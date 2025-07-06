import numpy as np
import pandas as pd
import math

'''根据数据集构造关联矩阵'''


def adjacency_matrix():
    print('读取数据')
    association_df = pd.read_excel('data/disease-metabolite.xlsx', sheet_name='关联关系_4763个', header=0)  # 共4763个关联关系
    print('association_df', association_df)

    uniqueMetabolite_df = pd.read_excel('data/disease-metabolite.xlsx', sheet_name='代谢物_2315个', header=0)
    unique_Metabolite = uniqueMetabolite_df['Metabolite ID'].tolist()  # 唯一的Metabolite 2315个

    uniqueDisease_df = pd.read_excel('data/disease-metabolite.xlsx', sheet_name='疾病_265个')
    unique_disease = uniqueDisease_df['Disease Name'].tolist()  # 唯一的疾病265个

    print(len(unique_Metabolite))
    print(len(unique_disease))

    association_matrix = np.zeros([len(unique_Metabolite), len(unique_disease)])  # 构造2315*265的全0矩阵

    print('构造association_matrix')
    count = 0
    for m in range(len(unique_Metabolite)):
        for n in range(len(unique_disease)):
            if len(association_df[np.logical_and(association_df['Metabolite ID'] == unique_Metabolite[m],
                                                 association_df['Disease Name'] == unique_disease[n])]):
                # np.logical_and(a,b) (逻辑与),当a,b均成立时为真
                association_matrix[m, n] = 1
                count += 1
    print('count', count)
    print('association_matrix', association_matrix[0: 10, 0: 10])

    # 保存结果
    result = pd.DataFrame(association_matrix)
    result.to_excel('output/association_matrix.xlsx', index=False)  # index=False设置不生成序号列
    # 注意，这样保存之后会多了一行一列行号序号，需要删除
    return result


#  print(adjacency_matrix())

'''疾病相似性'''
# 计算疾病语义相似性
def disese_sema_simi():
    """语义代码存在两个问题：
    1、语义相似性的分母计算，同一种疾病的不同编码算了多次(多个1相加，定义是只算一个1)；
    2、疾病的祖先节点的语义值更新（未按最大值更新）和定义也有出入
    两个问题均已解决
    """

    print("开始读取数据")
    meshid = pd.read_excel('data/disease-metabolite.xlsx', sheet_name='DAG_918个', header=0)
    disease = meshid['Disease Name'].tolist()  # 将disease列转化为列表
    id = meshid['DAG ID'].tolist()  # 将ID列转化为列表
    # print('disease', disease)
    print('disease', len(disease))
    # print('id', id)
    print('id', len(id))
    meshdis = pd.read_excel('data/disease-metabolite.xlsx', sheet_name='疾病_265个', header=0)
    unique_disease = meshdis['Disease Name'].tolist()
    # print('unique_disease', unique_disease)
    print('unique_disease', len(unique_disease))

    # 初始化字典，有重复也没关系
    for i in range(len(disease)):
        disease[i] = {}
        #  print(disease)

    print("开始计算每个病的语义值")
    # 计算每个病的语义值，又重复也没关系，之后再合并
    for i in range(len(disease)):
        # print(id[i],len(id[i]))
        if len(id[i]) > 3:
            disease[i][id[i]] = 1  # 对列表中第i个空字典的id[i](key值)赋值1；
            id[i] = id[i][:-4]  # 数组切片，对第i个元素从第一个开始到倒数第四个进行截取（取掉后面4位）
            # print('id[i]',id[i])
            # print(disease[i])
            if len(id[i]) > 3:
                disease[i][id[i]] = round(1 * 0.5, 5)  # round(number,num_digits)number：需要四舍五入的数;digits：需要小数点后保留的位数；
                id[i] = id[i][:-4]  # 数组切片，从第一个开始到倒数第四个进行截取
                # print(disease[i])
                if len(id[i]) > 3:
                    disease[i][id[i]] = round(1 * 0.5 * 0.5, 5)
                    id[i] = id[i][:-4]
                    # print(disease[i])
                    if len(id[i]) > 3:
                        disease[i][id[i]] = round(1 * 0.5 * 0.5 * 0.5, 5)
                        id[i] = id[i][:-4]
                        # print(disease[i])
                        if len(id[i]) > 3:
                            disease[i][id[i]] = round(1 * 0.5 * 0.5 * 0.5 * 0.5, 5)
                            id[i] = id[i][:-4]
                            # print(disease[i])
                            if len(id[i]) > 3:
                                disease[i][id[i]] = round(1 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5, 5)
                                id[i] = id[i][:-4]
                                # print(disease[i])
                                if len(id[i]) > 3:
                                    disease[i][id[i]] = round(1 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5, 5)
                                    id[i] = id[i][:-4]
                                    # print(disease[i])
                                    if len(id[i]) > 3:
                                        disease[i][id[i]] = round(1 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5, 5)
                                        id[i] = id[i][:-4]
                                        # print(disease[i])
                                    else:
                                        disease[i][id[i][:3]] = round(1 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5, 5)
                                        # print(disease[i])
                                else:
                                    disease[i][id[i][:3]] = round(1 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5, 5)
                                    # print(disease[i])
                            else:
                                disease[i][id[i][:3]] = round(1 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5, 5)
                                # print(disease[i])
                        else:
                            disease[i][id[i][:3]] = round(1 * 0.5 * 0.5 * 0.5 * 0.5, 5)
                            # print(disease[i])
                    else:
                        disease[i][id[i][:3]] = round(1 * 0.5 * 0.5 * 0.5, 5)
                        # print(disease[i])
                else:
                    disease[i][id[i][:3]] = round(1 * 0.5 * 0.5, 5)
                    # print(disease[i])
            else:
                disease[i][id[i][:3]] = round(1 * 0.5, 5)
                # print(disease[i])
        else:
            disease[i][id[i][:3]] = 1  # 对列表中第i个空字典的id[i](key值)截取前三位进行 赋值1；
            # print(disease[i])
    print('disease', disease)  # 最终结果

    print("合并相同的病不同ID的语义值")
    unique_disease = meshdis['Disease Name'].tolist()
    # print('unique_disease',unique_disease)

    # 这个name用来判断
    disease_name = meshid['Disease Name'].tolist()
    # print('disease_name',disease_name)
    unique_disease_name = meshdis['Disease Name'].tolist()
    # print('unique_disease_name',unique_disease_name)

    for i in range(len(unique_disease)):
        unique_disease[i] = {}
        for j in range(len(disease_name)):
            if unique_disease_name[i] == disease_name[j]:
                for key in disease[j].keys():
                    if key not in unique_disease[i].keys() or unique_disease[i][key] < disease[j][key]:
                        unique_disease[i][key] = disease[j][key]
            '''另一种实现方式，稍微复杂'''
            # if  len(unique_disease[i])!=0 and len(unique_disease[i].keys() & disease[j].keys()) != 0:
            #     for key in unique_disease[i].keys() & disease[j].keys():#输出键相同的一个集合
            #         if unique_disease[i][key] < disease[j][key]:
            #             unique_disease[i][key] = disease[j][key]
            #     for key1 in disease[j].keys() - unique_disease[i].keys():#判断键不同的情况，也需要添加进去
            #         unique_disease[i][key1] = disease[j][key1]
            # else:
            #     unique_disease[i].update(disease[j])
            # 存在更新不是最大值的问题，已解决
            # 用update更新字典，会有两种情况：有相同的键时：会使用最新的字典dict2中该key对应的value值。有新的键时：会直接把字典dict2中的key、value加入到dict1中。
    print('unique_disease', unique_disease)
    print(len(unique_disease))

    similarity = np.zeros([len(unique_disease_name), len(unique_disease_name)])  # np.zeros([n,m]):创建n行m列的全0矩阵
    print(similarity)

    print("计算相似度")
    # 求每个疾病语义值为1的个数
    count1 = []
    for i in range(len(unique_disease)):
        count = 0
        for m, n in unique_disease[i].items():
            if n == 1:
                count += 1
        count1.append(count)
    print('count1', count1)

    for m in range(len(unique_disease_name)):
        for n in range(len(unique_disease_name)):
            # denominator1 = sum(unique_disease[m].values())  + sum(unique_disease[n].values())
            denominator = sum(unique_disease[m].values()) - (count1[m] - 1) + \
                          sum(unique_disease[n].values()) - (count1[n] - 1)  # 疾病语义相似性的分母，存在多个1相加的问题，已解决
            # print('denominator1',denominator1)
            # print('denominator', denominator)
            numerator = 0  # 每循环一次都变为0
            for k, v in unique_disease[m].items():
                # print(k,v)
                if k in unique_disease[n].keys():
                    numerator += v + unique_disease[n].get(k)  # 利用get()函数操作时当字典中不存在输入的键时会返回一个None，这样程序运行时就不会出异常
            if m == n:  # 如果不加这句代码，对角线元素可能大于1
                numerator = numerator - (count1[m] - 1) - (count1[n] - 1)
            similarity[m, n] = round(numerator / denominator, 5)  # 给矩阵赋值
            # print(similarity[m, n])
    print(similarity)

    print("保存结果")
    result = pd.DataFrame(similarity)
    result.to_excel('output/disease_Semantic_simi.xlsx', index=False)
    return similarity


#  print(disese_sema_simi())


# 计算疾病高斯核相似性
def disease_Gaussian_Simi():
    # 读取数据
    association_matrix = pd.read_excel('output/association_matrix.xlsx').values  # .values没太懂
    print(association_matrix, association_matrix.shape[0])
    # 每列表示一个病和各metabolite是否有联系，每行表示一个metabolite和每个病是否有联系
    # 计算disease之间的相似度
    association_matrix = association_matrix.T  # 矩阵转置
    print(association_matrix, association_matrix.shape, len(association_matrix))  # 转置之后有265行，2315列，每行一个disease
    disease_similarity = np.zeros([len(association_matrix), len(association_matrix)])  # 265种病之间的相似度，初始化矩阵
    width = 0
    for m in range(len(association_matrix)):
        width += np.sum(association_matrix[m] ** 2) ** 0.5  # 按定义用二阶范数计算width parameter
    print('width', width)
    # 计算association_matrix
    count = 0
    for m in range(len(association_matrix)):
        for n in range(len(association_matrix)):
            disease_similarity[m, n] = math.exp((np.sum((association_matrix[m] - association_matrix[n]) ** 2) ** 0.5
                                                 * len(association_matrix) / width) * (
                                                    -1))  # 计算不同行（disease）之间的二阶范数，这个定义和雷秀娟的不一致，雷秀娟的定义更加准确，所以改为雷秀娟的了
            if m != n and disease_similarity[m, n] == 1:  # 自己加的m!=n and
                disease_similarity[m, n] = 0.8  # 这里是一个大问题，两个向量相同可以说它有一定相关度，可是计算出相关度等于1又不合理，只能定义一个值
                # 自己备注：为1的还有对角线元素，疾病对于自身的高斯核相似性为1，为什么要换成0.8,所以只替换非对角线元素
    print('disease_similarity', disease_similarity)
    disease_Gaus_similarity = np.zeros([len(association_matrix), len(association_matrix)])
    for m in range(len(association_matrix)):
        for n in range(len(association_matrix)):
            disease_Gaus_similarity[m][n] = 1/(1 + math.exp(-15*disease_similarity[m, n] + math.log(9999)))
    print('disease_Gaus_similarity', disease_Gaus_similarity)

    # 保存结果
    result = pd.DataFrame(disease_Gaus_similarity)
    result.to_excel('output/disease_Gaussian_Simi.xlsx', index=False)  # index=False设置不生成索引列
    return disease_Gaus_similarity


# print(disease_Gaussian_Simi())

'''代谢物相似性'''


# 代谢物功能相似性
def metabolite_func_simi():
    # 读取数据
    ass_mat = pd.read_excel('output/association_matrix.xlsx').values
    # print('ass_mat', ass_mat, ass_mat.shape)
    re_metabolite = pd.read_excel('data/disease-metabolite.xlsx', sheet_name='代谢物_2315个', header=0)
    unique_metabolite = re_metabolite['Metabolite ID'].tolist()
    # print(unique_metabolite)
    # 导入语义相似性矩阵
    semantic_mat = pd.read_excel('output/disease_Semantic_simi.xlsx').values
    # print('semantic_mat', semantic_mat)
    # print(semantic_mat.shape)

    # 寻找代谢物相关的疾病集合
    unique_metabolite_1 = {}
    for i in range(ass_mat.shape[0]):
        list = []
        for j in range(ass_mat.shape[1]):
            if ass_mat[i][j] == 1:
                list.append(j)
            unique_metabolite_1[i] = list
    # print('unique_metabolite_1', unique_metabolite_1)

    # 将疾病语义矩阵转化为字典存储
    unique_disease = {}
    for i in range(ass_mat.shape[1]):
        list_1 = []
        for j in range(ass_mat.shape[1]):
            list_1.append(semantic_mat[i][j])
            unique_disease[i] = list_1
    # print('unique_disease', unique_disease)

    list_result = []
    for key, value in unique_metabolite_1.items():
        # print(str(key)+  " : " + str(value) +'\n')
        list_small = []  # 用于存储一种代谢物和其他所有代谢物的关系
        for key2, value2 in unique_metabolite_1.items():
            dict_max = {}
            # 遍历第一个代谢物对应疾病list
            for jb1 in value:
                list_jb = []
                # 遍历第二个代谢物对应疾病list
                for jb2 in value2:
                    list_jb.append(unique_disease[jb1][jb2])
                max_val = max(list_jb)
                dict_max[jb1] = max_val
                # 插入对应列表对应索引的位置
            list_small.append(dict_max)
        list_result.append(list_small)
    # print('************************疾病关系最大值:' + str(list_result))

    # 将list_result转化为列表嵌套列表
    list2 = []
    for i in range(ass_mat.shape[0]):
        # print('1111',list_result[i])
        list1 = []
        for j in range(ass_mat.shape[0]):
            # print('22222',list_result[i][j])
            # print('33333',sum(list_result[i][j].values()))
            list1.append(sum(list_result[i][j].values()))
        # print('list1',list1)
        list2.append(list1)
    print('list2', list2)

    # 将代谢物疾病集合转化为列表存储
    list3 = []
    for key1, value1 in unique_metabolite_1.items():
        list3.append(value1)
    print('list3', list3)

    # 计算代谢物功能相似性的分子，分母和最终结果，并存入矩阵中
    func_simi_matrix = np.zeros([ass_mat.shape[0], ass_mat.shape[0]])
    # print(func_simi_matrix)

    for i in range(ass_mat.shape[0]):
        # print('44444',list2[i])
        for j in range(ass_mat.shape[0]):
            sum1 = list2[i][j] + list2[j][i]  # sum1为分子
            # print('sum1',sum1)
            sum2 = len(list3[i] + list3[j])  # sum2为分母
            # print('sum2',sum2)
            func_simi = sum1 / sum2
            # print(func_simi)
            if i != j and func_simi == 1:  # 非对角线元素如果计算结果为1，替换为0.8
                func_simi = 0.8
            func_simi_matrix[i][j] = round(func_simi, 5)  # 给矩阵赋值
    print(func_simi_matrix)
    result = pd.DataFrame(func_simi_matrix)
    result.to_excel('output/metabolite_func_simi.xlsx', index=False)
    return func_simi_matrix


# print(metabolite_func_simi())

# 代谢物高斯核相似性
def metabolite_Gaussian_Simi():
    # 读取数据
    association_matrix = pd.read_excel('output/association_matrix.xlsx').values  # .values没太懂
    print(association_matrix, association_matrix.shape[0])
    # 每列表示一个病和各metabolite是否有联系，每行表示一个metabolite和每个病是否有联系
    metabolite_similarity = np.zeros([len(association_matrix), len(association_matrix)])  # 1620种metabolite之间的相似度，初始化矩阵

    width = 0
    for m in range(len(association_matrix)):
        width += np.sum(association_matrix[m] ** 2) ** 0.5  # 按定义用二阶范数计算width parameter
    print('width', width)

    # 计算association_matrix
    count = 0
    for m in range(len(association_matrix)):
        for n in range(len(association_matrix)):
            metabolite_similarity[m, n] = math.exp((np.sum((association_matrix[m] - association_matrix[n]) ** 2) ** 0.5
                                                    * len(association_matrix) / width) * (-1))
            if m != n and metabolite_similarity[m, n] == 1:
                metabolite_similarity[m, n] = 0.8  # 这里是一个大问题，两个向量相同可以说它有一定相关度，可是计算出相关度等于1又不合理，只能定义一个值

    # 保存结果
    result = pd.DataFrame(metabolite_similarity)
    result.to_excel('output/metabolite_Gaussian_Simi.xlsx', index=False)
    return metabolite_similarity


# print(metabolite_Gaussian_Simi())

'''疾病相似性融合'''
# 读取数据
sematic_simi = pd.read_excel('output/disease_Semantic_simi.xlsx', header=0)  # header=0删除了第一行
dise_Gau_Simi = pd.read_excel('output/disease_Gaussian_Simi.xlsx', header=0)

print('sematic_similarity', sematic_simi, sematic_simi.shape)
print('disease_Gaussian_Similarity', dise_Gau_Simi, dise_Gau_Simi.shape)

# 构建疾病相似性网络
diease_ntework = np.zeros([len(sematic_simi), len(sematic_simi)])  # 构建疾病的初始网络
for i in range(len(sematic_simi)):
    for j in range(len(sematic_simi)):
        if sematic_simi[i][j] == 0:
            diease_ntework[i][j] = round(dise_Gau_Simi[i][j], 5)
        else:
            diease_ntework[i][j] = round((sematic_simi[i][j] * 0.9 + dise_Gau_Simi[i][j] * 0.1), 5)
print(diease_ntework)
result = pd.DataFrame(diease_ntework)
result.to_excel('output/diease_network_simi.xlsx', index=False)

