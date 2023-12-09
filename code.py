import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from statsmodels.stats.diagnostic import lilliefors
from statsmodels.stats.weightstats import ztest
from statsmodels.formula.api import ols
import statsmodels.api as sm
import statistics
import statsmodels.stats.multicomp as mc

###############################
#3

dx = pd.read_csv('DXSUM_PDXCONV_ADNIALL.csv', usecols = ['Phase', 'RID', 'DXCURREN'])
rid_dx = dx.dropna(subset='RID', inplace=True)
csf = pd.read_csv('UPENNMSMSABETA.csv', usecols = ['RID', 'ABETA42', 'ABETA40', 'ABETA38'])
csf.drop(csf[csf['RID'] == 975].index, inplace = True)

dx =  dx[dx['Phase'] == 'ADNI1']
dz = dx['DXCURREN'].to_numpy().astype('uint8')

rid_dx = dx['RID'].to_numpy()

rid_csf = csf['RID'].to_numpy()
ab_42 = csf['ABETA42'].to_numpy()
ab_40 = csf['ABETA40'].to_numpy()
ab_38 = csf['ABETA38'].to_numpy()

dicts = {}
for i in range(len(rid_dx)):
    dicts[rid_dx[i]] = dz[i]

NL_38 = []
MCI_38 = []
AD_38 = []

NL_42 = []
MCI_42 = []
AD_42 = []

#df = []

rid = list(set(rid_dx) & set(rid_csf))

for i in range(len(rid)):
    if(dicts.get(rid[i])):
        if(dicts.get(rid[i])==1):
            NL_38.append(ab_38[i])
            NL_42.append(ab_42[i])
            #df.append([1, ab_38[i], ab_42[i]])
        if(dicts.get(rid[i])==2):
            MCI_38.append(ab_38[i])
            MCI_42.append(ab_42[i])
            #df.append([2, ab_38[i], ab_42[i]])
        if(dicts.get(rid[i])==3):
            AD_38.append(ab_38[i])
            AD_42.append(ab_38[i])
            #df.append([3, ab_38[i], ab_42[i]])

NL_38 = np.array(NL_38)
MCI_38 = np.array(MCI_38)
AD_38 = np.array(AD_38)

NL_42 = np.array(NL_42)
MCI_42 = np.array(MCI_42)
AD_42 = np.array(AD_42)

fig0, ax0 = plt.subplots()
ax0.hist(ab_38, histtype='barstacked', rwidth=0.8, label='Total', color='cornflowerblue')
ax0.hist(NL_38, histtype='barstacked', rwidth=0.8, label='NL', color='palegreen')
ax0.hist(MCI_38, histtype='barstacked', rwidth=0.8, label='MCI', color='yellow')
ax0.hist(AD_38, histtype='barstacked', rwidth=0.8, label='AD', color='r')
ax0.set_label('Histograms')
ax0.set_xlabel('AB32(pg/ml)')
ax0.set_ylabel('Counts')
ax0.legend()

fig = plt.figure(figsize=(14,11))
ax = plt.subplot(2, 2, 1)
ax.set_title('Total')
sns.histplot(ab_38, kde=True, line_kws={'ls': '-', 'lw':2}, color='b', stat='density', facecolor='cornflowerblue', edgecolor='cornflowerblue', legend=True)
plt.axvline(ab_38.mean(), color='k', linestyle='-', linewidth=2)
plt.axvline(np.median(ab_38), color='purple', linestyle='-.', linewidth=2)
plt.axvline(statistics.mode(ab_38), color='dimgrey', linestyle='--', linewidth=2)
ax.set_xlabel('AB32(pg/ml)')
ax.legend(['PDF', 'Mean', 'Median', 'Mode'])

ax = plt.subplot(2, 2, 2)
ax.set_title('Normal')
sns.histplot(NL_38, kde=True, line_kws={'ls': '-', 'lw':2}, color='green', stat='density', facecolor='palegreen', edgecolor='palegreen', legend=True)
plt.axvline(NL_38.mean(), color='k', linestyle='-', linewidth=2)
plt.axvline(np.median(NL_38), color='purple', linestyle='-.', linewidth=2)
plt.axvline(statistics.mode(NL_38), color='dimgrey', linestyle='--', linewidth=2)
ax.set_xlabel('AB38(pg/ml)')
ax.legend(['PDF', 'Mean', 'Median', 'Mode'])

ax = plt.subplot(2, 2, 3)
ax.set_title('Mild Cognitive Impairment')
sns.histplot(MCI_38, kde=True, line_kws={'ls': '-', 'lw':2}, color='darkkhaki', stat='density', facecolor='yellow', edgecolor='yellow', legend=True)
plt.axvline(MCI_38.mean(), color='k', linestyle='-', linewidth=2)
plt.axvline(np.median(MCI_38), color='purple', linestyle='-.', linewidth=2)
plt.axvline(statistics.mode(MCI_38), color='dimgrey', linestyle='--', linewidth=2)
ax.set_xlabel('AB38(pg/ml)')
ax.legend(['PDF', 'Mean', 'Median', 'Mode'])

ax = plt.subplot(2, 2, 4)
ax.set_title('Alzheimer')
sns.histplot(AD_38, kde=True, line_kws={'ls': '-', 'lw':2}, color='maroon', stat='density', facecolor='r', edgecolor='r', legend=True)
plt.axvline(AD_38.mean(), color='k', linestyle='-', linewidth=2)
plt.axvline(np.median(AD_38), color='purple', linestyle='-.', linewidth=2)
plt.axvline(statistics.mode(AD_38), color='dimgrey', linestyle='--', linewidth=2)
ax.set_xlabel('AB38(pg/ml)')
ax.legend(['PDF', 'Mean', 'Median', 'Mode'])

###############################
#4

print('Skewness for Total data = ', stats.skew(ab_38))
print('Skewness for Normal = ', stats.skew(NL_38))
print('Skewness for Mild Cognitive Impairment = ', stats.skew(MCI_38))
print('Skewness for Alzheimer = ', stats.skew(AD_38))

###############################
#4

print('\nKolmogorov-Smirnov test:')
print('Total : statistic=', stats.kstest(ab_38, stats.norm.cdf(ab_38))[0], ', pvalue=', stats.kstest(ab_38, stats.norm.cdf(ab_38))[1])
print('Normal : statistic=', stats.kstest(NL_38, stats.norm.cdf(NL_38))[0], ', pvalue=', stats.kstest(NL_38, stats.norm.cdf(NL_38))[1])
print('Mild Cognitive Impairment : statistic=', stats.kstest(MCI_38, stats.norm.cdf(MCI_38))[0], ', pvalue=', stats.kstest(MCI_38, stats.norm.cdf(MCI_38))[1])
print('Alzheimer : statistic=', stats.kstest(AD_38, stats.norm.cdf(AD_38))[0], ', pvalue=', stats.kstest(AD_38, stats.norm.cdf(AD_38))[1])

print('\nLilliefors test:')
print('Total : statistic=', lilliefors(ab_38, dist='norm')[0], ', pvalue=', lilliefors(ab_38, dist='norm')[1])
print('Normal : statistic=', lilliefors(NL_38, dist='norm')[0], ', pvalue=', lilliefors(NL_38, dist='norm')[1])
print('Mild Cognitive Impairment : statistic=', lilliefors(MCI_38, dist='norm')[0], ', pvalue=', lilliefors(MCI_38, dist='norm')[1])
print('Alzheimer : statistic=', lilliefors(AD_38, dist='norm')[0], ', pvalue=', lilliefors(AD_38, dist='norm')[1])

print('\nJarque-Bera test:')
print('Total : ', stats.jarque_bera(ab_38))
print('Normal : ', stats.jarque_bera(NL_38))
print('Mild Cognitive Impairment : ', stats.jarque_bera(MCI_38))
print('Alzheimer : ', stats.jarque_bera(AD_38))

print('\nAnderson-Darling test:')
print('Total : statistic=', stats.anderson(ab_38)[0],', critical_values=', stats.anderson(ab_38)[1])
print('Normal :  statistic=', stats.anderson(NL_38)[0],', critical_values=', stats.anderson(NL_38)[1])
print('Mild Cognitive Impairment :  statistic=', stats.anderson(MCI_38)[0],', critical_values=', stats.anderson(MCI_38)[1])
print('Alzheimer :  statistic=', stats.anderson(AD_38)[0],', critical_values=', stats.anderson(AD_38)[1])

###############################
#7

print('\nZ Test:')
print('Normal Vs Mild Cognitive Impairment : statistic=', ztest(NL_38, MCI_38, alternative='two-sided', value=0)[0], ', P_value=', ztest(NL_38, MCI_38, alternative='two-sided', value=0)[1])
print('Normal Vs Alzheimer : statistic=', ztest(NL_38, AD_38, alternative='two-sided', value=0)[0], ', P_value=', ztest(NL_38, AD_38, alternative='two-sided', value=0)[1])
print('Alzheimer Vs Mild Cognitive Impairment : statistic=', ztest(AD_38, MCI_38, alternative='two-sided', value=0)[0], ', P_value=', ztest(AD_38, MCI_38, alternative='two-sided', value=0)[1])


print('\nNormal variance = ', np.var(NL_38))
print('Mild Cognitive Impairment variance = ', np.var(MCI_38))
print('Alzheimer variance = ', np.var(AD_38))

print('\nT Test:')
print('Normal Vs Mild Cognitive Impairment: ', stats.ttest_ind(NL_38, MCI_38, equal_var=True))
print('Normal Vs Alzheimer: ', stats.ttest_ind(NL_38, AD_38, equal_var=True))
print('Alzheimer Vs Mild Cognitive Impairment: ', stats.ttest_ind(NL_38, AD_38, equal_var=True))

###############################
#8

ar =  np.ones(len(NL_38), dtype= int) * 1
buf = {'Status': ar, 'Abeta38': NL_38}
df_NL_38 = pd.DataFrame(buf, dtype=np.int64)

ar =  np.ones(len(MCI_38), dtype= int) * 2
buf = {'Status': ar, 'Abeta38': MCI_38}
df_MCI_38 = pd.DataFrame(buf, dtype=np.int64)

ar =  np.ones(len(AD_38), dtype= int) * 3
buf = {'Status': ar, 'Abeta38': AD_38}
df_AD_38 = pd.DataFrame(buf, dtype=np.int64)

df_38 = pd.concat([df_NL_38, df_MCI_38, df_AD_38])

df_38['Status'].replace({1: 'NL', 2: 'MCI', 3: 'AD'}, inplace= True)

mod = ols('Abeta38 ~ Status', data=df_38).fit()
aov_table = sm.stats.anova_lm(mod, typ=1)
print('\nOne way ANOVA(AB38)\nANOVA Table:\n',aov_table)

fig = plt.figure()
ax = fig.subplots()
ax.set_title("Box Plot")
ax.boxplot([NL_38, MCI_38, AD_38], labels= ['NL', 'MCI', 'AD'], showmeans= True)
plt.xlabel('Health Status')
plt.ylabel('AB38(pg/ml)')

print('\nBONFERRONI test(Abeta38):')
comp = mc.MultiComparison(df_38['Abeta38'], df_38['Status'])
tbl, a1, a2 = comp.allpairtest(stats.ttest_ind, method= "bonf")
print(tbl)

####

ar =  np.ones(len(NL_42), dtype= int) * 1
buf = {'Status': ar, 'Abeta42': NL_42}
df_NL_42 = pd.DataFrame(buf, dtype=np.int64)

ar =  np.ones(len(MCI_42), dtype= int) * 2
buf = {'Status': ar, 'Abeta42': MCI_42}
df_MCI_42 = pd.DataFrame(buf, dtype=np.int64)

ar =  np.ones(len(AD_42), dtype= int) * 3
buf = {'Status': ar, 'Abeta42': AD_42}
df_AD_42 = pd.DataFrame(buf, dtype=np.int64)

df_42 = pd.concat([df_NL_42, df_MCI_42, df_AD_42])

df_42['Status'].replace({1: 'NL', 2: 'MCI', 3: 'AD'}, inplace= True)

mod = ols('Abeta42 ~ Status', data=df_42).fit()
aov_table = sm.stats.anova_lm(mod, typ=1)
print('\nOne way ANOVA(AB42)\nANOVA Table:\n',aov_table)

fig = plt.figure()
ax = fig.subplots()
ax.set_title("Box Plot")
ax.boxplot([NL_42, MCI_42, AD_42], labels= ['NL', 'MCI', 'AD'], showmeans= True)
plt.xlabel('Health Status')
plt.ylabel('AB42(pg/ml)')

print('\nBONFERRONI test(AB42):')
comp = mc.MultiComparison(df_42['Abeta42'], df_42['Status'])
tbl, a1, a2 = comp.allpairtest(stats.ttest_ind, method= "bonf")
print(tbl)

# df = pd.DataFrame(df, columns=['Status', 'Abeta38', 'Abeta42'], dtype=np.int64)
# model = ols('Status ~ C(Abeta38) + C(Abeta42) + C(Abeta38):C(Abeta42)', data=df).fit()
# anova_table = sm.stats.anova_lm(model, typ=2)
# print(anova_table)

plt.show()