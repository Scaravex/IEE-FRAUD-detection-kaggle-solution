
import re
import pandas as pd
import numpy as np
from scipy import stats
import datetime
import hashlib

def setbrowser(df):
    df.loc[df["id_31"]=="samsung browser 7.0",'lastest_browser']=1
    df.loc[df["id_31"]=="opera 53.0",'lastest_browser']=1
    df.loc[df["id_31"]=="mobile safari 10.0",'lastest_browser']=1
    df.loc[df["id_31"]=="google search application 49.0",'lastest_browser']=1
    df.loc[df["id_31"]=="firefox 60.0",'lastest_browser']=1
    df.loc[df["id_31"]=="edge 17.0",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 69.0",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 67.0 for android",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 63.0 for android",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 63.0 for ios",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 64.0",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 64.0 for android",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 64.0 for ios",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 65.0",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 65.0 for android",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 65.0 for ios",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 66.0",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 66.0 for android",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 66.0 for ios",'lastest_browser']=1

    return df


def clean_id31(df):
    df = df.copy()
    df['id_31'] = df['id_31'].str.replace("([0-9\.])", "")
    df['id_31'][df['id_31'].str.contains('chrome', regex=False)==True] = 'chrome'
    df['id_31'][df['id_31'].str.contains('Samsung', regex=False)==True] = 'Samsung'
    df['id_31'][df['id_31'].str.contains('samsung', regex=False)==True] = 'Samsung'
    df['id_31'][df['id_31'].str.contains('firefox', regex=False)==True] = 'firefox'
    df['id_31'][df['id_31'].str.contains('safari', regex=False)==True] = 'safari'
    df['id_31'][df['id_31'].str.contains('opera', regex=False)==True] = 'opera'
    df['id_31'] = df['id_31'].str.replace(" ", "")
    df.loc[df['id_31'].str.contains('Generic/Android', na=False), 'id_31']  = 'Android'
    df.loc[df['id_31'].str.contains('androidbrowser', na=False), 'id_31']  = 'Android'
    df.loc[df['id_31'].str.contains('androidwebview', na=False), 'id_31']  = 'Android'
    df.loc[df['id_31'].str.contains('android', na=False), 'id_31']  = 'Android'
    df.loc[df['id_31'].str.contains('chromium', na=False), 'id_31']  = 'chrome'
    df.loc[df['id_31'].str.contains('google', na=False), 'id_31']  = 'chrome'
    df.loc[df['id_31'].str.contains('googlesearchapplication', na=False), 'id_31']  = 'chrome'
    df.loc[df['id_31'].str.contains('iefordesktop', na=False), 'id_31']  = 'ie'
    df.loc[df['id_31'].str.contains('iefortablet', na=False), 'id_31']  = 'ie'
    df.loc[df.id_31.isin(df.id_31.value_counts()[df.id_31.value_counts() < 20].index), 'id_31'] = "rare"
    return df

def general_email(train, test):
    emails = {'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum', 'scranton.edu': 'other', 'optonline.net': 'other',
              'hotmail.co.uk': 'microsoft', 'comcast.net': 'other', 'yahoo.com.mx': 'yahoo', 'yahoo.fr': 'yahoo',
              'yahoo.es': 'yahoo', 'charter.net': 'spectrum', 'live.com': 'microsoft', 'aim.com': 'aol', 'hotmail.de': 'microsoft',
              'centurylink.net': 'centurylink', 'gmail.com': 'google', 'me.com': 'apple', 'earthlink.net': 'other', 
              'gmx.de': 'other', 'web.de': 'other', 'cfl.rr.com': 'other', 'hotmail.com': 'microsoft', 'protonmail.com': 'other',
              'hotmail.fr': 'microsoft', 'windstream.net': 'other', 'outlook.es': 'microsoft', 'yahoo.co.jp': 'yahoo',
              'yahoo.de': 'yahoo', 'servicios-ta.com': 'other', 'netzero.net': 'other', 'suddenlink.net': 'other',
              'roadrunner.com': 'other', 'sc.rr.com': 'other', 'live.fr': 'microsoft', 'verizon.net': 'yahoo',
              'msn.com': 'microsoft', 'q.com': 'centurylink', 'prodigy.net.mx': 'att', 'frontier.com': 'yahoo',
              'anonymous.com': 'other', 'rocketmail.com': 'yahoo', 'sbcglobal.net': 'att', 'frontiernet.net': 'yahoo',
              'ymail.com': 'yahoo', 'outlook.com': 'microsoft', 'mail.com': 'other', 'bellsouth.net': 'other',
              'embarqmail.com': 'centurylink', 'cableone.net': 'other', 'hotmail.es': 'microsoft', 'mac.com': 'apple',
              'yahoo.co.uk': 'yahoo', 'netzero.com': 'other', 'yahoo.com': 'yahoo', 'live.com.mx': 'microsoft', 'ptd.net': 'other',
              'cox.net': 'other', 'aol.com': 'aol', 'juno.com': 'other', 'icloud.com': 'apple'}
    us_emails = ['gmail', 'net', 'edu']
    for c in ['P_emaildomain', 'R_emaildomain']:
        train[c + '_bin'] = train[c].map(emails)
        test[c + '_bin'] = test[c].map(emails)
        
        train[c + '_suffix'] = train[c].map(lambda x: str(x).split('.')[-1])
        test[c + '_suffix'] = test[c].map(lambda x: str(x).split('.')[-1])
        
        train[c + '_suffix'] = train[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')
        test[c + '_suffix'] = test[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')
        
    return (train, test)

def make_day_feature(df, offset=0.58, tname='TransactionDT'):
    """
    Creates a day of the week feature, encoded as 0-6.
    """
    days = df[tname] / (3600 * 24)
    encoded_days = np.floor(days - 1 + offset) % 7
    return encoded_days

def make_hour_feature(df, tname='TransactionDT'):
    """
    Creates an hour of the day feature, encoded as 0-23.
    """
    hours = df[tname] / (3600)
    encoded_hours = np.floor(hours) % 24
    return encoded_hours


def make_pdc_amt_ratio(df):
    df_product_aveAmt = df.groupby(['ProductCD'])['TransactionAmt'].agg(['mean'])
    df_product_aveAmt.reset_index(inplace=True)
    df_ratio = pd.merge(df[['TransactionID','ProductCD',
                                             'TransactionAmt','isFraud']],
                           df_product_aveAmt,on='ProductCD',how='left')
    
    return df_ratio['TransactionAmt']/df_ratio['mean']

# Get card id for every transaction
def define_indexes(df):
    ''' Get Card id for every transaction'''
    def correct_card_id(x):
        x=x.replace('.0','')
        x=x.replace('-999','nan')
        return x

    cards_cols= ['card1', 'card2', 'card3', 'card5']
    for card in cards_cols:
        if '1' in card:
            df['card_id']= df[card].map(str)
        else :
            df['card_id']+= ' '+df[card].map(str)
    df['card_id'] = df['card_id'].apply(correct_card_id)
    return df

def device_hash(x):
    '''find unique device ID'''
    s =  str(x['id_30'])+str(x['id_31'])+str(x['id_32'])+str(x['id_33'])+str( x['DeviceType'])+ str(x['DeviceInfo'])
    h = hashlib.sha256(s.encode('utf-8')).hexdigest()[0:15]
    return h

def interaction(train, test):
    #function that allows to identify interactions between some features
    def features_interaction(df, feature_1, feature_2):
        return df[feature_1].astype(str) + '_' + df[feature_2].astype(str)
    #below some examples --> to be improved
    features_interactions = [
        'id_02__id_20',
        'id_02__D8',
        'addr1__addr2',
        'D11__DeviceInfo',
        'P_emaildomain_bin__C2',
        'card2__id_20',
        'addr1__card1',
        'TransactionAmt_lastdigit__ProductCD',
        'is_null_device_info__is_null_id_30'
    ]

    for new_feature in features_interactions:
        feature_1, feature_2 = new_feature.split('__')
        train[new_feature] = features_interaction(train, feature_1, feature_2)
        test[new_feature] = features_interaction(test, feature_1, feature_2)
        
    return (train, test)

def check_appearance(data, col_name):
    #check past appearence of a column
    data = data.copy()
    col_prev = col_name + "_PREV"
    col_new = col_name + "_PAST"
    data.reset_index(level=0, inplace=True)
    cp = data[[col_name]].copy()
    #faccio una copia della colonna e la shifto di 1
    cp = cp.shift(1)
    cp.rename(columns={col_name: col_prev}, inplace=True)
    data = data.join(cp, how="left")
    data[col_new] = (data[col_name] == data[col_prev]).astype(int)
    data.drop(col_prev, axis="columns", inplace=True)
    data = data.set_index('TransactionID')
    
    return data

def date_enricher(df_total):
    #enrich date
    START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')
    df_total['DT'] = df_total['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds = x)))
    df_total['DT_hour'] = df_total['DT'].dt.hour
    df_total['DT_day'] = df_total['DT'].dt.day
    df_total['DT_month'] = df_total['DT'].dt.month
    df_total['DT_year'] = df_total['DT'].dt.year
    
    return df_total
    

def make_count_1(feature, df):
   temp = df[feature].value_counts(dropna=False)
   new_feature = feature + '_count'
   df[new_feature] = df[feature].map(temp)

def make_count_2(feature, train, test):
    temp = pd.concat([train[feature], test[feature]], ignore_index=True).value_counts(dropna=True)
    new_feature = feature + '_count'
    train[new_feature] = train[feature].map(temp)
    test[new_feature] = test[feature].map(temp)
    
def flag_previous_transaction_below_amt(df, amt, df_final):   
    # Check if for a specific card id I had transations below 10$ in the past
    df = df.copy()
    df.reset_index(level=0, inplace=True)
    df_small_amt = df.loc[df['TransactionAmt']<=amt, ['card_id','TransactionDT', 'TransactionID']]   
    df_all = df[['card_id','TransactionDT','TransactionID']].merge(df_small_amt, how='left', left_on='card_id', right_on='card_id')
    #tieni solo quando la funzione è precedente a quello in analisi (/& maggiore d_y -30 gg)
    df_all = df_all.loc[(df_all.TransactionDT_x > df_all.TransactionDT_y) &  (df_all.TransactionDT_x>df_all.TransactionDT_x-60*24*30)]
    col_name = 'hasPreviousLesser'+format(amt)
    df_all[col_name] = 1
    df_all.rename(columns={'TransactionID_x': 'TransactionID'}, inplace=True)
    df_all = df_all[['TransactionID', col_name]].drop_duplicates()
    df_all = df_all.set_index('TransactionID')
    res = df_final.merge(df_all, how='left', on = "TransactionID")
    res[col_name].fillna(0, inplace=True)
    
    return(res)
    
def feature_engineering(train, test):
    '''
    Function that does several things, such as:
    #Data Augmentation
    #Combination of features
    #eliminating unuseful columns
    '''
    #first part: working on emails    
    #Drop columns with more than 90 precent of null values 
    train_missing_values = train.isnull().sum().sort_values(ascending=False) / len(train)
    test_missing_values = test.isnull().sum().sort_values(ascending=False) / len(test)
    train_missing_values = [str(x) for x in train_missing_values[train_missing_values > 0.85].keys()]
    test_missing_values = [str(x) for x in test_missing_values[test_missing_values > 0.85].keys()]
    dropped_columns = train_missing_values + test_missing_values
    #the newly created columns won't be in the dropped list
    #drop columns that have more than 90 precent of a same value
    dropped_columns = dropped_columns + [col for col in train.columns if train[col].value_counts(dropna=False, normalize=True).values[0] > 0.85]
    dropped_columns = dropped_columns + [col for col in test.columns if test[col].value_counts(dropna=False, normalize=True).values[0] > 0.85]
    dropped_columns.remove('isFraud')

    # stats.pearsonr(train["is_proton_mail"], train["isFraud"]) #coeff: 0.04 ;p-value: 0 macchina
    #Now creating emaildomain_bin; emaildomain_suffix for both R and P
    (train, test) = general_email(train, test)
    #flag if the email is the same in R_ and P_ --> an automatic system cannot make decision might put same email domain
    train['is_same_email_domain'] = train[['R_emaildomain_bin','P_emaildomain_bin']].where(train.R_emaildomain_bin==train.P_emaildomain_bin).notna().iloc[:, 0]
    test['is_same_email_domain']  = test[['R_emaildomain_bin','P_emaildomain_bin']].where(test.R_emaildomain_bin==test.P_emaildomain_bin).notna().iloc[:, 0]
    #stats.pearsonr(train["is_same_email_domain"], train["isFraud"]) #coeff: 0.27
    
    #TO DO: is email_domain_rare --> aggiunto count sotto
    #TO DO: drop P_emaildomain; R_emaildomain
    #add card ID and therefore unique identifier. It creates "card_id"
    train = define_indexes(train)
    test  = define_indexes(test)
    ##add device type
    train['device_hash'] = train.apply(lambda x: device_hash(x), axis=1)
    test['device_hash']  =  test.apply(lambda x: device_hash(x), axis=1) 

    #second part: Working on device
    #Long Names can appear strange, made on purpose complicated --> 
    train["len_DeviceInfo"] = train ["DeviceInfo"].apply(lambda x: np.nan if (pd.isnull(x)) else len(str(x)))
    test["len_DeviceInfo"]  =  test ["DeviceInfo"].apply(lambda x: np.nan if (pd.isnull(x)) else len(str(x)))
    # stats.pearsonr(train["len_DeviceInfo"], train["isFraud"]) #0.22 --> mettendo 0 invece che np.nan
    #often, there is Build with frauds
    train['device_Build'] = train ["DeviceInfo"].apply(lambda x: False if (pd.isnull(x)) else (x.find("Build"))>=0)
    test['device_Build']  =  test ["DeviceInfo"].apply(lambda x: False if (pd.isnull(x)) else (x.find("Build"))>=0)
    # stats.pearsonr(train["device_Build"], train["isFraud"]) #0.11
    train['is_null_id_30'] = train ["id_30"].apply(lambda x: False if (pd.isnull(x)) else True)
    test ['is_null_id_30'] =  test ["id_30"].apply(lambda x: False if (pd.isnull(x)) else True)

    train['is_null_device_info'] = train ["DeviceInfo"].apply(lambda x: False if (pd.isnull(x)) else True)
    test['is_null_device_info']  =  test ["DeviceInfo"].apply(lambda x: False if (pd.isnull(x)) else True)
    # stats.pearsonr(train["is_null_device_info"], train["isFraud"]) #0.10



    # Check past appearance for combination of card id and p email domain
    train['card_id_P_emaildomain'] = train['card_id']+'--'+train['P_emaildomain']
    train.sort_values(by= ["card_id_P_emaildomain", "TransactionDT"], inplace= True)
    train = check_appearance(train, 'card_id_P_emaildomain')
    #stats.pearsonr(train["card_id_P_emaildomain"], train["isFraud"]) #0.10
    # Check past appearance for combination of card id and p email domain
    test['card_id_P_emaildomain'] = test['card_id']+'--'+test['P_emaildomain']
    test.sort_values(by= ["card_id_P_emaildomain", "TransactionDT"], inplace= True)
    test = check_appearance(test, 'card_id_P_emaildomain') 
    
    train['DeviceInfo__P_emaildomain_bin'] = train['DeviceInfo']+'--'+train['P_emaildomain']
    train.sort_values(by= ["DeviceInfo__P_emaildomain_bin", "TransactionDT"], inplace= True)
    train = check_appearance(train, 'DeviceInfo__P_emaildomain_bin')
    #stats.pearsonr(train["card_id_P_emaildomain"], train["isFraud"]) #0.10
    # Check past appearance for combination of card id and p email domain
    test['DeviceInfo__P_emaildomain_bin'] = test['DeviceInfo']+'--'+test['P_emaildomain']
    test.sort_values(by= ["DeviceInfo__P_emaildomain_bin", "TransactionDT"], inplace= True)
    test = check_appearance(test, 'DeviceInfo__P_emaildomain_bin')     
    
    # Check past appearence for combination of p email domain and r email domain
    train['P_R_emaildomain'] = train['P_emaildomain']+'--'+train['R_emaildomain']
    train.sort_values(by= ["P_R_emaildomain", "TransactionDT"], inplace= True)
    train = check_appearance(train, 'P_R_emaildomain')      
    # Check past appearence for combination of p email domain and r email domain
    test['P_R_emaildomain'] = test['P_emaildomain']+'--'+test['R_emaildomain']
    test.sort_values(by= ["P_R_emaildomain", "TransactionDT"], inplace= True)
    test = check_appearance(test, 'P_R_emaildomain')
    
    train = train.drop(['P_R_emaildomain','card_id_P_emaildomain','DeviceInfo__P_emaildomain_bin'],axis=1)
    test =  test.drop(['P_R_emaildomain','card_id_P_emaildomain','DeviceInfo__P_emaildomain_bin'],axis=1)

    #third part: working on browsers --> id_31
    #idea that a modern browser is symptomatic of automatic systems    
    a = np.zeros(train.shape[0])
    train["lastest_browser"] = a
    a = np.zeros(test.shape[0])
    test ["lastest_browser"] = a    
    train = setbrowser(train)
    test  = setbrowser(test)
    # stats.pearsonr(train["device_Build"], train["isFraud"]) #0.11
    #often, there is no specific version with frauds
    train['browser_generic'] = train ["id_31"].apply(lambda x: False if (pd.isnull(x)) else (x.find("generic"))>=0)
    test['browser_generic']  =  test ["id_31"].apply(lambda x: False if (pd.isnull(x)) else (x.find("generic"))>=0)
    # stats.pearsonr(train["browser_generic"], train["isFraud"]) #0.065; p-value 0
    #idea that no hacker wanna use internet explore 
    train = clean_id31(train)
    test =  clean_id31(test)
    #stats.pearsonr(train["is_browser_ie"], train["isFraud"]) #coeff: '0.03 p-value 0
    # proved important feature
    train['screen_width']  = train['id_33'].str.split('x', expand=True)[0]
    train['screen_height'] = train['id_33'].str.split('x', expand=True)[1]
    test['screen_width']   =  test['id_33'].str.split('x', expand=True)[0]
    test['screen_height']  =  test['id_33'].str.split('x', expand=True)[1]
    
    #4--> working on date information
    #adding weekday feature
    train["weekday"] = make_day_feature(train).astype(str)
    test["weekday"] = make_day_feature(test).astype(str)
    #stats.pearsonr(train["weekday"], train["isFraud"]) #not so significant
    #adding hours feature
    train["hour"] = (np.floor(train["TransactionDT"] / 3600) % 24)
    test ["hour"] = (np.floor(test["TransactionDT"] / 3600) % 24)
    #stats.pearsonr(train["hour"], train["isFraud"]) #coeff: -0.013
    #add features to see if a relationship between day and hour exist
    #cancel if not needed
    train["day_hour"] = train["weekday"]+'_'+train["hour"].astype(str)
    test ["day_hour"] = test ["weekday"]+'_'+test ["hour"].astype(str)

    ###IDentificare dati intorno alle ore speccate    
    ##############################################
    #############################################
    '''
    def dt_features(data):
        data = data.copy()
        start_dt = data['TransactionDT'].min()
        data['TransactionDT_norm_days'] = ((data['TransactionDT'] - start_dt)/3600)/24
        data['TransactionDT_diff'] = data['TransactionDT_norm'].diff().fillna(0)
        data['TransactionDT_diff_days'] = data['TransactionDT_diff']/24
        ##it could be useful if we can use to study the difference inside the same dataset
        return data
    '''
    #train ['TransactionDT_norm_days'] = ((train['TransactionDT'] - train['TransactionDT'].min())/3600)/24
    #test  ['TransactionDT_norm_days'] = ((test ['TransactionDT'] - test ['TransactionDT'].min())/3600)/24
    

    train = date_enricher(train)
    test  = date_enricher(test)    
        
    #5 --> Other minors
    #Decimals might be specific
    train['TransactionAmt_decimal'] = ((train['TransactionAmt'] - train['TransactionAmt'].astype(int)) * 1000).astype(int)
    test['TransactionAmt_decimal'] = ((test['TransactionAmt'] - test['TransactionAmt'].astype(int)) * 1000).astype(int)
    #stats.pearsonr(train["TransactionAmt_decimal"], train["isFraud"]) #coeff: -0.05
    ##longer length is for foreign exchange
    train['_amount_decimal_len'] = train['TransactionAmt'].apply(lambda x: len(re.sub('0+$', '', str(x)).split('.')[1]))
    test['_amount_decimal_len'] = test['TransactionAmt'].apply(lambda x: len(re.sub('0+$', '', str(x)).split('.')[1]))

    #taking l'ultima cifra come feature    
    train["TransactionAmt_lastdigit"] = (train["TransactionAmt"].mod(1) * 100).apply(lambda x: str(int(x))[-1])
    test["TransactionAmt_lastdigit"]  = ( test["TransactionAmt"].mod(1) * 100).apply(lambda x: str(int(x))[-1])
    train['TransactionAmtcard1mean'] = train.groupby(['card_id'])['TransactionAmt'].transform('mean')
    test['TransactionAmtcard1mean']  =  test.groupby(['card_id'])['TransactionAmt'].transform('mean')
    #stats.pearsonr(train["TransactionAmtcard1mean"], train["isFraud"]) #coeff: -0.013

    # Check past appearance for combination of card id and p email domain
    train['card_id_device_hash'] = train['card_id']+'--'+train['device_hash']
    train.sort_values(by= ["card_id_device_hash", "TransactionDT"], inplace= True)
    train = check_appearance(train, 'card_id_device_hash')
    #stats.pearsonr(train["card_id_P_emaildomain"], train["isFraud"]) #0.10
    # Check past appearance for combination of card id and p email domain
    test['card_id_device_hash'] = test['card_id']+'--'+test['device_hash']
    test.sort_values(by= ["card_id_device_hash", "TransactionDT"], inplace= True)
    test = check_appearance(test, 'card_id_device_hash') 

    train = train.drop('card_id_device_hash',axis=1)
    test = test.drop('card_id_device_hash',axis=1)
        
    train['card_mv_day_fq'] = train.groupby(['card_id','DT_day','DT_month','DT_year'])['card_id'].transform('count')
    test ['card_mv_day_fq'] = test.groupby(['card_id','DT_day','DT_month','DT_year'])['card_id'].transform('count')
    #stats.pearsonr(train["card_mv_day_fq"], train["isFraud"]) #coeff: -0.015
    
    
    train['pdc_hour_Amt_mean'] = train.groupby(['ProductCD','DT_hour','DT_day','DT_month','DT_year'])['TransactionAmt'].transform('mean')
    test['pdc_hour_Amt_mean']  =  test.groupby(['ProductCD','DT_hour','DT_day','DT_month','DT_year'])['TransactionAmt'].transform('mean')
    #stats.pearsonr(train["card_mv_day_fq"], train["isFraud"]) #coeff: -0.013

    # this feature lead to over fitting
    #train.drop('card_id',axis=1,inplace=True)
    #df_total.drop('hour',axis=1,inplace=True)
    train.drop(['DT','DT_day','DT_month','DT_year'],axis=1,inplace=True)
    test.drop(['DT','DT_day','DT_month','DT_year'],axis=1,inplace=True)
      
    # proved important feature
    train['pdc_amt_ratio'] = train['TransactionAmt']/train.groupby('ProductCD')['TransactionAmt'].transform('mean')
    test['pdc_amt_ratio'] = test['TransactionAmt']/test.groupby('ProductCD')['TransactionAmt'].transform('mean')
    #stats.pearsonr(train["pdc_amt_ratio"], train["isFraud"]) #coeff: -0.015

    #Infine come identificare transazioni che hanno avuto un precedente con amount superiore a soglia stabilita
    # Check if for given card_id there are past transactions below a certain amount
    
    
    train2 = flag_previous_transaction_below_amt(df=train[['card_id','TransactionDT','TransactionAmt']],amt=10,df_final=train)
    test2  = flag_previous_transaction_below_amt(df=test[['card_id','TransactionDT','TransactionAmt']],amt=10,df_final=test)
    
    train['hasPreviousLesser10'] = train2['hasPreviousLesser10']
    test['hasPreviousLesser10'] = test2['hasPreviousLesser10']
    del train2
    del test2

    train2 = flag_previous_transaction_below_amt(df=train[['card_id','TransactionDT','TransactionAmt']],amt=2,df_final=train)
    test2  = flag_previous_transaction_below_amt(df=test[['card_id','TransactionDT','TransactionAmt']],amt=2,df_final=test)    
    train['hasPreviousLesser2'] = train2['hasPreviousLesser2']
    test['hasPreviousLesser2'] = test2['hasPreviousLesser2']
    del train2
    del test2

    #6 --> Feature Interaction
    #analyzing interactions should improve the model
    (train, test) = interaction(train, test)
    
    train = train.drop(['is_null_id_30','is_null_device_info'],axis = 1)
    test = test.drop(['is_null_id_30','is_null_device_info'], axis = 1)
    #7 Counter

    
    make_count_1("device_hash", train)
    make_count_1("device_hash", test)
    #This didn't help much, it actually worsen the whole
    #counting email domain, implies if an email is famous or not
    for feature in ['card1','addr1', 'id_36','P_emaildomain_bin','DeviceInfo']:
        make_count_2(feature, train, test)

    ''' algorithmic feature engineering
    from sklearn.decomposition import KernelPCA
    kpca = KernelPCA(n_components = 1, kernel = 'rbf')
    train['KPCA'] = kpca.fit_transform(train[['TransactionAmt_decimal','hour']])
    
    from sklearn.decomposition import TruncatedSVD

    n_comp = 2
    svd = TruncatedSVD(n_components=n_comp, algorithm='arpack')
    svd.fit(train[['TransactionAmt','TransactionAmt_decimal','hour']])
    print(svd.explained_variance_ratio_.sum())
    
    train['TSVD_test2'] = svd.transform(train[['TransactionAmt','TransactionAmt_decimal','hour']])[:,1]
    #stats.pearsonr(train["TSVD_test2"], train["isFraud"]) #coeff: -0.05

    test_features = svd.transform(train[['TransactionAmt_decimal','hour']])
    
    '''
    #finally remove the dropped columns
    dropped_columns.sort()
    train.drop(dropped_columns, axis=1, inplace=True)
    test.drop(dropped_columns, axis=1, inplace=True)
    print("dropped columns: ",len(dropped_columns))

    return (train, test)

