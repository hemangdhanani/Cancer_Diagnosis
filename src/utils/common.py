def get_gv_fea_dict(train_df, alpha, feature, df):
    value_count = train_df[feature].value_counts()
    gv_dict = dict()
        
    for i, denominator in value_count.items():
        vec = []
        for k in range(1,10):            
            cls_cnt = train_df.loc[(train_df['Class']==k) & (train_df[feature]==i)]
            vec.append((cls_cnt.shape[0] + alpha*10)/ (denominator + 90*alpha))        
        gv_dict[i]=vec
    return gv_dict

# Get Gene variation feature
def get_gv_feature(train_df, alpha, feature, df):
    
    gv_dict = get_gv_fea_dict(train_df, alpha, feature, df)
    # value_count is similar in get_gv_fea_dict
    value_count = train_df[feature].value_counts()
    
    # gv_fea: Gene_variation feature, it will contain the feature for each feature value in the data
    gv_fea = []

    for index, row in df.iterrows():
        if row[feature] in dict(value_count).keys():
            gv_fea.append(gv_dict[row[feature]])
        else:
            gv_fea.append([1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9])
#            
    return gv_fea