def is_outlier(df, groupby, filter_column=None, window='7D',z_score_sensitivity=2) -> pd.DataFrame:  #takes as input: dataframe, groupby - for eg. spaces_left, filter_columns-used to filter by parking_id, window- for eg. uses last 30D for calculating anomalies, returns updated dataframe
  df=df.copy()

  if filter_column is None:
      temp_dataframe=df[groupby]



  else:
      temp_dataframe=df.groupby(filter_column)[groupby]

  df[groupby] = df[groupby].clip(lower=0)


  Q1 = temp_dataframe.transform(lambda x: x.rolling(window, min_periods=1).quantile(0.25))
  Q3 = temp_dataframe.transform(lambda x: x.rolling(window, min_periods=1).quantile(0.75))


  IQR = Q3 - Q1
  c = 2
  min_t = Q1 - c*IQR
  max_t = Q3 + c*IQR

  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR


  df['iqr_lower'] = lower_bound
  df['iqr_lower']=df['iqr_lower'].clip(lower=0)

  df['iqr_upper'] = upper_bound

  df['is_outlier_iqr'] = (df[groupby] < lower_bound) | (df[groupby] > upper_bound)

  df['is_event_iqr_outlier']=df['is_outlier_iqr'] & df['is_event']




  return df
