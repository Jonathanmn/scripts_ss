def copy_and_rename_columns(df):
    """
    Copia las columnas 'Time', 'CH4_Avg', 'CO2_Avg', 'CO_Avg' de un DataFrame y las renombra a 'Time', 'CH4', 'CO2', 'CO'.
    
    Args:
        df (pd.DataFrame): El DataFrame original.
    
    Returns:
        pd.DataFrame: Un nuevo DataFrame con las columnas copiadas y renombradas.
    """
    # Copiar las columnas especificadas
    df_copy = df[['Time', 'CH4_Avg', 'CO2_Avg', 'CO_Avg']].copy()
    
    # Renombrar las columnas
    df_copy.rename(columns={'CH4_Avg': 'CH4', 'CO2_Avg': 'CO2', 'CO_Avg': 'CO'}, inplace=True)
    
    return df_copy


