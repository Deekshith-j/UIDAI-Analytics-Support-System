import pandas as pd
import numpy as np

def remove_duplicates(df):
    """
    Removes duplicate rows from the dataframe.
    """
    initial_count = len(df)
    df = df.drop_duplicates()
    final_count = len(df)
    print(f"Removed {initial_count - final_count} duplicate rows.")
    return df

def clean_state_name(state):
    """
    Standardizes state names.
    """
    if pd.isna(state) or str(state).strip() == '':
        return 'Unknown'
        
    state = str(state).lower().strip()
    
    # Common mappings
    mapping = {
        'andaman & nicobar islands': 'Andaman and Nicobar Islands',
        'a & n islands': 'Andaman and Nicobar Islands',
        'chhatisgarh': 'Chhattisgarh',
        'dadra & nagar haveli': 'Dadra and Nagar Haveli and Daman and Diu',
        'dadra and nagar haveli': 'Dadra and Nagar Haveli and Daman and Diu',
        'daman & diu': 'Dadra and Nagar Haveli and Daman and Diu',
        'daman and diu': 'Dadra and Nagar Haveli and Daman and Diu',
        'the dadra and nagar haveli and daman and diu': 'Dadra and Nagar Haveli and Daman and Diu',
        'jammu & kashmir': 'Jammu and Kashmir',
        'jammu and kashmir': 'Jammu and Kashmir',
        'jammu and  kashmir': 'Jammu and Kashmir',
        'orissa': 'Odisha',
        'pondicherry': 'Puducherry',
        'uttaranchal': 'Uttarakhand',
        'west bengal': 'West Bengal',
        'westbengal': 'West Bengal',
        'west  bengal': 'West Bengal',
        'west bangal': 'West Bengal',
        'west bengli': 'West Bengal',
        'delhi': 'Delhi', 
        'nct of delhi': 'Delhi'
    }
    
    if state in mapping:
        return mapping[state]
        
    # Heuristics
    state = state.replace('&', 'and')
    state = state.title()
    
    return state

def standardize_states(df, state_col='state'):
    """
    Applies state checking and cleaning.
    """
    if state_col in df.columns:
        df[state_col] = df[state_col].apply(clean_state_name)
        
        # Filter out likely garbage values (numbers, short strings that aren't states)
        # This list can be expanded based on the report
        filetered_states = [
            '100000', 'Balanagar', 'Jaipur', 'Madanapalle', 'Nagpur', 
            'Puttenahalli', 'Raja Annamalai Puram', 'Darbhanga'
        ]
        
        # We replace garbage with 'Unknown' or keep them? 
        # Better to mark them as Unknown or filtering them out if they are clearly not states.
        # Given "Jaipur" is a district, maybe the columns were shifted?
        # For now, let's keep them but maybe flag them? No, let's filter specific known bad values.
        
        df = df[~df[state_col].isin(filetered_states)]
        
    return df

def clean_dataset(df):
    """
    Wrapper for all cleaning steps.
    """
    df = remove_duplicates(df)
    df = standardize_states(df)
    return df
