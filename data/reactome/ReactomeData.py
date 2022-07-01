import pandas as pd

class ReactomeData():
    """ A class to subset and get reactome data.
    
    file1 = 'ReactomePathwaysRelation.txt'
    file2 = '../ms/....' - this should be a file with column "Proteins"
    file3 = 'UniProt2Reactome_All_Levels.txt"
    RD = ReactomeData(file1, file2, file3)
    RD.subset_species_reactome_data('Homo sapiens')
    RD.subset_on_protein_in_ms_data()
    RD.subset_hierarchies_on_file()
    RD.save_df("reactome", "HSA_ms_reactome.csv") 
    RD.save_df("path", 'HSA_ms_path.csv") 
    
    The two saved files are the reactome and path files for the reactome which can be used to
    create the neural network. 
    """
    
    def __init__(self,hierarchy_file_path, ms_data_file_path, reactome_all_data_file_path):
        self.hierarchy_file_path = hierarchy_file_path
        self.ms_data_file_path = ms_data_file_path
        self.reactome_all_data_file_path = reactome_all_data_file_path    
        self.ms_df = pd.read_csv(ms_data_file_path)
        self.reactome_df = pd.read_csv(self.reactome_all_data_file_path, index_col=False,
                        names = ['UniProt_id', 'Reactome_id', 'URL', 'Description','Evidence Code','Species'], sep="\t")
        self.path_df = pd.read_csv(hierarchy_file_path, names=['From','To'], index_col=False)
        
    def subset_species_sapiens_reactome_data(self, species = 'Homo sapiens'):
        df_species = self.reactome_df[self.reactome_df['Species'] == species]
        print(f"Number of rows of {species}: {len(df_species.index)}")
        self.reactome_df = df_species
        return self.reactome_df
        
    def subset_on_protein_in_ms_data(self):
        proteins_in_ms_data = self.ms_df['Proteins'].values
        self.reactome_df = self.reactome_df[self.df['Proteins'].isin(proteins_in_ms_data)]
        return self.df
        
    def subset_hierarchies_on_file(self):
        reactome_ids = self.reactome_df['Reactome_id'].values
        
        def keep_row(row):
            # Should both From and To contain the protein or just From
            protein1 =row['From'] 
            protein2 = row['To']
            if protein1 not in reactome_ids:
                return False
            if protein2 not in reactome_ids:
                return False
            return True
        
        subsetted['in_subset'] = self.path_df.apply(lambda x: keep_row(x), axis=1)
        subsetted = subsetted[subsetted['in_subset'] == True]
        subsetted.drop(['in_subset'], inplace=True)
        self.path_df = subsetted
 
    def save_df(self, df_id, save_path):
        if df_id == 'reactome':
            self.reactome_df.to_csv(save_path, index=False)
        if df_id == 'path':
            self.path_df.to_csv(save_path, index=False)
        
        
        
if __name__ == '__main__':
    print('--------------')
    print('|S C I E N C E|')
    print('--------------')
    print('\( o _ o )/')