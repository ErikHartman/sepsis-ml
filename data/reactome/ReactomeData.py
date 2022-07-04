import pandas as pd

class ReactomeData():
    """ A class to subset and get reactome data.
    
    file1 = 'data/reactome/ReactomePathwaysRelation.txt'
    file2 = 'data/ms/proteins.csv' 
    file3 = "data/reactome/UniProt2Reactome_All_Levels.txt"
    RD = ReactomeData(file1, file2, file3)
    RD.subset_species_reactome_data('Homo sapiens')
    RD.subset_on_proteins_in_ms_data()
    RD.subset_pathways_on_reactome_idx()
    RD.save_df("reactome", "HSA_ms_reactome.csv") 
    RD.save_df("path", "HSA_ms_path.csv")
    
    The two saved files are the reactome and path files for the reactome which can be used to
    create the network. 
    """
    
    def __init__(self,hierarchy_file_path, ms_data_file_path, reactome_all_data_file_path):
        self.hierarchy_file_path = hierarchy_file_path
        self.ms_data_file_path = ms_data_file_path
        self.reactome_all_data_file_path = reactome_all_data_file_path    
        self.ms_df = pd.read_csv(ms_data_file_path)
        self.reactome_df = pd.read_csv(self.reactome_all_data_file_path, index_col=False,
                        names = ['UniProt_id', 'Reactome_id', 'URL', 'Description','Evidence Code','Species'], sep="\t")
        self.path_df = pd.read_csv(hierarchy_file_path, names=['parent','child'], sep="\t", index_col=False)
        
    def subset_species_reactome_data(self, species = 'Homo sapiens'):
        df_species = self.reactome_df[self.reactome_df['Species'] == species]
        print(f"Number of rows of {species}: {len(df_species.index)}")

        self.reactome_df = df_species
        return self.reactome_df
        
    def subset_on_proteins_in_ms_data(self):
        proteins_in_ms_data = self.ms_df['Proteins'].values
        print(f'Number of reactome ids before subsetting: {len(self.reactome_df.index)}')
        self.reactome_df = self.reactome_df[self.reactome_df['UniProt_id'].isin(proteins_in_ms_data)]
        print(f"Number of reactome ids after subsettign: {len(self.reactome_df.index)}")
        print(f"Unique proteins in reactome df: {len(list(self.reactome_df['UniProt_id'].unique()))}")
        return self.reactome_df
        
    def subset_pathways_on_reactome_idx(self):
        """
        We want this function to use the reactome idx to subset the pathway-file.
        The first step is just to find in which "parent" the reactom idx exist.
        Thereafter we want to find which "To" exist in "parent" and iterate this step.
        This sounds like recursion to me!
        
        Want a function which takes list and "parent" as input and adds the parent to list.
        Thereafter the function will be called with the list and "Child". 
        
        Base case: Child is empty. Return List.
        """
        def add_pathways(counter, reactome_idx_list, parent):
            counter += 1
            print(f"Function called {counter} times.")
            print(f'Values in reactome_idx_list: {len(reactome_idx_list)}')
            if len(parent) == 0:
                print('Base case reached')
                return reactome_idx_list
            else:
                reactome_idx_list = reactome_idx_list + parent
                subsetted_pathway = self.path_df[self.path_df['parent'].isin(parent)]
                new_parent = list(subsetted_pathway['child'].unique())
                print(f"Values in new_parent: {len(new_parent)}")
                return add_pathways(counter, reactome_idx_list, new_parent)
                
        counter = 0    
        original_parent = list(self.reactome_df['Reactome_id'].unique())
        reactome_idx_list = []
        reactome_idx_list = add_pathways(counter, reactome_idx_list, original_parent)
        print(f"Final number of values in reactome list: {len(reactome_idx_list)}")
        # the reactome_idx_list now contains all reactome idx which we want to subset the path_df on.
        self.path_df = self.path_df[self.path_df['parent'].isin(reactome_idx_list)]
        print("Final number of unique connections in pathway: ", len(self.path_df.index))
        return self.path_df
 
 
 
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
    
    
    file1 = 'data/reactome/ReactomePathwaysRelation.txt'
    file2 = 'data/ms/proteins.csv' 
    file3 = "data/reactome/UniProt2Reactome_All_Levels.txt"
    RD = ReactomeData(file1, file2, file3)
    RD.subset_species_reactome_data('Homo sapiens')
    RD.subset_on_proteins_in_ms_data()
    RD.subset_pathways_on_reactome_idx()
    RD.save_df("reactome", "data/reactome/HSA_ms_reactome.csv") 
    RD.save_df("path", "data/reactome/HSA_ms_path.csv")
    
    
    
    
    
    