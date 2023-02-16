import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import streamlit as st 
from PIL import Image #To display images 
import pickle # to load model
from functions import find_target #Function for finding target
from functions import plot_null_values #Function to check null values
from functions import replace_yes_no #Function to replace irregularities in data 
from functions import rep_null_val #Function to replace null values
from functions import drop_columns #Function to drop columns 
from functions import cleaning_pipeline #Function that includes cleaning replacing null values and dropping unnecessary columns
from functions import data_dictionary # variable dictionary 
from functions import problem_stat # problem statement string 
from functions import steps # Steps to be performed 
from functions import target_inference# target variable inference
from functions import check_all_member_same_target # as the name suggests it check how many families does not have same target for all variable
from functions import check_with_head_or_not # checks for how many families are without heads
from functions import check_no_head_same_target # check for those familie if there is a case where if a family is wothout a head then if the target variable is different for different family members
from functions import null_val_repl_basis # inference text for null values replacement
from functions import null_val_replace_logic # null val replecement log text 
################################################################################################################################
#load model 
model=pickle.load(open('model.pkl','rb'))
################################################################################################################################
#functions
# To Improve speed and cache data
@st.cache_data(persist=True) 

def Read_data(filename):
    """Function reads csv data and returns as a dataframe"""
    return pd.read_csv(filename)

################################################################################################################################
#reading data     
income_train_df=Read_data('train.csv')
income_test_df=Read_data('test.csv')
###############################################################################################################################
#Separating columns in different data types 
float_columns=income_train_df.select_dtypes('float').columns.tolist()
int_columns=income_train_df.select_dtypes('int').columns.tolist()
object_columns=income_train_df.select_dtypes('object').columns.tolist()
#separating columns which indicate home ownership
own_variables = [x for x in income_train_df if x.startswith('tipo')]
###############################################################################################################################
# Title and Subheader
def main ():# beginning of the statement execution
    html_temp = """
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Income Qualifications </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.subheader("EDA Web App")
    st.text("Team: Ashish Sarkar, Prajwal Rajput,Shraddha Mungal,Aryan Patle")
    st.text("Mentor: Shriti Dutta")
    if st.sidebar.checkbox('Problem Statement'):
        problem_stat
    if st.sidebar.checkbox('Show Data Dictionary'):
        data_dictionary
    if st.sidebar.checkbox('Steps To be performed'):
         steps 
    if st.sidebar.checkbox('Finding Target Variable and Checking for Bias'):
        Target=find_target(income_train_df,income_test_df)
        st.text(f"The {find_target(income_train_df,income_test_df)} is the Output Variable")
        st.subheader(f"{Target} variable cardinal distribution")
        st.write((income_train_df[Target].value_counts()/len(income_train_df))*100)
        st.subheader("Mappings")
        """  
             * 1 : Extreme Poverty
             * 2 : Moderate Poverty
             * 3 : Vulnerable Households
             * 4 : Non-vulnerable Households"""
        st.subheader("Inference")
        target_inference
    if st.sidebar.checkbox('Check Dataset information'):
        st.subheader("Dataset Sample")
        st.write(income_train_df.iloc[:50])
        st.subheader("Dataset Shape")
        st.text(f"The Training Dataset Consists of {income_train_df.shape[0]} rows and {income_train_df.shape[1]} columns")
        st.subheader("Columns Info")
        data_dim=st.radio("Column type information",("float_columns","int_columns","object_columns"))
        if data_dim=="float_columns":
            st.text(f'Out of 143 {len(float_columns)} are Float_type type columns')
            st.write(float_columns)
            st.subheader("Null Values Status")
            st.write(plot_null_values(income_train_df[float_columns]))
            st.subheader("Inference")
            st.text(
                """                 *   v2a1 6860 ==>71.77%  variable explaination => Monthly rent payment
                *   v18q1 7342 ==>76.82% variable explaination => Number of tablets household owns
                *   rez_esc 7928 ==>82.95% variable explaination => Years behind in school
                *   meaneduc 5 ==>0.05% variable explaination => average years of education for adults (18+)
                *   SQBmeaned 5 ==>0.05% variable explaination => square of the mean years of education of adults""")
        if data_dim=="object_columns":
            st.text(f'Out of 143 {len(object_columns)} are Categorical type columns')
            st.write(object_columns)
            st.subheader("Null Values Status")
            st.write(plot_null_values(income_train_df[object_columns]))
            st.subheader("Inference")
            st.text('There Are no null values in object columns')
        if data_dim=="int_columns":
            st.text(f'Out of 143 {len(int_columns)} are Integer type columns')
            st.write(int_columns)
            st.subheader("Null Values Status")
            st.write(plot_null_values(income_train_df[int_columns]))
            st.subheader("Inference")
            st.text('There Are no null values in integer type columns')
    if st.sidebar.checkbox("Check Family members information"):
        st.subheader("Check if all family members have the same poverty level or not")
        st.text(f'There are {check_all_member_same_target(income_train_df)} households where the family members do not have same poverty level') 
        st.subheader("Check if all families are with Heads")
        st.text(f'There are {check_with_head_or_not(income_train_df)} families without family Head')  
        st.subheader("Check if families without Heads have different poverty levels")
        st.text(f"There are {check_no_head_same_target(income_train_df)} families with no Heads and different poverty levels") 
        st.subheader("Inference")
        st.text("There are 85 Families which have different poverty levels but all of them have Family heads")   
    if st.sidebar.checkbox("Basis for Null Value replacement"):
        null_val_repl_basis
        st.text(f"column {float_columns[0]} has {income_train_df[float_columns[0]].isnull().sum()} null values")
        # Plot of the home ownership variables for home missing rent payments
        plt.rcParams["figure.figsize"] = (3,3)
        fig, x = plt.subplots()
        income_train_df.loc[income_train_df['v2a1'].isnull(), own_variables].sum().sort_values().plot(kind='bar')
        plt.xticks([0, 1, 2, 3, 4],[ 'Owns and Paying','Rented', 'Precarious',  'Other', 'Owns and Paid Off'],rotation = 90)
        plt.title('Home Ownership Status for Households Missing Rent', size = 5)
        #plotting figure
        st.pyplot(fig)
        st.text(f"colum {float_columns[1]} has {income_train_df[float_columns[1]].isnull().sum()} null values")
        heads = income_train_df.loc[income_train_df['parentesco1'] == 1].copy()
        heads.groupby('v18q')['v18q1'].apply(lambda x: x.isnull().sum())
        """Lets look at v18q1 (total nulls: 7342) : number of tablets household owns
        #why the null values, Lets look at few rows with nulls in v18q1
        #Columns related to number of tablets household owns
        #v18q, owns a tablet
        #Since this is a household variable, it only makes sense to look at it on a household level
        #so we'll only select the rows for the head of household."""
        fig, x = plt.subplots()
        income_train_df[float_columns[1]].value_counts().sort_index().plot(kind='bar')
        plt.xlabel(float_columns[0])
        plt.ylabel('Value_counts')
        #plotting figure
        st.pyplot(fig)
        st.text(f"colum {float_columns[2]} has {income_train_df[float_columns[2]].isnull().sum()} null values")
        """Years behind in school
        #why the null values, Lets look at few rows with nulls in rez_esc
        #Columns related to Years behind in school
        #Age in years
        # Lets look at the data with not null values first."""
        st.text("Now grouping rez_esc age to check the pattern")
        st.write(income_train_df[income_train_df['rez_esc'].notnull()]['age'].describe())
        """There is one value that has Null for the 'behind in school' column with age between 7 and 17"""
        st.write(income_train_df.loc[(income_train_df['rez_esc'].isnull() & 
                     ((income_train_df['age'] > 7) & (income_train_df['age'] < 17)))]['age'].describe())
        """There is only one member in household for the member with age 10 and who is 'behind in school'. This explains why the member is 
        behind in school."""
        st.write(income_train_df[(income_train_df['age'] ==10) & income_train_df['rez_esc'].isnull()].head())
        st.text(f"colum {float_columns[3]} has {income_train_df[float_columns[3]].isnull().sum()} null values")
        st.write(income_train_df[income_train_df['meaneduc'].isnull()].loc[:,['age','meaneduc','edjefe','edjefa','instlevel1','instlevel2','instlevel3','instlevel4','instlevel5','instlevel6','instlevel7','instlevel8','instlevel9']])
        st.text(f"colum {float_columns[7]} has {income_train_df[float_columns[7]].isnull().sum()} null values")
        st.write(income_train_df[income_train_df['SQBmeaned'].isnull()].loc[:, ['SQBmeaned','meaneduc','edjefe','edjefa','instlevel1','instlevel2']])
        st.text("We have also found that there are few columns where 'yes' should be mapped with 1 and 0 with 'no' as other values follow the same pattern")
        """'dependency' column"""
        st.write(income_train_df['dependency'].value_counts())
        st.subheader("Inference")
        """'edjefe' column"""
        st.write(income_train_df['edjefe'].value_counts())
        """'edjefa' column"""
        st.write(income_train_df['edjefa'].value_counts())
        """'meaneduc' column"""
        st.write(income_train_df['meaneduc'].value_counts())
        
        st.subheader("Inference")
        null_val_replace_logic
    if st.sidebar.checkbox("Feature Selection"):
        st.subheader("Columns Break Up")
        id_ = ['Id', 'idhogar', 'Target']
        ind_bool = ['v18q', 'dis', 'male', 'female', 'estadocivil1', 'estadocivil2', 'estadocivil3', 
            'estadocivil4', 'estadocivil5', 'estadocivil6', 'estadocivil7', 
            'parentesco1', 'parentesco2',  'parentesco3', 'parentesco4', 'parentesco5', 
            'parentesco6', 'parentesco7', 'parentesco8',  'parentesco9', 'parentesco10', 
            'parentesco11', 'parentesco12', 'instlevel1', 'instlevel2', 'instlevel3', 
            'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 
            'instlevel9', 'mobilephone']
        ind_ordered = ['rez_esc', 'escolari', 'age']
        hh_bool = ['hacdor', 'hacapo', 'v14a', 'refrig', 'paredblolad', 'paredzocalo', 
           'paredpreb','pisocemento', 'pareddes', 'paredmad',
           'paredzinc', 'paredfibras', 'paredother', 'pisomoscer', 'pisoother', 
           'pisonatur', 'pisonotiene', 'pisomadera',
           'techozinc', 'techoentrepiso', 'techocane', 'techootro', 'cielorazo', 
           'abastaguadentro', 'abastaguafuera', 'abastaguano',
            'public', 'planpri', 'noelec', 'coopele', 'sanitario1', 
           'sanitario2', 'sanitario3', 'sanitario5',   'sanitario6',
           'energcocinar1', 'energcocinar2', 'energcocinar3', 'energcocinar4', 
           'elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4', 
           'elimbasu5', 'elimbasu6', 'epared1', 'epared2', 'epared3',
           'etecho1', 'etecho2', 'etecho3', 'eviv1', 'eviv2', 'eviv3', 
           'tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5', 
           'computer', 'television', 'lugar1', 'lugar2', 'lugar3',
           'lugar4', 'lugar5', 'lugar6', 'area1', 'area2']
        hh_ordered = [ 'rooms', 'r4h1', 'r4h2', 'r4h3', 'r4m1','r4m2','r4m3', 'r4t1',  'r4t2', 
              'r4t3', 'v18q1', 'tamhog','tamviv','hhsize','hogar_nin',
              'hogar_adul','hogar_mayor','hogar_total',  'bedrooms', 'qmobilephone']
        hh_cont = ['v2a1', 'dependency', 'edjefe', 'edjefa', 'meaneduc', 'overcrowding']
        st.text(f'id columns==> {id_} \n with 90% unique value or is target variable')
        st.text(f'ind_bool columns==> {ind_bool} \n with 0|1 value for an individual')
        st.text(f'ind_ordered columns==> {ind_ordered} \n with ordered numerical value')
        st.text(f'hh_bool columns==> {hh_bool} \n with 0|1 value for house hold')
        st.text(f'hh_cont columns==> {hh_cont} \n with house hold continous values')
        st.subheader("Cheking for multicollinearity")
        st.text("Checking for redundant household variables select the ones with 'parentesco1' value as 1 only remove the ones which does not have a head of family")
        "Taking families with heads only"
        heads = income_train_df.loc[income_train_df['parentesco1'] == 1, :]
        heads = heads[id_ + hh_bool + hh_cont + hh_ordered]
        corr_matrix = heads.corr()
        "Correlation Matrix"
        st.write(corr_matrix)
        "Selecting the upper traingle of corr_matrix"
        #Selecting the upper traingle of corr_matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(bool))
        plt.rcParams["figure.figsize"] = (8,8)
        fig, x = plt.subplots()
        sns.heatmap(corr_matrix.loc[corr_matrix['tamhog'].abs() > 0.9, corr_matrix['tamhog'].abs() > 0.9],annot=True, cmap = sns.color_palette("Set2", 8), fmt='.3f')
        st.write(fig)
        cols_to_drop = [column for column in upper.columns if any(abs(upper[column]) > 0.95)]
        st.text(f'Columns to drop {cols_to_drop} \n as they have above 0.95 correlation')
        #st.write()
        """
#### Inference
There are several variables here having to do with the size of the house:
*   r4t3, Total persons in the household
*   tamhog, size of the household
*   tamviv, number of persons living in the household
*   hhsize, household size
*   hogar_total, # of total individuals in the household
*   These variables are all highly correlated with one another.
*   Removing the male as well, as this would not be needed in model creation
*   Removing the Id and 'idhogar' as well, as this would not be needed in model creation
##### There are some Squared Variables and we understand that these would not add any value to the classification model. 
##### Hence dropping these features - 
*   'SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe', 'SQBhogar_nin', 'SQBovercrowding', 'SQBdependency', 'SQBmeaned', 'agesq','Id','idhogar','coopele', 'area2', 'tamhog', 'hhsize', 'hogar_total', 'r4t3', 'area2', 'male'"""
    if st.sidebar.checkbox("Model Building"):
        st.text("Random Forrest Classifier was selected as per instructions")
        image = Image.open('Model 1 performance.jpg')
        image2= Image.open('feature imp.jpg')
        st.image(image,caption='Model1 performance')
        st.image(image2,caption='Feature Importance')
        """From above figure, meaneduc, dependency, overcrowding has significant influence on the model."""
        st.text("Post Cross-Validtion and taking 150 trees the \n accuracy scores average increased from \n 94.25547753700772 to 94.30783352929198")
    if st.sidebar.checkbox("Predictions"):
        """Upload csv file for feeding data to model"""
        data_upload=st.file_uploader("Upload File",type=[".csv"])
        dftest=pd.read_csv(data_upload)
        if data_upload is not None:
            st.write(dftest)
            dftest=cleaning_pipeline(dftest)
            predictions=model.predict(dftest)
            st.text(f"{predictions}")
    
    
    
    
if __name__=='__main__': #check for main executed when programme is called 
    main()