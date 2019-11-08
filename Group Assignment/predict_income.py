# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import pandas as pd
from math import sqrt, log, exp

# Function used for analyzing data
def get_unique_from_cols(dataset, columns):
    for column in columns:
        print(column)
        print(dataset[column].dtype)
        print(dataset[column].unique())
        print(len(dataset[column].unique()))

def parse_eur(data):
    column = data['Yearly Income in addition to Salary (e.g. Rental Income)']
    temp = [s.replace(' EUR' , '') for s in column]
    data['Yearly Income in addition to Salary (e.g. Rental Income)'] = temp
    return data



# This function was ultimately not used in the best public submission of the code
def replace_unacceptable_values(dataset,column_name,acceptable_values,replacement_value):
    for i in range(len(dataset[column_name])):
        if dataset[column_name][i] not in acceptable_values:
            dataset[column_name][i] = replacement_value
    return dataset


# This function was ultimately not used in the best public submission of the code
def replace_unacceptable_categorical(dataset):
    acceptable_genders = ['other','female', 'male', 'unknown']
    replacement_gender = 'other'
    dataset = replace_unacceptable_values(dataset,'Gender', acceptable_genders, replacement_gender)

    acceptable_degrees = ['Bachelor', 'Master', 'PhD', 'No']
    replacement_degree = 'unknown degree'
    dataset = replace_unacceptable_values(dataset,'University Degree', acceptable_degrees, replacement_degree)
    return dataset

# I looked at graph and noticed big gap between these
def process_city(dataset):
    dataset['Small City'] = dataset['Size of City'] <= 3000
    return dataset


# Prev removed outliers but turned out to produce a worse result
def training_data_preprocessing(dataset):
    #dataset = dataset[dataset['Income in EUR'] < 2500000]
    dataset['Total Yearly Income [EUR]'] = dataset['Total Yearly Income [EUR]'].apply(np.log)
    dataset = dataset.dropna(subset=['Size of City','Age','Year of Record'])
    return dataset


# Importing the dataset
training_dataset = pd.read_csv(
    'tcd-ml-1920-group-income-train.csv')
test_dataset = pd.read_csv(
    'tcd-ml-1920-group-income-test.csv')

# Removing Instance Col
training_dataset = training_dataset.loc[:, [
    'Year of Record','Housing Situation','Crime Level in the City of Employement','Work Experience in Current Job [years]','Satisfation with employer','Gender','Age','Country','Size of City','Profession','University Degree','Wears Glasses','Hair Color','Body Height [cm]','Yearly Income in addition to Salary (e.g. Rental Income)','Total Yearly Income [EUR]', ]]
test_dataset = test_dataset.loc[:, ['Year of Record','Housing Situation','Crime Level in the City of Employement','Work Experience in Current Job [years]','Satisfation with employer','Gender','Age','Country','Size of City','Profession','University Degree','Wears Glasses','Hair Color','Body Height [cm]','Yearly Income in addition to Salary (e.g. Rental Income)','Total Yearly Income [EUR]', ]]

# I kept this code here commented as ultimately removed it but felt it was
# smart to replace these values.

# ***    BELOW COMMENTED SECTION IS NOW REDUNDANT KEPT FOR REFERECNE     ***
# From analyzing data:
# - Gender has 6 unique options ['0' 'other' 'female' 'male' nan 'unknown']
# - University Degree has 6 unique options ['Bachelor' 'Master' 'PhD' 'No' '0' nan]
# - Hair Color has 7 unique options ['Blond' 'Black' 'Brown' nan 'Red' 'Unknown' '0']
# - This is present in both training and test datasets

# *Current* Strategy: 
# - Gender - keep ['other' 'female' 'male' 'unknown'], mark '0' and nan as 'unknown'
# - University Degree - keep ['Bachelor' 'Master' 'PhD' 'No'] mark '0' and nan as 'No'
# - Hair Color - keep ['Blond' 'Black' 'Brown' 'Red' 'Unknown' ] mark '0' and nan as 'Unknown'

# acceptable_professions = list(set(training_dataset['Profession']).intersection(set(test_dataset['Profession'])))
# replacement_profession = 'other profession'
# training_dataset = replace_unacceptable_values(training_dataset,'Profession', acceptable_professions, replacement_profession)
# test_dataset = replace_unacceptable_values(test_dataset,'Profession', acceptable_professions, replacement_profession)

# training_dataset = replace_unacceptable_categorical(training_dataset)
# test_dataset = replace_unacceptable_categorical(test_dataset)

# acceptable_countries = list(set(training_dataset['Country']))
# replacement_country = 'other country'
# test_dataset = replace_unacceptable_values(test_dataset,'Country', acceptable_countries, replacement_country)

# ******************************************************************************

training_dataset=parse_eur(training_dataset)
training_dataset['Yearly Income in addition to Salary (e.g. Rental Income)'] = pd.to_numeric(training_dataset['Yearly Income in addition to Salary (e.g. Rental Income)'], errors='coerce')
training_dataset['Work Experience in Current Job [years]'] = pd.to_numeric(training_dataset['Work Experience in Current Job [years]'], errors='coerce')
training_dataset = training_dataset.replace(np.nan, 0, regex=True)


get_unique_from_cols(training_dataset, [
    'Year of Record','Housing Situation','Crime Level in the City of Employement','Work Experience in Current Job [years]','Satisfation with employer','Gender','Age','Country','Size of City','Profession','University Degree','Wears Glasses','Hair Color','Body Height [cm]','Yearly Income in addition to Salary (e.g. Rental Income)' ])

training_dataset= training_dataset[:201709]






training_dataset = process_city(training_dataset)
test_dataset = process_city(test_dataset)
training_dataset = training_data_preprocessing(training_dataset)


# This falls for dummy variable trap but somehow produces a better result,
# My logic here is that as I drop cols from training and test anyways it assumes
# the dummy variable removal that is why I don't need to remove dummy variable
# here.
training_dataset = pd.get_dummies(training_dataset, drop_first=False)
training_dataset.fillna(training_dataset.mean(), axis=0,inplace=True)




test_dataset =test_dataset[:100000]

test_dataset=parse_eur(test_dataset)
test_dataset['Yearly Income in addition to Salary (e.g. Rental Income)'] = pd.to_numeric(test_dataset['Yearly Income in addition to Salary (e.g. Rental Income)'], errors='coerce')
test_dataset['Work Experience in Current Job [years]'] = pd.to_numeric(test_dataset['Work Experience in Current Job [years]'], errors='coerce')
test_dataset = test_dataset.replace(np.nan, 0, regex=True)

test_dataset = pd.get_dummies(test_dataset,drop_first=False)
test_dataset.fillna(test_dataset.mean(), axis=0,inplace=True)

# Changing income col name to match training
test_dataset = test_dataset.rename(columns={"Income": "Total Yearly Income [EUR]"})
common_cols = [col for col in set(training_dataset.columns).intersection(test_dataset.columns)]
training_dataset = training_dataset[common_cols]
test_dataset = test_dataset[common_cols]

y = training_dataset['Total Yearly Income [EUR]']

# **** Tried Backward Elimination here - Ultimately overfit, so removed ********

# This disgusting drop, is a list of all professions which have previously been
# removed via backward elimination, I didn't let backward elimination run fully
# as was worried it would just overfit the data...which it could be doing already

# Nevermind totally overfits so commented out this line of code xD
# training_dataset = training_dataset.drop(['Profession_systems access management engineer','Profession_Bar Steward','Profession_staff analyst level ii','Profession_travel guide','Profession_tender','Profession_agricultural engineer','Profession_technical solutions professional','Profession_timekeeping specialist','Profession_Auction Worker','Profession_Biologist','Profession_Biochemist','Profession_Bank Manager','Profession_Amusement Arcade Worker','Profession_Advertising Staff','Profession_analyst manager','Profession_application support reporting specialist','Profession_temporary painter','Profession_summer college intern','Profession_Broadcaster','Profession_Arts','Profession_Blacksmith','Profession_teller','Profession_student data analyst','Profession_thermostat repairer','Profession_Arborist','Profession_Beautician','Profession_waitress','Profession_Administrator','Profession_stationary engineer','Profession_statistical analyst','Profession_unit manager','Profession_unit clerk','Small City','Profession_testifier/searcher','Profession_Bar Manager','Profession_administrative aide','Profession_bookbinder','Profession_Art Historian','Profession_translator','Profession_technical sales manager','Profession_Baggage Handler','Profession_webform team member','Profession_usher','Profession_wireless coordinator','Profession_Bailiff','Profession_Art Critic','Profession_assistant civil engineer','Profession_chemical technician','Country_Belize','Profession_systems administrator - computer software','Profession_Aerial Erector','Profession_Accounts Clerk','Profession_Animal Breeder','Profession_Artexer','Profession_Bookmaker','Profession_Balloonist','Profession_communication specialist','Profession_web application developer','Profession_tool grinder','Profession_agency counsel','Profession_Actor','Profession_special consultant','Profession_Actuary','Profession_special assistant','Profession_strategic operations policy analyst','Profession_structural iron and steel worker','Profession_valve installer','Profession_Beauty Therapist','Profession_special projects manager','Profession_sociologist','Profession_telephone operator','Profession_tax examiner','Profession_community assistant','Profession_stonemason','Profession_account executive ','Profession_structural engineer','Profession_assistant business services associate','Profession_special assistant to assistant deputy commissioner','Profession_Builder','Profession_claims adjuster','Profession_staff auditor','Profession_therapist','Profession_cafeteria attendant','Profession_team coach team leader','Profession_architectural designer','Country_Uganda','Profession_Aeronautical Engineer','Profession_business manager','Profession_training specialists','Profession_vp for project management','Profession_surgeon','Profession_Ambulance Crew','Country_Cabo Verde','Country_Ukraine','Profession_Art Restorer','Profession_support worker','Country_Micronesia','Profession_calendar assistant','Profession_accountant','Profession_Barber','Profession_assistant commissioner of administration','Profession_storage engineer','Profession_summer graduate intern','Profession_assistant commissioner of enforcement','Profession_audit engineer','Profession_Airman','Profession_software developer','Profession_Brewer','Profession_branch chief','Profession_administrative office assistant','Profession_watershed maintainer','Profession_Aircraft Maintenance Engineer','Profession_administrative management auditor','Profession_staff attorney','Profession_virtual systems engineer','Profession_Bodyshop','Profession_staff analyst','Profession_air & noise pollution inspector','Profession_Ambulance Driver','Profession_actor','Profession_Blind Fitter','Profession_testing lead','Profession_special underwriting project manager','Profession_sr. analyst','Profession_tailor','Profession_social science research assistant','Country_Tanzania','Profession_butcher','Profession_capacity building assistance specialist','Profession_upholsterer','Profession_administrative staff analyst','Profession_Audit Clerk','Country_Brunei','Country_Vanuatu','Profession_unix system administrator','Profession_urban technology architect','Profession_case analyst','Profession_yardmaster','Profession_Acupuncturist','Profession_workforce planning intern','Profession_Barmaid','Country_Bahamas','Country_Kenya','Profession_Blinds Installer','Profession_administrative associate to the executive director','Profession_Audit Manager','Profession_city assessor','Profession_system administrator','Profession_Acoustic Engineer','Profession_Accounts Assistant','Profession_audiologist','Profession_Building Advisor','Profession_strategic partnership liaison','Profession_threat analysttimekeeper','Profession_social service specialist','Country_Myanmar','Profession_street ambassador','Profession_stock clerk','Profession_supervising housing groundskeeper','Profession_stucco mason','Profession_business intelligence developer','Profession_supervising public health advisor','Profession_word processor ','Profession_trainer','Profession_back end developer','Profession_senior windows administrator','Profession_chief engineer of dispute resolutions','Profession_agricultural scientist','Profession_Area Manager','Profession_Auctioneer','Profession_administrative assistant','Profession_Botanist','Profession_staff analyst ii','Profession_tile installer','Profession_senior trainer','Profession_senior title examiner','Profession_bailiff','Profession_architect','Profession_Book Binder','Profession_standards specialist','Profession_strategic account manager','Profession_welder','Profession_strategic initiatives coordinator','Profession_administrative coordinator','Profession_Brewery Worker','Profession_Aerobic Instructor','Profession_veterinarian','Profession_urban and regional planner','Profession_animalbreeder','Profession_Airport Controller','Profession_summer it intern','Profession_Assistant Teacher','Profession_senior project manager','Profession_Astronomer','Profession_accessibility program manager','Country_South Korea','Country_Spain','Profession_assistant commissioner','Profession_technical investigator','Profession_service asset configuration manager','Profession_summer communications assistant','Profession_senior urban designer','Profession_Bank Clerk','Profession_Book Seller','Profession_actuary','Profession_Anaesthetist','Profession_Auditor','Profession_technical project manager/product owner','Profession_trial preparation assistant','Profession_woodworker','Profession_vision screening assistant','Country_Colombia','Profession_Building Control','Profession_triage supervisor','Profession_appraiser','Profession_Analytical Chemist','Profession_chief diversity officer','Profession_case management nurse','Country_South Africa','Country_United Kingdom','Profession_Arbitrator','Profession_special education teacher','Profession_caster','Profession_trainer & curriculum development specialist','Profession_unix/linux systems lead','Profession_.net software developer','Profession_strategic planning associate','Profession_senior research analyst','Profession_supervising health nurse','Profession_senior rackets investigator','Profession_speech-language pathologist','Profession_social worker','Profession_space analyst','Profession_sr. internal auditor','Profession_technical writer','Profession_sharepoint developer','Profession_Auxiliary Nurse','Profession_ship engineer','Profession_supervising nurse','Profession_tort attorney','Profession_truck driver','Profession_supervising physician','Profession_water resources analyst','Profession_statistician','Profession_space scientists','Profession_staff counsel','Profession_staff analyst 2','Profession_Baptist Minister','Profession_surveyor','Profession_sql/oracle database administrator','Profession_steel workers','Profession_Anthropologist','Profession_Bacon Curer','Profession_tour guide','Profession_Occupations','Profession_aide','Profession_triage nurse','Profession_Baker','Profession_bridge operator','Profession_auditor','Profession_assistant commissioner of the office of placement administration','Profession_compliance manager','Profession_sergeant','Profession_archivist','Profession_advertising and promotions manager','Profession_Aromatherapist','Profession_Builders Labourer','Profession_Analyst','Country_Maldives','Country_Suriname','Country_Poland','Country_Sudan','Profession_sheet metal worker','Profession_computer support specialist','Profession_Ambulance Controller','Profession_Almoner','Profession_Bill Poster','Profession_senior stationary engineer','Profession_community coordinator','Profession_Brewery Manager','Profession_application solution manager','Country_Iraq','Profession_communications equipment operator','Profession_senior mobile developer','Profession_service desk agent','Profession_statistical assistant','Profession_advertising sales agent','Profession_atmospheric scientist','Profession_Administration Assistant','Profession_service desk manager','Profession_taper','Profession_senior product manager','Profession_water use inspector','Profession_timekeeper','Profession_bid operations liaison','Country_Luxembourg','Profession_business solution architect','Profession_Book-Keeper','Profession_staff photographer','Profession_senior financial reporting investment analyst','Profession_taxi driver','Profession_Barrister','Profession_umpire','Profession_Branch Manager','Profession_vessel construction manager','Profession_senior safety accident investigator','Profession_senior service desk agent','Profession_social scientist','Profession_server support engineer','Profession_ticket taker'],axis=1)
training_dataset = training_dataset.drop("Total Yearly Income [EUR]",1)
training_dataset = training_dataset.drop("Size of City",1)
cols = list(training_dataset.columns)

# This auto backward elimination code was adapted from:
# https://gist.github.com/vb100/177bad75b7506f93fbe12323353683a0
# It seems standard enough however still wanted to credit
# pmax = 1
# while (len(cols)>0):
#     p= []
#     X_1 = training_dataset[cols]
#     X_1 = sm.add_constant(X_1)
#     model = sm.OLS(y,X_1).fit()
#     p = pd.Series(model.pvalues.values[1:],index = cols)      
#     pmax = max(p)
#     feature_with_p_max = p.idxmax()
#     if(pmax>0.05):
#         print(feature_with_p_max)
#         cols.remove(feature_with_p_max)
#     else:
#         break
selected_features_BE = cols
X = training_dataset.loc[:, selected_features_BE].values
Y = y.values
X_test_real = test_dataset.loc[:, selected_features_BE].values

## I did have cross val previously but took too long to run so just opted 80:20
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Running Model
regression = LinearRegression()
regression.fit(X_train,Y_train)

# Local Test
pred = regression.predict(X_test)
print("Root Mean squared Error")
print((sqrt(mean_squared_error(np.exp(Y_test), np.exp(pred)))))

# # Kaggle Test
# pred_test = regression.predict(X_test_real)
# print(pred_test)
# pred_test = np.exp(pred_test)
# output = pd.read_csv('tcd ml 2019-20 income prediction submission file.csv')
# for i in range(len(pred_test)):
#     output['Income'][i] = pred_test[i]
# print(output)

# output.to_csv('tcd ml 2019-20 income prediction submission file.csv',index=False, encoding='utf8')
