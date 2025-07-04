import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns


#### CONSTANTS ####
ROOT_FILE = r"world_bank_growth_potential_index\world_bank_data_2025.csv"
ROOT_FOLDER = r"world_bank_growth_potential_index"
WEIGHTS = {
    "GDP Growth (% Annual)": 0.15, 
    "GDP per Capita (Current USD)": 0.25, # lower weight because it is in USD and hence not relative
    "Unemployment Rate (%)": -0.25,    # I consider a negative trend in unemployment to be worse than inflation
    "Inflation (CPI %)": -0.20,         
    "Current Account Balance (% GDP)": 0.15 # potential for future development
}
REQUIRED_COLUMNS = [
    "country_name",
    "country_id",
    "year",
    "GDP Growth (% Annual)",
    "GDP per Capita (Current USD)",
    "Unemployment Rate (%)",
    "Inflation (CPI %)",
    "Current Account Balance (% GDP)"
]
ECONOMIC_INDICATORS = [
    "GDP Growth (% Annual)",
    "GDP per Capita (Current USD)",
    "Unemployment Rate (%)",
    "Inflation (CPI %)",
    "Current Account Balance (% GDP)"
]


#### FUNCTIONS USED ####


def extract_raw_data():
    """ Reads raw data from WORLD_BANK_DATA_2025.CSV and returns dataframe for analysis """
    return pd.read_csv(ROOT_FILE)


def fill_na_with_median(group):
    """ Fills null values in dataframe with median by column by country """
    return group.fillna(group.median(numeric_only = True))


def get_yearly_average_dataframe(df):
    """ 
    Returns a new data frame with averages for each indicator by year 
    @input_parameters  - clean_df
    """
    res_df = df.groupby('year')[ECONOMIC_INDICATORS].mean().reset_index()
    return res_df


def clean_data(original_df):
    """ 
    Returns a new data frame after cleaning 
     - Removes rows with null country names/ids
     - Filters only REQUIRED_COLUMNS
     - Saves cleaned dataframe to a csv file for future reference
    @input_parameters  - raw_df or original_df
    """
    original_df = original_df[original_df['country_name'].notna() & original_df['country_id'].notna()]  # remove rows with missing country names/ids
    original_df = original_df[REQUIRED_COLUMNS]                                                         
    original_df = original_df.groupby("country_name", group_keys=False).apply(fill_na_with_median).reset_index(drop=True)  # replacing null values with median by country    
    clean_df = original_df[original_df.isna().sum(axis = 1) <= 0]   # removing any leftover rows with null values

    clean_df.to_csv(ROOT_FOLDER + r"\world_bank_cleaned.csv", index=False) 
    return clean_df


def get_2025_values(df):
    """ 
    Returns a new data frame with data for ECONOMIC_INDICATORS by country for the year 2025
    @input_parameters  - clean_df
    """
    return df[df["year"] == 2025][["country_name"] + ECONOMIC_INDICATORS].reset_index(drop=True)


def predict_2030_values(df_2025,slopes_df):
    """ 
    Returns a new data frame with predicted data for ECONOMIC_INDICATORS for the years 2025 to 2030
    - saves dataframe with predictions to a .csv file for future reference
    @input_parameters  - clean_df and slopes_with_gdp_df
    """
    my_df = df_2025.copy()
    my_df[ECONOMIC_INDICATORS] = my_df[ECONOMIC_INDICATORS] + (5 * slopes_df[ECONOMIC_INDICATORS])
    my_df.to_csv(ROOT_FOLDER + r"\world_bank_2030_prediction.csv", index=False)
    return my_df


def get_slope_with_gdp_growth(df, indicators, time_col, group_col):
    """ 
    Returns dataframe with slopes for all indicators including GDP growth % 
    This function is primariy for plotting trendline of indicator values across years
    @input_parameters  - clean_df
    @returned_variables :
    - slopes_with_gdp_df - dataframe with slopes of all indicators by country
    """
    slopes = []    
    for country, group in df.groupby(group_col):
        country_slopes = {group_col: country}
        x = group[time_col].values
        for indicator in indicators:
            y = group[indicator].values
            if len(y) > 1:
                slope, intercept = np.polyfit(x, y, 1)
            else:
                slope = np.nan  # Not enough points to fit
            country_slopes[indicator] = slope
        slopes.append(country_slopes)    
    slopes_df = pd.DataFrame(slopes)
    slopes_df.to_csv(ROOT_FOLDER + r"\world_bank_slopes.csv", index=False)
    return slopes_df


def compute_slopes_without_gdp_growth(df, indicators, time_col, group_col):
    """ 
    DESCRIPTION : 
    - Returns dataframe with slopes for all indicators except  GDP growth % (since growth % is already a rate of change)
    - This function is used during calculation of index metrics
    @input_parameters  - clean_df
    @returned_variables : 
    - slopes_without_gdp_df - dataframe with slopes of all indicators by country
    """
    slopes = []    
    for country, group in df.groupby(group_col):
        country_slopes = {group_col: country}
        x = group[time_col].values
        for indicator in indicators:
            if indicator == "GDP Growth (% Annual)":
                country_slopes[indicator] = group[indicator].mean()
                continue    # gdp growth is already a slope
            y = group[indicator].values
            if len(y) > 1:
                slope, intercept = np.polyfit(x, y, 1)
            else:
                slope = np.nan  # Not enough points to fit
            country_slopes[indicator] = slope
        slopes.append(country_slopes)    
    slopes_df = pd.DataFrame(slopes)
    return slopes_df


def normalize_data(df):
    """ 
    DESCRIPTION : 
    - Min-max normalization for cleaned data
    - Clips outlier data (0.015 , 0.985)
    @input_parameters - clean_df
    @returned variables - normalized_df
    """
    normalized_df = df.copy()
    for col in normalized_df.columns:
        if col == 'country_name':
            continue
        valueList = normalized_df[col].values
        minValue = np.min(valueList)
        maxValue = np.max(valueList)
        numerator= (valueList - minValue)
        denominator =(maxValue - minValue)
        if denominator == 0:
            normalized_df[col] = 0*valueList
        else:
            normalized_df[col] = numerator/denominator
    #removing outliers 
    normalized_df[ECONOMIC_INDICATORS] = normalized_df[ECONOMIC_INDICATORS].clip(0.015,0.985)
    normalized_df.to_csv(ROOT_FOLDER + r"\world_bank_normalized.csv", index=False)
    return normalized_df


def apply_weights_and_get_scores(df1_normalized,df1_real):
    """
    DESCRIPTION : 
    - Multiplies ECONOMIC_INDICATORS in normalized_df by a list of WEIGHTS to get weighted values of indicators
    - A new dataframe is returned with a 'score' column added containing the raw index scores for each country
    @input_parameters : 
    - normalized_2030_df - normalized predicted values for each economic indicator
    - predicted_2030_df - raw predicted values for each economic indicator
    """
    weighted_df = df1_normalized.copy()

    for indicator in ECONOMIC_INDICATORS:
        weighted_df[indicator] *= WEIGHTS[indicator]

    # now sum up economic indicators to create a dataframe of scores by country
    final_df = df1_real.copy()
    final_df['score'] = weighted_df[ECONOMIC_INDICATORS].sum(axis=1)
    return weighted_df,final_df


def get_ranked_index(score_frame):
    """ Returns a new dataframe with a 'rank' columns and the countries arranged by GPI scores """
    ranked_df =  score_frame.sort_values('score', ascending=False).reset_index(drop=True)
    ranked_df.index = np.arange(1, len(ranked_df)+1, 1)
    ranked_df.index.name = 'rank'
    return ranked_df

def get_normalized_scores(df):
    """
    DESCRIPTION: 
    This function takes the raw scores and normalizes them from values 0 to 100
    - Min-max normalization
    - Saves data to .csv file for future reference
    @input_parameters:
    - raw_scores_df
    @returned_variable:
    - normalized_scores_df - GPI scores normalized( 0 to 100)
    """
    res_df = df.copy()
    valueList = res_df['score'].values
    minValue = np.min(valueList)
    maxValue = np.max(valueList)
    numerator = valueList - minValue
    denominator = maxValue - minValue

    if denominator == 0:
        res_df['normalized_score'] = 0 * valueList 
    else:
        res_df['normalized_score'] = (numerator / denominator) * 100
    # Save to CSV (with both raw and normalized scores)
    res_df.to_csv(ROOT_FOLDER + r"\growth_potential_index.csv", index=True)    
    return res_df


def get_predicted_average_trends(df_2025, slopes_df, start_year, end_year):
    """
    This function ITERATIVELY PREDICTS TRENDS between 2025 and 2030 by adding slopes to the previous values
    @input_parameters:
    - df_2025 - values of ECONOMIC_INDICATORS for all countriesthe year 2025 
    - slopes_df - dataframe containing slopes of all indicators including GDP growth
    - start_year - 2025
    - end_year - 2030
    @returned_variables:
    - trend_df - yearly average indicator values for years 2025 - 2030
    """
    current_row = df_2025[df_2025['year'] == 2025].iloc[0].copy()
    avg_slopes = slopes_df[ECONOMIC_INDICATORS].mean()    
    rowList = []
    for year in range(start_year, end_year + 1):
        row = current_row.copy()
        row['year'] = year
        rowList.append(row)
        current_row = current_row + avg_slopes  
    
    trend_df = pd.DataFrame(rowList)
    trend_df = trend_df[['year'] + ECONOMIC_INDICATORS]    
    trend_df.reset_index(drop=True, inplace=True) 
    return trend_df


def plot_combined_indicator_cor_heatmap(combined_weighted_df,ECONOMIC_INDICATORS):
    """ plots indicator correlation heatmap """
    plt.figure(figsize=(12,8))
    sns.heatmap(combined_weighted_df[ECONOMIC_INDICATORS].corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Between Economic Indicators (Based on predicted values)')
    plt.tight_layout()
    plt.savefig(ROOT_FOLDER + r"\indicator_combined_correlation_heatmap.png")
    plt.show()


def plot_bar_chart(ranked_df, size):
    """ plots bar chart """
    ranked_df = ranked_df[ranked_df.index <= size]
    ranked_df = ranked_df.sort_values('score', ascending=False)    
    plt.figure(figsize=(12,8))
    sns.barplot(data=ranked_df, x='country_name', y='normalized_score',hue = 'country_name', palette="rocket")
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Top {size} Countries by Growth Potential Index')
    plt.tight_layout()
    plt.savefig(ROOT_FOLDER + r"\top_{}_countries_bar_chart.png".format(size))
    plt.show()


def plot_box_chart(df,title,file_name):
    """ 
    This function plots a series of 5 box charts(subplots) stacked vertically on top of each other
    - one subplot for each ECONOMIC_INDICATOR 
    """
    sns.set_theme(style="dark")  # or use sns.set_style("darkgrid")
    palette = sns.color_palette("Set2", len(ECONOMIC_INDICATORS))
    fig, axes = plt.subplots(
        nrows=len(ECONOMIC_INDICATORS),
        ncols=1,
        figsize=(10, 4 * len(ECONOMIC_INDICATORS)),
        facecolor="#F5F5F5"
    )
    fig.suptitle(title, fontsize=16, color='black')
    for i, (indicator, ax) in enumerate(zip(ECONOMIC_INDICATORS, axes)):
        sns.boxplot(data=df, x=indicator, color=palette[i], ax=ax)
        ax.set_title(f'Distribution of {indicator}')
        ax.set_xlabel(indicator)
        ax.set_ylabel('')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97]) 
    plt.savefig(ROOT_FOLDER + "/" + file_name, facecolor="#F5F5F5")
    plt.close()


def plot_global_indicator_trendline(actual_df, predicted_df):
    """
    DESCRIPTION:
    - This function plots the global trends of all 5 ECONOMIC_INDICATORS in the timeframe 2010 - 2030 using a LINE CHART
    - Years 2010 - 2025 are documented values represented by solid lines in a line graph
    - Years 2025 - 2030 are predictions, this timeframe is marked by green shading on the graph and trends are represented with dotted lines
    @input_parameters:
    - actual_df - Dataframe with values of indicators from 2010 to 2025 (yearly_average_df)
    - predicted_df - Dataframe with values of indicators from 2025 to 2030 (predicted_yearly_df)
    """
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 8))
    
    plt.plot(np.arange(2010,2026,1), actual_df["GDP Growth (% Annual)"], color = '#FA0404', label=f"GDP Growth (% Annual) (Actual)")
    plt.plot(np.arange(2025,2031,1), predicted_df["GDP Growth (% Annual)"], color = '#FA0404', linestyle="--", label=f"GDP Growth (% Annual) (Predicted)")

    plt.plot(np.arange(2010,2026,1), 0.0001 * actual_df["GDP per Capita (Current USD)"],color = "#FA8704", label=f"GDP per Capita (in 10000 USDs) (Actual)")
    plt.plot(np.arange(2025,2031,1), 0.0001 * predicted_df["GDP per Capita (Current USD)"],color = "#FA8704",linestyle="--", label=f"GDP per Capita (in 10000 USDs) (Predicted)")

    plt.plot(np.arange(2010,2026,1), actual_df["Unemployment Rate (%)"],color = "#86E428F8", label=f"Unemployment Rate (%) (Actual)")
    plt.plot(np.arange(2025,2031,1), predicted_df["Unemployment Rate (%)"],color = "#86E428F8", linestyle="--", label=f"Unemployment Rate (%) (Predicted)")

    plt.plot(np.arange(2010,2026,1), actual_df["Inflation (CPI %)"],color = "#07AAE6F8", label=f"Inflation (CPI %) (Actual)")
    plt.plot(np.arange(2025,2031,1), predicted_df["Inflation (CPI %)"],color = "#07AAE6F8", linestyle="--", label=f"Inflation (CPI %) (Predicted)")

    plt.plot(np.arange(2010,2026,1), actual_df["Current Account Balance (% GDP)"],color = "#E607D0F8", label=f"Current Account Balance (% GDP) (Actual)")
    plt.plot(np.arange(2025,2031,1), predicted_df["Current Account Balance (% GDP)"],color = "#E607D0F8", linestyle="--", label=f"Current Account Balance (% GDP) (Predicted)")
    
    #shading predicted region in green
    plt.axvspan(2025, 2030, color='lightgreen', alpha=0.2, label='Forecasted')
    
    plt.title("Global Economic Trends (2010–2030)", fontsize=16)
    plt.xlabel("Year")
    plt.ylabel("Indicator Value")
    plt.legend(title="Indicator", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()    
    plt.savefig(ROOT_FOLDER + r"\indicator_trends_across_years.png")
    # ensuring that years remainwhole numbers
    plt.show()


def plot_global_indicator_trendline_wrapper(clean_df,slopes_with_gdp_df):
    """ 
    Wrapper function for plotting global indicators trendline
    @input_parameters:
    - clean_df 
    - slopes_with_gdp_df  
    """
    yearly_average_df = get_yearly_average_dataframe(clean_df)
    predicted_yearly_df = get_predicted_average_trends(yearly_average_df, slopes_with_gdp_df,2025,2030)
    plot_global_indicator_trendline(yearly_average_df,predicted_yearly_df)



def plot_data_wrapper(clean_df,clean_2025_df,predicted_2030_df,weighted_df,slopes_with_gdp_df,normalized_ranked_df):
    """ wrapper function for making all plots """
    # correlation heatmap
    plot_combined_indicator_cor_heatmap(weighted_df, ECONOMIC_INDICATORS)  
    # top 10 countries by GPI score chart
    plot_bar_chart(normalized_ranked_df,10)       
    # BOX PLOTS TO SHOW DISTRIBUTION OF clean_df and weighted_df ECONOMIC_INDICATORS
    plot_box_chart(clean_2025_df,title = 'RAW DISTRIBUTIONS OF ALL INDICATORS [2025]',file_name='boxplot_raw_indicators_stacked.png')
    plot_box_chart(weighted_df,title = 'PROJECTED WEIGHTED DISTRIBUTIONS OF ALL INDICATORS [2030]',file_name='boxplot_weighted_indicators_stacked.png')
    # global indicator trendline
    plot_global_indicator_trendline_wrapper(clean_df,slopes_with_gdp_df)

def save_summary_statistics(df, filename):
    """
    Function to print summary statistics of predictions by ECONOMIC_INDICATORS in a csv file for future reference
    @input_parameters:
    - predicted_2030_df
    - filename
    """
    summary = df[ECONOMIC_INDICATORS].describe().T
    summary.to_csv(ROOT_FOLDER + filename, index=True)
    

def main():
    #### EXTRACTING DATA FROM FILE ####
    original_df = extract_raw_data() #extracts data from ROOT_FILE

    # data cleaning
    clean_df = clean_data(original_df)

    # slope calculation using np.polyfit()
    slopes_without_gdp_df = compute_slopes_without_gdp_growth(clean_df,ECONOMIC_INDICATORS,'year','country_name').copy()
    slopes_with_gdp_df = get_slope_with_gdp_growth(clean_df,ECONOMIC_INDICATORS,'year','country_name').copy()

    # getting data frame with only the values of indicators from 2025
    clean_2025_df = get_2025_values(clean_df).copy()

    # predicting values of indicators upto 2030 and normalizing resulting values by column
    predicted_2030_df = predict_2030_values(clean_2025_df,slopes_with_gdp_df)
    normalized_2030_df = normalize_data(predicted_2030_df)

    # applying weights, scoring and ranking
    weighted_df, raw_scores_df = apply_weights_and_get_scores(normalized_2030_df, predicted_2030_df)
    ranked_df = get_ranked_index(raw_scores_df)
    normalized_ranked_df = get_normalized_scores(ranked_df)

    # wrapper function for plotting data in various formats
    plot_data_wrapper(clean_df,clean_2025_df, predicted_2030_df, weighted_df, slopes_with_gdp_df, normalized_ranked_df)

    # saving summary statsitcs in a csv file
    save_summary_statistics(predicted_2030_df,r'\summary_stats.csv')


if __name__ == "__main__":
    main()


