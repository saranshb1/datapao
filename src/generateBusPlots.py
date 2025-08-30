
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd

def generate_analysis_plots(df: pd.DataFrame, save=True, show=True, plots=None):
    """
    Generate multiple exploratory analysis plots from a pandas DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the required columns:
        - 'timestamp'
        - 'electric_powerDemand'
        - 'busRoute'
        - 'itcs_numberOfPassengers_central'
        - 'wheelSpeed_mean'
        - 'gnss_altitude'
        - 'temperature_ambient'
    
    save : bool, default=True
        Whether to save plots as JPG files.
    
    show : bool, default=True
        Whether to display plots interactively.
    
    plots : list or None, default=None
        List of plots to generate. Options:
        ["energy_demand", "passenger_load", "speed_altitude", 
         "seasonal_trend", "temp_vs_demand"]
        If None, generate all plots.
    """
    
    if plots is None:
        plots = ["energy_demand", "passenger_load", "speed_altitude", 
                 "seasonal_trend", "temp_vs_demand"]

    # ---------------- 1. Energy Demand Profile ----------------
    if "energy_demand" in plots:
        plt.figure(figsize=(14,6))
        plt.plot(df['timestamp'], df['electric_powerDemand'], lw=0.7, color="steelblue")
        plt.axhline(0, color="red", linestyle="--", lw=1)
        plt.title("Energy Demand Profile Over Time", fontsize=16, pad=15)
        plt.xlabel("Time")
        plt.ylabel("Electric Power Demand (mW)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save: plt.savefig("energyDemandProfile.jpg", dpi=300, bbox_inches="tight")
        if show: plt.show()
        else: plt.close()

    # ---------------- 2. Passenger Load by Route ----------------
    if "passenger_load" in plots:
        plt.figure(figsize=(12,6))
        # Fixing the FutureWarning by setting `hue=None`
        sns.boxplot(data=df, x="busRoute", y="itcs_numberOfPassengers_central", 
                    palette="Set2", hue=None, legend=False)
        plt.title("Passenger Load Distribution by Bus Route", fontsize=16, pad=15)
        plt.xlabel("Bus Route")
        plt.ylabel("Central Passenger Count")
        plt.xticks(rotation=45)
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        if save: plt.savefig("passengerLoadDistribution.jpg", dpi=300, bbox_inches="tight")
        if show: plt.show()
        else: plt.close()

    # ---------------- 3. Speed vs Altitude ----------------
    if "speed_altitude" in plots:
        plt.figure(figsize=(10,6))
        hb = plt.hexbin(df['wheelSpeed_mean'], df['gnss_altitude'], 
                        gridsize=150, cmap="viridis", mincnt=1, bins="log")
        plt.colorbar(hb, label="Log(Count)")
        plt.title("Speed vs Altitude Profile (Hexbin)", fontsize=16, pad=15)
        plt.xlabel("Vehicle Speed (m/s)")
        plt.ylabel("Altitude (m)")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        if save: plt.savefig("speedvsaltitude.jpg", dpi=300, bbox_inches="tight")
        if show: plt.show()
        else: plt.close()

    # ---------------- 4. Seasonal Trend ----------------
    if "seasonal_trend" in plots:
        df['month_year'] = df['timestamp'].dt.to_period('M')
        monthly = df.groupby('month_year').agg({
            'temperature_ambient':'mean',
            'electric_powerDemand':'mean'
        }).reset_index()
        monthly['month_year_dt'] = monthly['month_year'].dt.to_timestamp()

        fig, ax1 = plt.subplots(figsize=(16,6))
        ax1.plot(monthly['month_year_dt'], monthly['temperature_ambient'], 
                 color="orange", marker="o", label="Avg Temperature (K)")
        ax1.set_ylabel("Temperature (K)", color="orange")
        ax1.tick_params(axis='y', labelcolor="orange")

        ax2 = ax1.twinx()
        ax2.plot(monthly['month_year_dt'], monthly['electric_powerDemand'], 
                 color="steelblue", marker="s", label="Avg Power Demand (mW)")
        ax2.set_ylabel("Electric Power Demand (mW)", color="steelblue")
        ax2.tick_params(axis='y', labelcolor="steelblue")

        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
        plt.title("Seasonal Trends: Temperature vs Energy Demand", fontsize=16, pad=15)
        plt.grid(True, alpha=0.3)
        fig.tight_layout()
        if save: plt.savefig("seasonalTrendTemperateVSEnergy.jpg", dpi=300, bbox_inches="tight")
        if show: plt.show()
        else: plt.close()

    # ---------------- 5. Temperature vs Energy Demand ----------------
    if "temp_vs_demand" in plots:
        plt.figure(figsize=(10,6))
        hb = plt.hexbin(df['temperature_ambient'], df['electric_powerDemand'],
                        gridsize=200, cmap="viridis", mincnt=1, bins="log")
        plt.colorbar(hb, label="Log(Count of Points)")
        plt.title("Ambient Temperature vs. Energy Demand", fontsize=16, pad=15)
        plt.xlabel("Ambient Temperature (K)")
        plt.ylabel("Electric Power Demand (mW)")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        if save: plt.savefig("temperatureEnergyHVACAnalysis.jpg", dpi=300, bbox_inches="tight")
        if show: plt.show()
        else: plt.close()
