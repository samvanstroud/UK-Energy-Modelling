import matplotlib.pyplot as plt
import pandas as pd

from src import matplotlib_style  # noqa: F401
from src.data import historical_demand
from src.units import Units as U  # noqa: F401
from tests.config import OUTPUT_DIR

OUTPUT_PATH = OUTPUT_DIR / "data" / "historical_demand"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


def test_demand_era5() -> None:
    df = historical_demand.demand_era5(resample="D")
    assert df.shape[0] > 0
    assert df.columns.tolist() == ["demand"]
    assert df["demand"].dtype == "pint[GW]"
    assert df["demand"].min() >= 0.0
    assert df.index.name == "date"
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.dtype == "datetime64[ns]"
    assert not df.index.has_duplicates


def test_demand_espeni() -> None:
    df = historical_demand.demand_espeni(resample="D")
    assert df.shape[0] > 0
    assert df.columns.tolist() == ["demand"]
    assert df["demand"].dtype == "pint[GW]"
    assert df["demand"].min() >= 0.0
    assert df.index.name == "date"
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.dtype == "datetime64[ns]"
    assert not df.index.has_duplicates


def test_demand_comparisons() -> None:
    era5_df = historical_demand.demand_era5("ME")
    espeni_df = historical_demand.demand_espeni("ME")
    era5_yearly = era5_df.resample("ME").mean().reset_index()
    espeni_yearly = espeni_df.resample("ME").mean().reset_index()
    plt.figure(figsize=(8, 4))
    plt.plot(era5_yearly["date"], era5_yearly["demand"], label="ERA5", color="blue")
    plt.plot(espeni_yearly["date"], espeni_yearly["demand"], label="ESPENI", color="orange")
    plt.title("Monthly Electricity Demand Comparison")
    plt.xlabel("Year")
    plt.ylabel("Electricity Demand (GW)")
    plt.legend()
    plt.savefig(OUTPUT_PATH / "demand_comparison.png")
    plt.close()


def test_demand_era5_weather_adjusted() -> None:
    df = historical_demand.demand_era5(resample="D", weather_adjusted=True)
    assert df.shape[0] > 0
    assert df.columns.tolist() == ["demand"]
    assert df["demand"].dtype == "pint[GW]"
    assert df["demand"].min() >= 0.0
    assert df.index.name == "date"
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.dtype == "datetime64[ns]"
    assert not df.index.has_duplicates

    # plot the weather-adjusted demand compared with the default
    default_df = historical_demand.demand_era5(resample="ME")
    adjusted_df = historical_demand.demand_era5(resample="ME", weather_adjusted=True)
    plt.figure(figsize=(8, 4))
    plt.plot(default_df.index, default_df["demand"], label="Default ERA5", color="blue")
    plt.plot(adjusted_df.index, adjusted_df["demand"], label="Weather Adjusted ERA5", color="green")
    plt.title("Monthly Electricity Demand: Default vs Weather Adjusted")
    plt.xlabel("Year")
    plt.ylabel("Electricity Demand (GW)")
    plt.legend()
    plt.savefig(OUTPUT_PATH / "demand_era5_weather_adjusted_comparison.png")
    plt.close()


def test_gas_demand() -> None:
    df = historical_demand.historical_gas_demand(old_gas_data=False, filter_ldz=True)
    assert df.shape[0] > 0
    assert df.columns.tolist() == ["demand"]
    assert df["demand"].dtype == "pint[TWh]"
    assert df["demand"].min() >= 0.0
    assert df.index.name == "date"
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.dtype == "datetime64[ns]"
    assert not df.index.has_duplicates

    df_old = historical_demand.historical_gas_demand(old_gas_data=True, filter_ldz=False)

    # plot the gas demand
    plt.figure(figsize=(8, 4))
    plt.plot(df.index, df["demand"], label="Gas Demand")
    plt.plot(df_old.index, df_old["demand"], label="Old Gas Demand")
    plt.title("Daily Gas Demand")
    plt.xlabel("Date")
    plt.ylabel("Gas Demand (TWh)")
    plt.legend()
    plt.savefig(OUTPUT_PATH / "gas_demand.png")
    plt.close()


def test_hdd_era5() -> None:
    df = historical_demand.hdd_era5()
    assert df is not None
    assert not df.empty
    assert len(df.columns) == 1
    assert df.index.dtype == "datetime64[ns]"
    assert isinstance(df, pd.DataFrame)


def test_hdd_era5_resample() -> None:
    df_raw = historical_demand.hdd_era5()
    df = historical_demand.hdd_era5(resample="ME")
    assert df is not None
    assert not df.empty
    assert len(df.columns) == 1
    assert df.index.dtype == "datetime64[ns]"
    assert isinstance(df, pd.DataFrame)
    assert len(df) < len(df_raw)


def test_hdd_era5_plot() -> None:
    df = historical_demand.hdd_era5()
    plt.figure()
    plt.plot(df.index, df["hdd"], label="HDD2")
    plt.xlabel("Date")
    plt.ylabel("ERA5 HDD")
    plt.legend()
    plt.savefig(OUTPUT_PATH / "hdd_era5.png")
    plt.close()
