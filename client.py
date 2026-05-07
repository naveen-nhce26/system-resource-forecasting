import requests
import matplotlib.pyplot as plt


BASE_URL = "http://127.0.0.1:8000/api/v1/forecast"


def main():
    print("\n==============================")
    print(" SALES FORECASTING SYSTEM")
    print("==============================\n")

    state = input("Enter State Name: ").strip()
    weeks = input("Enter Forecast Weeks: ").strip()

    if not weeks.isdigit():
        print("\nInvalid weeks input.")
        return

    weeks = int(weeks)

    url = f"{BASE_URL}/{state}?weeks={weeks}"

    try:
        response = requests.get(url)

        if response.status_code != 200:
            print("\nAPI Error:")
            print(response.text)
            return

        data = response.json()

        print("\n========== FORECAST RESULT ==========\n")
        print(f"State           : {data['state']}")
        print(f"Forecast Weeks  : {data['forecast_weeks']}")
        print(f"Best Model      : {data['best_model']}")
        print("\nPredictions:\n")

        predictions = data["predictions"]

        for i, value in enumerate(predictions, start=1):
            print(f"Week {i}: {value:,.2f}")

        # Visualization
        weeks_list = [f"W{i}" for i in range(1, len(predictions) + 1)]

        plt.figure(figsize=(10, 5))
        plt.plot(weeks_list, predictions, marker='o')
        plt.title(f"{state} Sales Forecast")
        plt.xlabel("Weeks")
        plt.ylabel("Predicted Sales")
        plt.grid(True)

        plt.tight_layout()
        plt.savefig("forecast.png")
        print("\nForecast graph saved as forecast.png")
        plt.close()

    except Exception as e:
        print("\nError:")
        print(str(e))


if __name__ == "__main__":
    main()