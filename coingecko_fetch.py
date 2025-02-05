import requests
import json
import time
import os

def fetch_coin_info(coin_id):
    """
    Fetch info for a given coin ID from the CoinGecko API.
    Returns a dict with name, symbol, and English description (if available).
    """
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            name = data.get("name", "")
            symbol = data.get("symbol", "")
            description = data.get("description", {}).get("en", "")
            return {"name": name, "symbol": symbol, "description": description}
        elif response.status_code == 429:
            print(f"Rate limit hit. Retrying {coin_id} after a delay...")
            time.sleep(5)  # Longer delay for rate limits
            return fetch_coin_info(coin_id)  # Retry fetching the same coin
        else:
            print(f"Error fetching {coin_id}: Status code {response.status_code}")
    except Exception as e:
        print(f"Error fetching {coin_id}: {e}")
    return None

if __name__ == "__main__":
    # List of coin IDs to fetch
    coin_ids = ["chainlink", "solana"]#, "polkadot", "cardano", "chainlink", "solana"]

    results = []
    for coin_id in coin_ids:
        print(f"Fetching data for {coin_id}...")
        coin_data = fetch_coin_info(coin_id)
        if coin_data:
            results.append(coin_data)
        time.sleep(5)

    # Ensure the "data" directory exists
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the fetched data to a JSON file
    output_file = os.path.join(output_dir, "cryptos.json")
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Data fetching complete! See {output_file}.")
    except Exception as e:
        print(f"Error saving data to {output_file}: {e}")
