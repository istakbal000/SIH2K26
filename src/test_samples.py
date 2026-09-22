import pandas as pd
import json
import argparse
import logging
from pathlib import Path
import joblib

from predict import predict_dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Check the model on known samples from the dataset")
    parser.add_argument("--data", type=str, default="dataset/firDetection.csv", help="Path to data CSV with known labels")
    parser.add_argument("--n", type=int, default=10, help="Number of samples to test")
    parser.add_argument("--alarm", type=int, choices=[0, 1], default=None, help="Only test rows with this label (None = mixed)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, choices=['all', 'test'], default='test',
                        help="'test' = only the held-out chronological test portion (last 20%), 'all' = whole dataset")
    parser.add_argument("--model", type=str, default="models/best_model.pkl")
    parser.add_argument("--scaler", type=str, default="models/scaler.pkl")
    parser.add_argument("--medians", type=str, default="models/medians.json")
    args = parser.parse_args()

    df = pd.read_csv(args.data)

    if 'Fire Alarm' not in df.columns:
        logger.error("No 'Fire Alarm' column in data")
        return

    if args.split == 'test':
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.sort_values('timestamp').reset_index(drop=True)
        elif 'UTC' in df.columns:
            df['UTC'] = pd.to_numeric(df['UTC'], errors='coerce')
            df = df.sort_values('UTC').reset_index(drop=True)
        else:
            logger.warning("No 'timestamp'/'UTC' column, assuming rows are already chronological")
        n = len(df)
        cutoff = int(n * 0.8)
        df = df.iloc[cutoff:]
        logger.info(f"Using chronological TEST portion: rows {cutoff}-{n} ({len(df)} samples)")

    if args.alarm is not None:
        pool = df[df['Fire Alarm'] == args.alarm]
    else:
        pool = df

    if len(pool) < args.n:
        logger.warning(f"Only {len(pool)} rows match, using all of them")
    samples = pool.sample(n=min(args.n, len(pool)), random_state=args.seed)

    model = joblib.load(args.model)
    scaler = joblib.load(args.scaler)
    medians = json.loads(Path(args.medians).read_text())

    results = []
    for _, row in samples.iterrows():
        actual = int(row['Fire Alarm'])
        row = row.drop(labels=['Fire Alarm'])
        pred_result = predict_dict(row.to_dict(), model, scaler, medians)
        predicted = pred_result['prediction']
        results.append({
            'row': row.name,
            'actual': actual,
            'predicted': predicted,
            'confidence': pred_result['confidence'],
            'correct': actual == predicted,
        })

    res_df = pd.DataFrame(results)
    correct = res_df['correct'].sum()
    total = len(res_df)

    print("\n" + "=" * 70)
    print(f" Sample Accuracy: {correct}/{total} = {correct / total * 100:.2f}%")
    print("=" * 70)
    print(res_df.to_string(index=False))
    print()

    if args.alarm is None:
        match = res_df[res_df['correct'] == True]
        print("\nMISMATCHES (rows where model was wrong):")
        if match.empty:
            print("  none!")
        else:
            mis = res_df[res_df['correct'] == False]
            if mis.empty:
                print("  none!")
            else:
                print(mis.to_string(index=False))

if __name__ == "__main__":
    main()