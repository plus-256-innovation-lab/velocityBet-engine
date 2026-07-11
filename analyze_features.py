import pandas as pd
import numpy as np

df = pd.read_csv('marble_race_data.csv')
MARBLES = [c.replace('_x', '') for c in df.columns if c.endswith('_x')]
df = df.drop(columns=['race_id', 'seed'], errors='ignore')
df = df.reset_index(drop=True)
df['race_id'] = (df['tick'].diff() < 0).cumsum() + 1

print('Races:', df['race_id'].nunique())
print('Rows:', len(df))
print('Tick resets:', int((df['tick'].diff() < 0).sum()))

# y vs z leadership at last snapshot per race
matches_y = matches_z = 0
for rid in df['race_id'].unique():
    sub = df[df['race_id'] == rid]
    w = sub['winner'].iloc[0]
    last = sub.iloc[-1]
    y_leader = min(MARBLES, key=lambda m: last[f'{m}_y'])
    z_leader = max(MARBLES, key=lambda m: last[f'{m}_z'])
    matches_y += w == y_leader
    matches_z += w == z_leader

n = df['race_id'].nunique()
print(f'Winner == last y-leader (min y): {matches_y}/{n} ({100*matches_y/n:.1f}%)')
print(f'Winner == last z-leader (max z): {matches_z}/{n} ({100*matches_z/n:.1f}%)')

# First race early rows
r1 = df[df['race_id'] == 1].head(5)
print('\nRace 1 first 5 rows - y ranks vs z ranks for Red (winner):')
for _, row in r1.iterrows():
    ys = {m: row[f'{m}_y'] for m in MARBLES}
    zs = {m: row[f'{m}_z'] for m in MARBLES}
    y_rank = pd.Series(ys).rank(ascending=True)[ 'Red']
    z_rank = pd.Series(zs).rank(ascending=False)['Red']
    y_gap = row['Red_y'] - min(ys.values())
    z_gap = max(zs.values()) - row['Red_z']
    print(f"  tick {int(row['tick'])}: Red y_rank={y_rank:.0f} y_gap={y_gap:.3f} | z_rank={z_rank:.0f} z_gap={z_gap:.3f}")

# Check y monotonicity within race
print('\nIs y always decreasing for leader? Sample race 1 y range:')
sub = df[df['race_id']==1]
for m in MARBLES[:3]:
    print(f'  {m}: y {sub[f"{m}_y"].iloc[0]:.1f} -> {sub[f"{m}_y"].iloc[-1]:.1f}, z {sub[f"{m}_z"].iloc[0]:.1f} -> {sub[f"{m}_z"].iloc[-1]:.1f}')

# Correlation of y_rank=1 with eventual winner at various progress points
def engineer_y(df, marbles):
    df = df.copy()
    y_cols = [f'{m}_y' for m in marbles]
    y_ranks = df[y_cols].rank(axis=1, ascending=True)
    for m in marbles:
        df[f'{m}_y_rank'] = y_ranks[f'{m}_y']
    leader_y = df[y_cols].min(axis=1)
    for m in marbles:
        df[f'{m}_y_gap'] = df[f'{m}_y'] - leader_y
    return df

def engineer_z(df, marbles):
    df = df.copy()
    z_cols = [f'{m}_z' for m in marbles]
    z_ranks = df[z_cols].rank(axis=1, ascending=False)
    for m in marbles:
        df[f'{m}_z_rank'] = z_ranks[f'{m}_z']
    return df

dfy = engineer_y(df, MARBLES)
dfz = engineer_z(df, MARBLES)
df = dfy.join(dfz[[f'{m}_z_rank' for m in MARBLES]])

for m in MARBLES:
    leader_y = (df[f'{m}_y_rank'] == 1).sum()
    leader_z = (df[f'{m}_z_rank'] == 1).sum()
print(f"\nRows where each marble is y-rank-1 (sample): Yellow { (df['Yellow_y_rank']==1).sum() }")

# At 50% race progress, does y-rank-1 predict winner?
df['race_progress'] = df.groupby('race_id')['tick'].transform(lambda x: (x-x.min())/(x.max()-x.min()))
mid = df[(df['race_progress'] > 0.45) & (df['race_progress'] < 0.55)]
correct_y = sum(mid.apply(lambda r: MARBLES[int(r[f"{r['winner']}_y_rank"])-1] == r['winner'] if False else (r[f"{r['winner']}_y_rank"] == 1), axis=1))
# simpler:
for label, rank_col in [('y_rank', 'y_rank'), ('z_rank', 'z_rank')]:
    hits = 0
    total = 0
    for rid in df['race_id'].unique():
        sub = df[df['race_id']==rid]
        w = sub['winner'].iloc[0]
        mid_row = sub.iloc[len(sub)//2]
        if mid_row[f'{w}_{rank_col}'] == 1:
            hits += 1
        total += 1
    print(f'Mid-race {label} leader == winner: {hits}/{total} ({100*hits/total:.1f}%)')
