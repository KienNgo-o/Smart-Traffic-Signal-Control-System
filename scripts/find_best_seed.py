import pandas as pd

def main():
    df = pd.read_csv('master_evaluation_results.csv')
    row_min = df.loc[df['avg_wait_d3qn'].idxmin()]
    row_max = df.loc[df['avg_wait_improvement_pct'].idxmax()]
    print(f"best_by_avg_wait: seed={int(row_min['seed'])}, avg_wait_d3qn={row_min['avg_wait_d3qn']:.3f}, experiment={row_min['experiment']}")
    print(f"best_by_improvement: seed={int(row_max['seed'])}, improvement_pct={row_max['avg_wait_improvement_pct']:.3f}, experiment={row_max['experiment']}")

if __name__ == '__main__':
    main()
