# Premier League Predictor & Season Simulator

A statistical football prediction engine that uses the **Dixon-Coles Model** and **Monte Carlo Simulations** to forecast match outcomes and simulate the remainder of the Premier League season.

##  Overview

This project aims to provide data-driven insights into the Premier League by modeling team strengths (attack and defense) and simulating thousands of potential season outcomes. It accounts for home advantage, historical performance, and recent form to estimate probabilities for the title race, Top 4 finish, and relegation.

##  Key Features

- **Dixon-Coles Model**: A sophisticated Poisson-based model that accounts for the low-scoring nature of football by introducing a dependency parameter ($\rho$) for 0-0, 1-0, 0-1, and 1-1 results.
- **Monte Carlo Simulations**: Runs 10,000+ simulations of the remaining fixtures to provide probabilistic forecasts for final league positions and points.
- **Dynamic Team Ratings & Form**: Evaluates short and long-term momentum. Core attacking and defensive traits are isolated using a rolling multi-season sliding window, alongside 5-match and 10-match rolling form features (goals, xG, points).
- **xG Integration**: Scrapes Expected Goals (xG) data from **Understat** to better reflect performance quality beyond just final scores.
- **ELO Rating System**: A custom implementation scaled by a Margin of Victory multiplier and inter-season regression to track zero-sum relative strength over time.
- **Automated Data Fetching**: Scripts to fetch the latest results and upcoming fixtures for the 2025/26 season.

##  Project Structure

```text
├── data/
│   ├── raw/             # Original match data and scraped xG
│   └── processed/       # Cleaned data with engineered features
├── src/
│   ├── model.py         # Dixon-Coles model implementation
│   ├── simulator.py     # Monte Carlo season simulation engine
│   ├── elo.py           # ELO rating calculations
│   ├── features.py      # Feature engineering (rolling averages, form, etc.)
│   ├── fetch_current_season.py # Data ingestion for the 2025/26 season
│   └── understat_scraper.py   # Scraper for Understat xG metrics
├── results/             # Simulation outputs (CSV and analysis)
└── notebooks/           # Exploratory data analysis and model prototyping
```

##  How It Works

### 1. The Dixon-Coles Model
The core prediction engine uses a Poisson distribution to model goals scored by each team. Team strengths are parameterized as:
- **$\alpha_i$ (Attack)**: Ability of team $i$ to score goals.
- **$\beta_i$ (Defense)**: Ability of team $i$ to prevent goals.
- **$\gamma$ (Home Advantage)**: The constant boost given to the home team.

The probability of a score $(x, y)$ is calculated as:
$$P(X=x, Y=y) = \tau_{\rho, \lambda, \mu}(x, y) \cdot \frac{e^{-\lambda}\lambda^x}{x!} \cdot \frac{e^{-\mu}\mu^y}{y!}$$
Where $\tau$ is the Dixon-Coles adjustment factor for low-scoring matches.

### 2. Season Simulation
Once the model is fitted on historical data, it predicts the outcome probabilities (Win/Draw/Loss) for all remaining fixtures. The simulator then:
1. Picks a result for each fixture based on those probabilities.
2. Updates the league table for that specific simulation.
3. Repeats this process 10,000 times.
4. Aggregates the results to find the probability of various outcomes (e.g., "Arsenal has a 12% chance of winning the league").

### 3. Dynamic Team Ratings & Form Tracking
A team's strength and tactical setup evolve significantly across years. To prevent five-year-old form from dragging down current predictions, the core Dixon-Coles parameters are kept "dynamic" utilizing a sliding multi-season window (e.g., heavily weighting the most recent 3 seasons of play). Concurrently, the data pipeline engineers 5-match and 10-match rolling averages (for points earned, goals scored/conceded, clean sheets, and Expected Goals) to capture immediate tactical trends, injuries, or managerial changes influencing short-term form.

### 4. Custom ELO Rating System
To distill a team's true relative strength into a single continuous metric, a custom zero-sum ELO system is strictly calculated:
- **Initialization**: Every team starts at a baseline rating of 1500.
- **Expected Score**: Win probability is determined by the logistic curve $E_A = 1 / (1 + 10^{(R_B - R_A)/400})$.
- **Margin Multiplier**: Standard ELO models only care about who won. Our custom implementation scales the rating transfer by a Goal Difference multiplier: $M = \ln(|GoalDiff| + 1) \times 2.5$. Dominating a team 4-0 generates a much larger rating swing than squeaking out a 1-0 win.
- **Post-Match Update**: Ratings are exchanged using a volatility factor of $K=32$ factoring in the margin: $R_{new} = R_{old} + K \times M \times (\text{Actual} - \text{Expected})$.
- **Season Regression**: Off-season transfers and management changes heavily alter a squad. When a new season starts, every team's ELO experiences a 30% regression back to the league mean (1500) to account for these off-season equalization realities.

## Getting Started

### Prerequisites
- Python 3.8+
- Requirements: `numpy`, `pandas`, `scipy`, `sklearn`, `standard-library`

### Usage

1. **Fetch Latest Data**:
   ```bash
   python src/fetch_current_season.py
   ```

2. **Run Simulations**:
   ```bash
   python src/simulator.py
   ```

3. **Check Results**:
   Detailed CSVs and summary logs will be generated in the `results/` directory, including:
   - `simulation_results.csv`: Probabilities for Win/Top4/Relegation.
   - `relegation_details.csv`: Deep dive into the bottom-of-the-table battle.

##  Results Summary
The simulator provides a comprehensive breakdown of:
- **Title Race**: Probability of each contender lifting the trophy.
- **Champions League Race**: Estimated chances of finishing in the top 4/6.
- **Relegation Battle**: Probability of finishing in the bottom 3.
- **Expected Points**: Average points tally over all simulations.
