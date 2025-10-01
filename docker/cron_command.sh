30 2 * * * cd "/path/to/repo" && docker compose exec -T api \
  ppp-ml-train-all-models --min-years 8 --split-year 2020 --horizon 10 \
  >> logs/train_$(date +\%F).log 2>&1