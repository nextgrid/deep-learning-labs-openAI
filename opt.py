import optuna


def objective(trial):
    x = trial.suggest_uniform('x', -10, 10)
    return (x - 2) ** 2


storage = optuna.storages.RedisStorage(
    url='redis://127.0.0.1:6379/db1',
)

study = optuna.create_study(storage=storage)
study.optimize(objective, n_trials=100)


print(study.best_params)
