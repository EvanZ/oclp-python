"""All native Dagster asset declarations for the bike-demand example."""

from bike_demand_service.dagster.assets.bootstrap import (
    bike_demand_release_cycle,
    bike_demand_release_cycle_request,
    bike_demand_training_plan,
)
from bike_demand_service.dagster.assets.preparation import (
    bike_demand_prepare_features,
    bike_demand_raw_source,
)
from bike_demand_service.dagster.assets.release import bike_demand_model_release
from bike_demand_service.dagster.assets.training import (
    bike_demand_chart_holdout_forecast,
    bike_demand_chart_temporal_validation,
    bike_demand_evaluate_candidate,
    bike_demand_score_holdout,
    bike_demand_train_final_model,
    bike_demand_train_fold,
)

ALL_ASSETS = [
    bike_demand_release_cycle_request,
    bike_demand_release_cycle,
    bike_demand_training_plan,
    bike_demand_raw_source,
    bike_demand_prepare_features,
    bike_demand_train_fold,
    bike_demand_evaluate_candidate,
    bike_demand_chart_temporal_validation,
    bike_demand_train_final_model,
    bike_demand_score_holdout,
    bike_demand_chart_holdout_forecast,
    bike_demand_model_release,
]
