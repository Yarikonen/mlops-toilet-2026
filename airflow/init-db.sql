
CREATE TABLE IF NOT EXISTS raw_toilets (
    _id VARCHAR(100) PRIMARY KEY,
    Latitude DOUBLE PRECISION,
    Longitude DOUBLE PRECISION,
    Name VARCHAR(500),
    Address VARCHAR(500),
    Suburb VARCHAR(200),
    Postcode VARCHAR(20),
    "Group" VARCHAR(100),
    Category VARCHAR(100),
    Accessible VARCHAR(50),
    ChangingPlace VARCHAR(50),
    Gender VARCHAR(100),
    loaded_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);


CREATE TABLE IF NOT EXISTS toilets_grid_stats (
    lat_bin BIGINT NOT NULL,
    lon_bin BIGINT NOT NULL,
    toilet_count BIGINT,
    PRIMARY KEY (lat_bin, lon_bin)
);


CREATE TABLE IF NOT EXISTS toilets_anomalies (
    lat_bin BIGINT NOT NULL,
    lon_bin BIGINT NOT NULL,
    toilet_count BIGINT,
    z_score DOUBLE PRECISION,
    PRIMARY KEY (lat_bin, lon_bin)
);


CREATE TABLE IF NOT EXISTS toilets_hotspots (
    lat_bin BIGINT NOT NULL,
    lon_bin BIGINT NOT NULL,
    toilet_count BIGINT,
    z_score DOUBLE PRECISION,
    PRIMARY KEY (lat_bin, lon_bin)
);

CREATE TABLE IF NOT EXISTS toilets_suburb_stats (
    Suburb VARCHAR(200) NOT NULL,
    Postcode VARCHAR(20),
    avg_lat DOUBLE PRECISION,
    avg_lon DOUBLE PRECISION,
    accessibility_rate DOUBLE PRECISION,
    toilet_count BIGINT,
    PRIMARY KEY (Suburb)
);


-- =============================
-- Lab 2: ML pipeline tables
-- =============================

-- Feature table for training dataset (created/overwritten by Spark job)
CREATE TABLE IF NOT EXISTS ml_features (
    lat_bin BIGINT NOT NULL,
    lon_bin BIGINT NOT NULL,
    toilet_count BIGINT,
    avg_distance_to_center DOUBLE PRECISION,
    min_distance_to_center DOUBLE PRECISION,
    accessibility_rate DOUBLE PRECISION,
    avg_gender_encoded DOUBLE PRECISION,
    avg_category_encoded DOUBLE PRECISION,
    park_count BIGINT,
    sport_count BIGINT,
    shopping_count BIGINT,
    suburb_accessibility DOUBLE PRECISION,
    suburb_toilet_count BIGINT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (lat_bin, lon_bin)
);

-- Table for model predictions per run
CREATE TABLE IF NOT EXISTS ml_predictions (
    lat_bin BIGINT NOT NULL,
    lon_bin BIGINT NOT NULL,
    predicted_count INTEGER,
    actual_count INTEGER,
    gap INTEGER,
    model_version VARCHAR(50),
    run_id VARCHAR(100),
    predicted_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (lat_bin, lon_bin, run_id)
);

-- Table for ML run metrics
CREATE TABLE IF NOT EXISTS ml_runs (
    run_id VARCHAR(100) PRIMARY KEY,
    experiment_name VARCHAR(200),
    rmse DOUBLE PRECISION,
    mae DOUBLE PRECISION,
    r2 DOUBLE PRECISION,
    model_version VARCHAR(50),
    n_estimators INTEGER,
    max_depth INTEGER,
    learning_rate DOUBLE PRECISION,
    trained_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50)
);

CREATE INDEX IF NOT EXISTS idx_ml_predictions_run_id ON ml_predictions(run_id);


-- =============================
-- Lab 3 (extension): Accessible classification tables
-- =============================

-- Per-toilet dataset (1 row = 1 toilet)
CREATE TABLE IF NOT EXISTS ml_accessible_features (
    _id VARCHAR(100) PRIMARY KEY,
    lat_bin BIGINT,
    lon_bin BIGINT,
    distance_to_center DOUBLE PRECISION,
    gender_encoded DOUBLE PRECISION,
    category_encoded DOUBLE PRECISION,
    has_changing_place BIGINT,
    suburb_toilet_count BIGINT,
    grid_toilet_count BIGINT,
    is_accessible BIGINT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Predictions per run
CREATE TABLE IF NOT EXISTS ml_accessible_predictions (
    _id VARCHAR(100) NOT NULL,
    predicted_accessible INTEGER,
    predicted_proba DOUBLE PRECISION,
    actual_accessible INTEGER,
    model_version VARCHAR(50),
    run_id VARCHAR(100),
    predicted_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (_id, run_id)
);

-- Run metrics for accessible classification
CREATE TABLE IF NOT EXISTS ml_accessible_runs (
    run_id VARCHAR(100) PRIMARY KEY,
    experiment_name VARCHAR(200),
    accuracy DOUBLE PRECISION,
    f1 DOUBLE PRECISION,
    roc_auc DOUBLE PRECISION,
    model_version VARCHAR(50),
    trained_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50)
);

CREATE INDEX IF NOT EXISTS idx_ml_accessible_predictions_run_id ON ml_accessible_predictions(run_id);


