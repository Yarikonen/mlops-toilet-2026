
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


