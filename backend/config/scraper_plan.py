from typing import Dict, Any

# Canonical run plan for scrapers by category
SCRAPER_RUN_PLAN: Dict[str, Dict[str, Any]] = {
	"parliamentary": {
		"triggers": "daily@06:00",
		"expected_runtime": "15-45m",
		"dependencies": ["network", "postgres(openpolicy_scrapers)"],
		"target_database": "openpolicy_scrapers",
		"outputs": ["bills", "representatives", "committees", "debates"],
	},
	"provincial": {
		"triggers": "weekly@Mon 07:00",
		"expected_runtime": "30-90m",
		"dependencies": ["network", "postgres(openpolicy_scrapers)"],
		"target_database": "openpolicy_scrapers",
		"outputs": ["representatives", "committees"],
	},
	"municipal": {
		"triggers": "monthly@1st 08:00",
		"expected_runtime": "2-6h",
		"dependencies": ["network", "postgres(openpolicy_scrapers)"],
		"target_database": "openpolicy_scrapers",
		"outputs": ["representatives"],
	},
	"civic": {
		"triggers": "daily@06:15",
		"expected_runtime": "10-30m",
		"dependencies": ["network", "postgres(openpolicy_scrapers)"],
		"target_database": "openpolicy_scrapers",
		"outputs": ["datasets"],
	},
	"update": {
		"triggers": "every@2h",
		"expected_runtime": "5-15m",
		"dependencies": ["postgres(openpolicy_scrapers)"],
		"target_database": "openpolicy_scrapers",
		"outputs": ["housekeeping"],
	},
	"one_time": {
		"triggers": "manual",
		"expected_runtime": "variable",
		"dependencies": ["network", "postgres(openpolicy_scrapers)"],
		"target_database": "openpolicy_scrapers",
		"outputs": ["initial_load"],
	},
	"special": {
		"triggers": "manual/scheduled",
		"expected_runtime": "variable",
		"dependencies": ["network", "postgres(openpolicy_scrapers)"],
		"target_database": "openpolicy_scrapers",
		"outputs": ["special_datasets"],
	},
}