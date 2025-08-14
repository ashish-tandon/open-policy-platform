#!/usr/bin/env python3
import runpy
import sys

if __name__ == "__main__":
	# Delegate to backend script
	runpy.run_path("backend/scripts/run_category.py", run_name="__main__")